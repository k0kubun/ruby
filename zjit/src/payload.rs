use std::ffi::c_void;
use std::ptr::NonNull;
use crate::codegen::IseqCallRef;
use crate::stats::CompileError;
use crate::{cruby::*, profile::IseqProfile, virtualmem::CodePtr};
use crate::options::get_option;

pub use crate::jit_frame::JITFrame;

/// This is all the data ZJIT stores on an ISEQ. We mark objects in this struct on GC.
#[derive(Debug)]
pub struct IseqPayload {
    /// Type information of YARV instruction operands
    pub profile: IseqProfile,
    /// JIT code versions. Different versions should have different assumptions.
    pub versions: Vec<IseqVersionRef>,
    /// JIT code versions compiled for `body->jit_exception`, which enter the ISEQ
    /// at a catch-table continuation. Kept separate from `versions` because they
    /// are not usable as the ISEQ's ordinary entry point.
    pub exception_versions: Vec<IseqVersionRef>,
    /// The continuations `exception_versions` were compiled for, one per version
    /// that compiled successfully.
    pub exception_entries: Vec<ExceptionEntryCode>,
    /// Head of the dispatch chain over `exception_entries`, which is what
    /// `body->jit_exception` points at. Rebuilt from scratch after invalidation.
    pub exception_dispatch: Option<CodePtr>,
    /// Whether a previous compilation of this ISEQ was invalidated due to
    /// singleton class creation (violation of [`crate::hir::Invariant::NoSingletonClass`]).
    pub was_invalidated_for_singleton_class_creation: bool,
    /// Whether `self` is guaranteed to be a heap (non-immediate) object for this
    /// ISEQ. Set at compile triggers (entry point / function stub hit) where the
    /// owning class is known via the method entry, and consumed in `iseq_to_hir`
    /// to type the `self`-producing instructions (`LoadSelf` / `SelfParam`
    /// `LoadArg`) as `HeapBasicObject`. Defaults to `false` (the conservative
    /// `BasicObject`) when the owner is unknown.
    /// See [`crate::cruby::iseq_self_is_heap_object`].
    pub self_is_heap_object: bool,
    /// Extra compiled versions this ISEQ has been granted on top of `--zjit-max-versions`
    /// so that a frozen ivar dispatch can pick up a shape its profile never saw. Only
    /// [`crate::profile::rb_zjit_ivar_reprofile`] grants these, and only against evidence
    /// from the fallback path. Capped at [`MAX_IVAR_RESPECIALIZATIONS`].
    pub ivar_respecializations: u8,
    /// Whether an ivar fallback in this ISEQ has spent a compiled version's worth of
    /// re-profiling windows without earning a recompile. Sampling costs a non-leaf call on the
    /// fallback path, so once the evidence says a recompile would not help, later compiles of
    /// this ISEQ leave the sampling out.
    pub ivar_reprofile_giveup: bool,
    /// Number of extra versions granted because a PatchPoint invalidation would
    /// otherwise have left this ISEQ permanently side-exiting. See
    /// [`crate::codegen::invalidate_iseq_version`].
    pub invalidation_recompiles: u8,
    /// Memoized [`crate::codegen::iseq_may_write_block_code`]. Answering it scans the
    /// whole ISEQ once per question, and every JITFrame we build asks (there is one per
    /// GC-able call site), so recomputing makes compiling an ISEQ quadratic in its size.
    /// The answer only depends on which bare opcodes the ISEQ contains, which never
    /// changes after the ISEQ is compiled, so it is safe to cache for the ISEQ's lifetime.
    pub may_write_block_code: Option<bool>,
    /// Memoized [`crate::codegen::iseq_may_expose_locals`]. Cached for the same reason
    /// as `may_write_block_code`: answering it scans the whole ISEQ, and every call site
    /// in the ISEQ asks.
    pub may_expose_locals: Option<bool>,
    /// Whether this ISEQ is sitting in the background compile queue. Dedupes
    /// enqueues: the interpreter keeps calling (and keeps incrementing
    /// `jit_entry_calls`) while the request waits, and a JIT-to-JIT stub may hit
    /// the same ISEQ meanwhile. Cleared when the compile thread takes it off the
    /// queue. See [`crate::bgcompile`].
    pub bg_queued: bool,
}

/// How many extra versions a single ISEQ may earn for ivar shape respecialization.
/// Each one strictly adds a shape to a dispatch that was previously falling back, so the
/// process terminates on its own; the cap bounds code growth for an ISEQ whose receivers
/// keep changing shape.
pub const MAX_IVAR_RESPECIALIZATIONS: u8 = 2;

/// Upper bound on the extra versions [`IseqPayload::invalidation_recompiles`] can grant.
/// Invalidation is an external event (a constant or method was redefined), not a
/// mis-speculation, so it should not consume the respecialization budget. We still cap
/// the total so that an ISEQ whose assumptions keep getting busted cannot recompile forever.
pub const MAX_INVALIDATION_RECOMPILES: u8 = 8;

impl IseqPayload {
    fn new() -> Self {
        Self {
            profile: IseqProfile::new(),
            versions: vec![],
            exception_versions: vec![],
            exception_entries: vec![],
            exception_dispatch: None,
            was_invalidated_for_singleton_class_creation: false,
            self_is_heap_object: false,
            ivar_respecializations: 0,
            ivar_reprofile_giveup: false,
            invalidation_recompiles: 0,
            may_write_block_code: None,
            may_expose_locals: None,
            bg_queued: false,
        }
    }

    /// Every compiled version of this ISEQ, ordinary entries and exception
    /// handler entries alike. GC callbacks must visit all of them.
    pub fn all_versions(&self) -> impl Iterator<Item = &IseqVersionRef> {
        self.versions.iter().chain(self.exception_versions.iter())
    }

    /// Mutable counterpart of [`IseqPayload::all_versions`]
    pub fn all_versions_mut(&mut self) -> impl Iterator<Item = &mut IseqVersionRef> {
        self.versions.iter_mut().chain(self.exception_versions.iter_mut())
    }

    /// Number of versions this ISEQ may compile: `--zjit-max-versions`, plus any it earned
    /// by proving from its ivar fallback path that a recompile would specialize a shape it
    /// is missing, plus one for each version killed by PatchPoint invalidation so that dead
    /// code does not eat into the budget meant for respecialization.
    pub fn version_limit(&self) -> usize {
        crate::codegen::max_iseq_versions()
            + self.ivar_respecializations as usize
            + self.invalidation_recompiles as usize
    }

    /// Profile counts are used for compilation policy.
    /// When we deoptimize a method that can be recompiled, we need to update the count to collect more profiles.
    /// Otherwise, we will generate the same code that was just deoptimized.
    pub fn reset_profiles_remaining(&mut self, insn_idx: YarvInsnIdx) {
        let num_profiles = get_option!(num_profiles);
        self.profile.entry_mut(insn_idx).set_profiles_remaining(num_profiles);
    }
}

/// JIT code version. When the same ISEQ is compiled with a different assumption, a new version is created.
#[derive(Debug)]
pub struct IseqVersion {
    /// ISEQ pointer. Stored here to minimize the size of PatchPoint.
    pub iseq: IseqPtr,

    /// Compilation status of the ISEQ. It has the JIT code address of the first block if Compiled.
    pub status: IseqStatus,

    /// The objects ZJIT baked into this version's JIT code, and where in the code
    /// region each one sits. See [`crate::gc::GcOffsets`].
    pub gc_offsets: crate::gc::GcOffsets,

    /// JIT-to-JIT calls from the ISEQ. The IseqPayload's ISEQ is the caller of it.
    pub outgoing: Vec<IseqCallRef>,

    /// JIT-to-JIT calls to the ISEQ. The IseqPayload's ISEQ is the callee of it.
    pub incoming: Vec<IseqCallRef>,

    /// Re-profiling windows this version's ivar fallback paths may still close without earning a
    /// recompile. See [`crate::profile::rb_zjit_ivar_reprofile`]: sampling is a C call on a path
    /// that is otherwise exit-free, so a version whose fallbacks keep failing to make the case
    /// for a recompile stops paying for the evidence.
    pub ivar_reprofile_windows: u8,
}

/// How many windows an ivar fallback may close without earning a recompile before the version
/// stops sampling. A fallback that has handed the same unspecializable mix of shapes to this
/// many windows in a row is not about to change its mind, and every sample after that is a call
/// on a hot path buying nothing.
pub const MAX_IVAR_REPROFILE_WINDOWS: u8 = 4;

/// We use a raw pointer instead of Rc to save space for refcount
pub type IseqVersionRef = NonNull<IseqVersion>;

impl IseqVersion {
    /// Bytes the JIT-to-JIT call bookkeeping of this version owns on the Rust
    /// heap: the incoming and outgoing edge vectors, plus the `IseqCall`
    /// allocations themselves. Each `IseqCall` is created by its caller and
    /// pushed onto that caller's `outgoing`, so counting only `outgoing`
    /// attributes every allocation exactly once.
    pub fn iseq_call_heap_size(&self) -> usize {
        use crate::codegen::IseqCall;
        // Rc<T> allocates two counters ahead of the value.
        let rc_bytes = 2 * size_of::<usize>() + size_of::<IseqCall>();
        self.outgoing.capacity() * size_of::<IseqCallRef>()
            + self.incoming.capacity() * size_of::<IseqCallRef>()
            + self.outgoing.len() * rc_bytes
    }

    /// Bytes the JIT entry point table in [`IseqStatus::Compiled`] owns. One
    /// entry per `opt_table` arity a JIT-to-JIT caller may enter through.
    pub fn jit_entry_heap_size(&self) -> usize {
        match &self.status {
            IseqStatus::Compiled(code_ptrs) => code_ptrs.jit_entry_ptrs.capacity() * size_of::<CodePtr>(),
            _ => 0,
        }
    }

    /// Every byte this version owns, including the `IseqVersion` allocation
    /// itself. Used both by the `mem_*` walker and to account versions that
    /// outlive their ISEQ (see [`crate::gc::rb_zjit_iseq_free`]).
    pub fn total_heap_size(&self) -> usize {
        size_of::<IseqVersion>()
            + self.gc_offsets.heap_size()
            + self.jit_entry_heap_size()
            + self.iseq_call_heap_size()
    }

    /// Check if this version was invalidated
    pub fn is_invalidated(&self) -> bool {
        self.status == IseqStatus::Invalidated
    }

    /// Allocate a new IseqVersion to be compiled
    pub fn new(iseq: IseqPtr) -> IseqVersionRef {
        let version = Self {
            iseq,
            status: IseqStatus::NotCompiled,
            gc_offsets: Default::default(),
            outgoing: vec![],
            incoming: vec![],
            ivar_reprofile_windows: MAX_IVAR_REPROFILE_WINDOWS,
        };
        let version_ptr = Box::into_raw(Box::new(version));
        NonNull::new(version_ptr).expect("no null from Box")
    }
}

/// A compiled exception handler entry and the continuation PC it was compiled
/// for. `body->jit_exception` dispatches on `cfp->pc` over these.
#[derive(Clone, Copy, Debug)]
pub struct ExceptionEntryCode {
    /// Continuation the interpreter resumes at, i.e. the `cfp->pc` this entry expects
    pub pc: *const VALUE,
    /// Machine code address of the entry
    pub code_ptr: CodePtr,
}

/// Set of CodePtrs for an ISEQ
#[derive(Clone, Debug, PartialEq)]
pub struct IseqCodePtrs {
    /// Entry for the interpreter
    pub start_ptr: CodePtr,
    /// Entries for JIT-to-JIT calls
    pub jit_entry_ptrs: Vec<CodePtr>,
}

#[derive(Debug, PartialEq)]
pub enum IseqStatus {
    Compiled(IseqCodePtrs),
    CantCompile(CompileError),
    NotCompiled,
    Invalidated,
}

/// Get a pointer to the payload object associated with an ISEQ. Create one if none exists.
pub fn get_or_create_iseq_payload_ptr(iseq: IseqPtr) -> *mut IseqPayload {
    type VoidPtr = *mut c_void;

    crate::bgcompile::assert_gvl_held("get_or_create_iseq_payload");

    unsafe {
        let payload = rb_iseq_get_zjit_payload(iseq);
        if payload.is_null() {
            // Allocate a new payload with Box and transfer ownership to the GC.
            // We drop the payload with Box::from_raw when the GC frees the ISEQ and calls us.
            // NOTE(alan): Sometimes we read from an ISEQ without ever writing to it.
            // We allocate in those cases anyways.
            let new_payload = IseqPayload::new();
            let new_payload = Box::into_raw(Box::new(new_payload));
            crate::stats::incr_counter!(allocated_iseq_payload_count);
            rb_iseq_set_zjit_payload(iseq, new_payload as VoidPtr);

            new_payload
        } else {
            payload as *mut IseqPayload
        }
    }
}

/// Get a pointer to the payload object associated with an ISEQ, or null if ZJIT
/// has never allocated one. Unlike [`get_or_create_iseq_payload_ptr`] this never
/// allocates, which matters on paths that only want to release what is already
/// there (see [`crate::gc::rb_zjit_iseq_free`]).
pub fn get_iseq_payload_ptr(iseq: IseqPtr) -> *mut IseqPayload {
    unsafe { rb_iseq_get_zjit_payload(iseq) as *mut IseqPayload }
}

/// Get the payload object associated with an ISEQ. Create one if none exists.
pub fn get_or_create_iseq_payload(iseq: IseqPtr) -> &'static mut IseqPayload {
    let payload_non_null = get_or_create_iseq_payload_ptr(iseq);
    payload_ptr_as_mut(payload_non_null)
}

/// Convert an IseqPayload pointer to a mutable reference. Only one reference
/// should be kept at a time.
pub fn payload_ptr_as_mut(payload_ptr: *mut IseqPayload) -> &'static mut IseqPayload {
    // SAFETY: we should have the VM lock and all other Ruby threads should be asleep. So we have
    // exclusive mutable access.
    // Hmm, nothing seems to stop calling this on the same
    // iseq twice, though, which violates aliasing rules.
    unsafe { payload_ptr.as_mut() }.unwrap()
}
