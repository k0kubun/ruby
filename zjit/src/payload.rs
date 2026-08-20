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
    /// Extra compiled versions this ISEQ has been granted so that a frozen `invokeblock`
    /// dispatch can pick up block handlers its profile never saw. Only
    /// [`crate::profile::rb_zjit_block_reprofile`] grants these, and only against evidence from
    /// the generic fallback path. Capped at [`MAX_BLOCK_RESPECIALIZATIONS`].
    pub block_respecializations: u8,
    /// Whether a `yield` fallback in this ISEQ has spent a compiled version's worth of
    /// re-profiling windows without earning a recompile. See [`ivar_reprofile_giveup`]: sampling
    /// costs a non-leaf call on the fallback path, so once the evidence says a recompile would
    /// not help, later compiles of this ISEQ leave the sampling out.
    ///
    /// [`ivar_reprofile_giveup`]: IseqPayload::ivar_reprofile_giveup
    pub block_reprofile_giveup: bool,
}

/// How many extra versions a single ISEQ may earn for ivar shape respecialization.
/// Each one strictly adds a shape to a dispatch that was previously falling back, so the
/// process terminates on its own; the cap bounds code growth for an ISEQ whose receivers
/// keep changing shape.
pub const MAX_IVAR_RESPECIALIZATIONS: u8 = 2;

/// How many extra versions a single ISEQ may earn for `invokeblock` handler respecialization.
///
/// The shared core iterators this exists for compile once during boot, from a profile of their
/// first few yields, and then never see a reason to recompile: their dispatch does not
/// side-exit, it branches to `rb_vm_invokeblock()`. Three is what it takes to converge on a
/// Rails boot: one window closes on the loader's traffic, one on the framework's, and one on the
/// application's. Past that a site is genuinely handler-polymorphic and more versions would not
/// help.
pub const MAX_BLOCK_RESPECIALIZATIONS: u8 = 3;

impl IseqPayload {
    fn new() -> Self {
        Self {
            profile: IseqProfile::new(),
            versions: vec![],
            was_invalidated_for_singleton_class_creation: false,
            self_is_heap_object: false,
            ivar_respecializations: 0,
            ivar_reprofile_giveup: false,
            block_respecializations: 0,
            block_reprofile_giveup: false,
        }
    }

    /// Number of versions this ISEQ may compile, including any it earned by proving from
    /// its ivar fallback path that a recompile would specialize a shape it is missing.
    pub fn version_limit(&self) -> usize {
        crate::codegen::max_iseq_versions()
            + self.ivar_respecializations as usize
            + self.block_respecializations as usize
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
    /// Re-profiling windows this version's `invokeblock` fallback paths may still close without
    /// earning a recompile before going dormant. Same trade as [`Self::ivar_reprofile_windows`]:
    /// sampling is a C call on a path that would otherwise just be a call to
    /// `rb_vm_invokeblock()`, and a version whose fallbacks keep failing to make the case for a
    /// recompile stops paying for it.
    pub block_reprofile_windows: u8,
    /// Fallback executions this version still has to skip before it samples another block
    /// handler. Decremented in JIT code; written by
    /// [`crate::profile::rb_zjit_block_reprofile`], which sets it to 0 while a window is open
    /// and to [`BLOCK_REPROFILE_COOLDOWN`] when the windows run out.
    ///
    /// Unlike the ivar path's one-shot give-up, this goes dormant rather than silent. What a
    /// shared iterator's `yield` is handed changes completely between boot and steady state --
    /// `Array#each` spends boot yielding to RubyGems' `&:strip!` and the rest of the process
    /// yielding to the application's blocks -- so a site that gave up on the first mix it saw
    /// would stay frozen on it. The cooldown makes each verdict provisional at a cost of one
    /// sample per [`BLOCK_REPROFILE_COOLDOWN`] fallbacks.
    ///
    /// Raced between threads: it is a plain load/store, and a lost decrement only delays or
    /// duplicates a sample.
    pub block_reprofile_countdown: u32,
}

/// How many windows a `yield` fallback may close without earning a recompile before the version
/// stops sampling until its cooldown expires. See [`MAX_IVAR_REPROFILE_WINDOWS`].
pub const MAX_BLOCK_REPROFILE_WINDOWS: u8 = 4;

/// Fallback executions a dormant `yield` site skips before it opens another re-profiling window.
/// Large enough that the sampling call is under a thousandth of the fallbacks at the hottest
/// site in a lobsters run, small enough that a site notices a new steady state within it.
pub const BLOCK_REPROFILE_COOLDOWN: u32 = 100_000;

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
            block_reprofile_windows: MAX_BLOCK_REPROFILE_WINDOWS,
            block_reprofile_countdown: 0,
        };
        let version_ptr = Box::into_raw(Box::new(version));
        NonNull::new(version_ptr).expect("no null from Box")
    }
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

    unsafe {
        let payload = rb_iseq_get_jit_payload(iseq);
        if payload.is_null() {
            // Allocate a new payload with Box and transfer ownership to the GC.
            // We drop the payload with Box::from_raw when the GC frees the ISEQ and calls us.
            // NOTE(alan): Sometimes we read from an ISEQ without ever writing to it.
            // We allocate in those cases anyways.
            let new_payload = IseqPayload::new();
            let new_payload = Box::into_raw(Box::new(new_payload));
            crate::stats::incr_counter!(allocated_iseq_payload_count);
            rb_iseq_set_jit_payload(iseq, new_payload as VoidPtr);

            new_payload
        } else {
            payload as *mut IseqPayload
        }
    }
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
