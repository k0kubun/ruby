use std::ffi::c_void;
use std::ptr::NonNull;
use crate::codegen::IseqCallRef;
use crate::options::{get_option, NumExits};
use crate::stats::CompileError;
use crate::{cruby::*, profile::IseqProfile, virtualmem::CodePtr};

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
    /// The lexical scope (`rb_cref_t`) of the frame that triggered the latest compilation
    /// of this ISEQ, set at compile triggers like `self_is_heap_object`. Recorded with
    /// folded constant paths so that constant definitions can re-resolve them instead of
    /// invalidating every constant of the same name. Marked (pinned) by [`crate::gc`].
    pub compile_cref: Option<VALUE>,
    /// Number of recompile exits before invalidating the current version. See `exit_recompile`.
    pub num_exits_until_invalidate: NumExits,
    /// Memoized result of [`crate::codegen::iseq_may_write_block_code`]. The bare
    /// opcodes of an ISEQ never change, so this is computed at most once.
    pub may_write_block_code: Option<bool>,
    /// Whether the interpreter's back-edges, rather than calls, opened the profiling window
    /// of this ISEQ (see `rb_zjit_iseq_backedge`), i.e. it runs long loops but is called rarely.
    pub hot_loop: bool,
    /// The GC count (`rb_gc_count()`) when the interpreter last started profiling this ISEQ,
    /// i.e. at the start of the profiling window before a compile. See [`Self::is_lukewarm`].
    pub profile_start_gc_count: usize,
    /// Number of returns and loop back-edges left before the current version, if it's
    /// lukewarm, is invalidated to recompile it with full inlining. Decremented by the
    /// lukewarm version's code. See `exit_recompile`.
    pub tier_up_counter: i64,
    /// Whether a lukewarm version of this ISEQ got hot. Later versions are not lukewarm.
    pub tiered_up: bool,
}

impl IseqPayload {
    fn new() -> Self {
        Self {
            profile: IseqProfile::new(),
            versions: vec![],
            was_invalidated_for_singleton_class_creation: false,
            self_is_heap_object: false,
            compile_cref: None,
            num_exits_until_invalidate: get_option!(num_exits_until_invalidate),
            may_write_block_code: None,
            hot_loop: false,
            profile_start_gc_count: 0,
            tier_up_counter: i64::MAX,
            tiered_up: false,
        }
    }

    /// Maximum number of versions of this ISEQ. Tiering up from a lukewarm version doesn't use
    /// up a version that could fix a failed speculation.
    pub fn max_versions(&self) -> usize {
        crate::codegen::max_iseq_versions() + self.tiered_up as usize
    }

    /// Whether the ISEQ about to be compiled is lukewarm: it reached the call threshold, but
    /// its last `--zjit-num-profiles` calls were spread over `--zjit-lukewarm-gc-count` or more
    /// GCs, so it's called rarely. Inlining into such an ISEQ multiplies its compile time, which
    /// is never paid back, and makes it recompile when a guard of a callee shared by many callers
    /// fails. GCs are counted because they measure how much the program ran without counting
    /// compile time, and are deterministic.
    ///
    /// This is like how HotSpot keeps lukewarm methods out of its expensive tier: its invocation
    /// counters decay over time (`-XX:CounterHalfLifeTime`), so only methods called frequently
    /// enough get there. ISEQs that loops made hot are never lukewarm. Like HotSpot's tier 3
    /// code, lukewarm code counts how often it runs, and an ISEQ that got hot after its profiling
    /// window is recompiled once with full inlining.
    pub fn is_lukewarm(&self) -> bool {
        let gc_count = get_option!(lukewarm_gc_count);
        gc_count > 0 && !self.hot_loop && !self.tiered_up
            && unsafe { rb_gc_count() }.wrapping_sub(self.profile_start_gc_count) >= gc_count
    }
}

/// Remember when the profiling window of an ISEQ started. See [`IseqPayload::is_lukewarm`].
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_iseq_profile_started(iseq: IseqPtr) {
    // The interpreter calls this without the VM lock, which other Ractors need to be
    // stopped by before creating the payload.
    with_vm_lock(src_loc!(), || {
        get_or_create_iseq_payload(iseq).profile_start_gc_count = unsafe { rb_gc_count() };
    });
}

/// Remember that back-edges made an ISEQ hot. See [`IseqPayload::hot_loop`].
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_iseq_mark_hot_loop(iseq: IseqPtr) {
    get_or_create_iseq_payload(iseq).hot_loop = true;
}

/// JIT code version. When the same ISEQ is compiled with a different assumption, a new version is created.
#[derive(Debug)]
pub struct IseqVersion {
    /// ISEQ pointer. Stored here to minimize the size of PatchPoint.
    pub iseq: IseqPtr,

    /// Compilation status of the ISEQ. It has the JIT code address of the first block if Compiled.
    pub status: IseqStatus,

    /// GC offsets of the JIT code. These are the addresses of objects that need to be marked.
    pub gc_offsets: Vec<CodePtr>,

    /// Byte range of the side-exit descriptors registered for the JIT code. The GC
    /// marks and moves the objects they refer to, like it does for `gc_offsets`.
    pub exit_descs: std::ops::Range<u32>,

    /// JIT-to-JIT calls from the ISEQ. The IseqPayload's ISEQ is the caller of it.
    pub outgoing: Vec<IseqCallRef>,

    /// JIT-to-JIT calls to the ISEQ. The IseqPayload's ISEQ is the callee of it.
    pub incoming: Vec<IseqCallRef>,

    /// Objects referenced by this version's patch points (crefs and expected values of
    /// `StableConstantNames`). They're pinned by [`crate::gc`] because the patch points
    /// hold them as raw VALUEs.
    pub patch_point_objects: Vec<VALUE>,

    /// Per-call-site caches of a receiver class that the JIT code writes at run time. See
    /// [`crate::hir::Function::klass_caches`]. Marked and moved by [`crate::gc`].
    pub klass_caches: Vec<Box<VALUE>>,
}

/// We use a raw pointer instead of Rc to save space for refcount
pub type IseqVersionRef = NonNull<IseqVersion>;

impl IseqVersion {
    /// Check if this version was invalidated
    pub fn is_invalidated(&self) -> bool {
        self.status == IseqStatus::Invalidated
    }

    /// Allocate a new IseqVersion to be compiled
    pub fn new(iseq: IseqPtr) -> IseqVersionRef {
        let version = Self {
            iseq,
            status: IseqStatus::NotCompiled,
            gc_offsets: vec![],
            exit_descs: 0..0,
            outgoing: vec![],
            incoming: vec![],
            patch_point_objects: vec![],
            klass_caches: vec![],
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
            rb_iseq_set_jit_payload(iseq, new_payload as VoidPtr);

            new_payload
        } else {
            payload as *mut IseqPayload
        }
    }
}

/// Get a pointer to the payload object associated with an ISEQ, or null if never allocated.
pub fn get_iseq_payload_ptr(iseq: IseqPtr) -> *mut IseqPayload {
    unsafe { rb_iseq_get_jit_payload(iseq) as *mut IseqPayload }
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
