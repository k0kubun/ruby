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
    /// Number of recompile exits before invalidating the current version. See `exit_recompile`.
    pub num_exits_until_invalidate: NumExits,
    /// Sites in this ISEQ whose recompile exits fired, kept across versions. See [`ExitSites`].
    pub exit_sites: ExitSites,
    /// Number of versions allowed beyond --zjit-max-versions because a final version hit a
    /// recompile exit at a site that had never exited before. See `exit_recompile`.
    pub extra_versions: usize,
}

/// Per-site recompile exit history of an ISEQ, like HotSpot's per-bytecode `trap_state` or JSC's
/// `FrequentExitSite`. When a guard built with `Some(Recompile::Kind)` side-exits, `exit_recompile`
/// records `(insn_idx, kind)` on the payload of the ISEQ that owns the guard's frame. For a guard
/// in an inlined callee, that is the callee's payload, which is also where the inlined code's
/// profiles come from. The next compilation (of the callee, or of any ISEQ inlining it) then
/// stops speculating on that kind at that instruction and emits the generic path instead, while
/// speculating as usual everywhere else.
///
/// Like HotSpot's `PerBytecodeTrapLimit`, a site is de-speculated only once it has exited in
/// `--zjit-site-exit-threshold` different versions. The first recompile at a site re-speculates
/// with the re-profiled types, which is often right (e.g. monomorphic -> polymorphic dispatch).
#[derive(Debug, Default)]
pub struct ExitSites {
    /// Unsorted and typically tiny
    sites: Vec<ExitSite>,
}

#[derive(Debug)]
struct ExitSite {
    insn_idx: u32,
    kind: crate::hir::Recompile,
    /// Number of different versions that exited at this site
    num_versions: u8,
    /// Identity of the version that last exited at this site
    last_version: usize,
}

impl ExitSites {
    /// Record an exit of a given kind at the instruction from the version `version` (any
    /// identity of the compiled code). With `force`, de-speculate the site right away.
    /// Return true if the site has newly become de-speculated.
    pub fn record(&mut self, insn_idx: u32, kind: crate::hir::Recompile, version: usize, force: bool) -> bool {
        let threshold = get_option!(site_exit_threshold).max(1);
        let site = match self.sites.iter_mut().position(|site| site.insn_idx == insn_idx && site.kind == kind) {
            Some(i) => &mut self.sites[i],
            None => {
                self.sites.push(ExitSite { insn_idx, kind, num_versions: 0, last_version: 0 });
                self.sites.last_mut().unwrap()
            }
        };
        let before = site.num_versions;
        if site.num_versions == 0 || site.last_version != version {
            site.num_versions = site.num_versions.saturating_add(1);
            site.last_version = version;
        }
        if force {
            site.num_versions = site.num_versions.max(threshold);
        }
        before < threshold && site.num_versions >= threshold
    }

    /// Return true if recompile exits of the kind at the instruction reached the threshold
    pub fn contains(&self, insn_idx: u32, kind: crate::hir::Recompile) -> bool {
        let threshold = get_option!(site_exit_threshold).max(1);
        self.sites.iter().any(|site| site.insn_idx == insn_idx && site.kind == kind && site.num_versions >= threshold)
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }
}

impl IseqPayload {
    /// The maximum number of versions this ISEQ may have now
    pub fn version_limit(&self) -> usize {
        crate::codegen::max_iseq_versions() + self.extra_versions
    }

    fn new() -> Self {
        Self {
            profile: IseqProfile::new(),
            versions: vec![],
            was_invalidated_for_singleton_class_creation: false,
            self_is_heap_object: false,
            num_exits_until_invalidate: get_option!(num_exits_until_invalidate),
            exit_sites: ExitSites::default(),
            extra_versions: 0,
        }
    }
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

    /// JIT-to-JIT calls from the ISEQ. The IseqPayload's ISEQ is the caller of it.
    pub outgoing: Vec<IseqCallRef>,

    /// JIT-to-JIT calls to the ISEQ. The IseqPayload's ISEQ is the callee of it.
    pub incoming: Vec<IseqCallRef>,
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
            outgoing: vec![],
            incoming: vec![],
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
