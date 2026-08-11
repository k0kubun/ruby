//! Profiler for runtime information.

// We use the YARV bytecode constants which have a CRuby-style name
#![allow(non_upper_case_globals)]

use std::collections::HashMap;
use crate::{cruby::*, payload::{get_or_create_iseq_payload, IseqVersion}, options::{get_option, NumProfiles}};
use crate::distribution::{Distribution, DistributionSummary};
use crate::stats::{Counter, incr_counter_by};
use crate::stats::Counter::profile_time_ns;
use crate::stats::with_time_stat;

/// Ephemeral state for profiling runtime information
struct Profiler {
    cfp: CfpPtr,
    iseq: IseqPtr,
    insn_idx: YarvInsnIdx,
}

impl Profiler {
    fn new(ec: EcPtr) -> Self {
        let cfp = unsafe { get_ec_cfp(ec) };
        let iseq = unsafe { get_cfp_iseq(cfp) };
        Profiler {
            cfp,
            iseq,
            insn_idx: unsafe { get_cfp_pc(cfp).offset_from(get_iseq_body_iseq_encoded(iseq)) as usize },
        }
    }

    // Get an instruction operand that sits next to the opcode at PC.
    fn insn_opnd(&self, idx: usize) -> VALUE {
        unsafe { get_cfp_pc(self.cfp).add(1 + idx).read() }
    }

    // Peek at the nth topmost value on the Ruby stack.
    // Returns the topmost value when n == 0.
    fn peek_at_stack(&self, n: isize) -> VALUE {
        unsafe {
            let sp: *mut VALUE = get_cfp_sp(self.cfp);
            *(sp.offset(-1 - n))
        }
    }

    fn peek_at_self(&self) -> VALUE {
        unsafe { rb_get_cfp_self(self.cfp) }
    }

    fn peek_at_block_handler(&self) -> VALUE {
        unsafe { rb_vm_get_untagged_block_handler(self.cfp) }
    }
}

/// API called from zjit_* instruction. opcode is the bare (non-zjit_*) instruction.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_profile_insn(bare_opcode: u32, ec: EcPtr) {
    with_vm_lock(src_loc!(), || {
        with_time_stat(profile_time_ns, || profile_insn(bare_opcode as ruby_vminsn_type, ec));
    });
}

/// Profile a YARV instruction
fn profile_insn_sample(
    bare_opcode: ruby_vminsn_type,
    profiler: &mut Profiler,
    profile: &mut IseqProfile,
) -> bool {
    match bare_opcode {
        YARVINSN_opt_nil_p => profile_operands(profiler, profile, 1),
        YARVINSN_opt_plus  => profile_operands(profiler, profile, 2),
        YARVINSN_opt_minus => profile_operands(profiler, profile, 2),
        YARVINSN_opt_mult  => profile_operands(profiler, profile, 2),
        YARVINSN_opt_div   => profile_operands(profiler, profile, 2),
        YARVINSN_opt_mod   => profile_operands(profiler, profile, 2),
        YARVINSN_opt_eq    => profile_operands(profiler, profile, 2),
        YARVINSN_opt_neq   => profile_operands(profiler, profile, 2),
        YARVINSN_opt_lt    => profile_operands(profiler, profile, 2),
        YARVINSN_opt_le    => profile_operands(profiler, profile, 2),
        YARVINSN_opt_gt    => profile_operands(profiler, profile, 2),
        YARVINSN_opt_ge    => profile_operands(profiler, profile, 2),
        YARVINSN_opt_and   => profile_operands(profiler, profile, 2),
        YARVINSN_opt_or    => profile_operands(profiler, profile, 2),
        YARVINSN_opt_empty_p => profile_operands(profiler, profile, 1),
        YARVINSN_opt_aref  => profile_operands(profiler, profile, 2),
        YARVINSN_opt_ltlt  => profile_operands(profiler, profile, 2),
        YARVINSN_opt_aset  => profile_operands(profiler, profile, 3),
        YARVINSN_opt_not   => profile_operands(profiler, profile, 1),
        YARVINSN_getinstancevariable => profile_self(profiler, profile),
        YARVINSN_setinstancevariable => profile_self(profiler, profile),
        YARVINSN_definedivar   => profile_self(profiler, profile),
        YARVINSN_opt_regexpmatch2    => profile_operands(profiler, profile, 2),
        YARVINSN_objtostring   => profile_operands(profiler, profile, 1),
        YARVINSN_opt_length    => profile_operands(profiler, profile, 1),
        YARVINSN_opt_size      => profile_operands(profiler, profile, 1),
        YARVINSN_opt_succ      => profile_operands(profiler, profile, 1),
        YARVINSN_invokeblock   => profile_block_handler(profiler, profile),
        YARVINSN_getblockparamproxy => profile_getblockparamproxy(profiler, profile),
        YARVINSN_invokesuper   => profile_invokesuper(profiler, profile),
        YARVINSN_opt_send_without_block | YARVINSN_send => {
            let cd: *const rb_call_data = profiler.insn_opnd(0).as_ptr();
            let argc = num_arguments_on_stack(cd);
            // Profile all the arguments and self (+1).
            profile_operands(profiler, profile, argc + 1);
            profile_splat_length(profiler, profile, unsafe { (*cd).ci });
        }
        YARVINSN_splatkw => profile_operands(profiler, profile, 2),
        _ => return false,
    }

    true
}

/// Profile a YARV instruction
fn profile_insn(bare_opcode: ruby_vminsn_type, ec: EcPtr) {
    let profiler = &mut Profiler::new(ec);
    let profile = &mut get_or_create_iseq_payload(profiler.iseq).profile;
    let _ = profile_insn_sample(bare_opcode, profiler, profile);

    // Once we profile the instruction enough times, we stop profiling it.
    let entry = profile.entry_mut(profiler.insn_idx);
    entry.profiles_remaining = entry.profiles_remaining.saturating_sub(1);
    if entry.profiles_remaining == 0 {
        unsafe { rb_zjit_iseq_insn_set(profiler.iseq, profiler.insn_idx as u32, bare_opcode); }
    }
}

/// Reset existing profile counters and install profiling instructions throughout an ISEQ.
/// Newly reached instructions initialize their counters from the same option.
pub(crate) fn reset_profiles_remaining(iseq: IseqPtr) {
    let profile = &mut get_or_create_iseq_payload(iseq).profile;
    let num_profiles = get_option!(num_profiles);
    for entry in &mut profile.entries {
        entry.profiles_remaining = num_profiles;
    }
    unsafe { rb_zjit_profile_enable(iseq) };
}

/// Return the argc as stated in the calldata plus:
/// * 1 if there is an explicit blockarg, since that will be passed on the stack
pub fn num_arguments_on_stack(cd: *const rb_call_data) -> usize {
    let ci = unsafe { (*cd).ci };
    let flags = unsafe { rb_vm_ci_flag(ci) };
    let has_blockarg = (flags & VM_CALL_ARGS_BLOCKARG) != 0;
    (unsafe { vm_ci_argc(ci) }) as usize + has_blockarg as usize
}

const DISTRIBUTION_SIZE: usize = 8;

pub type TypeDistribution = Distribution<ProfiledType, DISTRIBUTION_SIZE>;

pub type TypeDistributionSummary = DistributionSummary<ProfiledType, DISTRIBUTION_SIZE>;

pub type SplatLength = u32;

/// `None` records an unknown length so this distribution covers the same
/// executions as the operand type profile.
pub type SplatLengthDistribution = Distribution<Option<SplatLength>, DISTRIBUTION_SIZE>;

pub type SplatLengthDistributionSummary = DistributionSummary<Option<SplatLength>, DISTRIBUTION_SIZE>;

/// Profile the Type of top-`n` stack operands
fn profile_operands(profiler: &mut Profiler, profile: &mut IseqProfile, n: usize) {
    let entry = profile.entry_mut(profiler.insn_idx);
    if entry.opnd_types.is_empty() {
        entry.opnd_types.resize(n, TypeDistribution::new());
    }

    for (i, profile_type) in entry.opnd_types.iter_mut().enumerate() {
        let obj = profiler.peek_at_stack((n - i - 1) as isize);
        // TODO(max): Handle GC-hidden classes like Array, Hash, etc and make them look normal or
        // drop them or something
        let ty = ProfiledType::new(obj);
        VALUE::from(profiler.iseq).write_barrier(ty.class());
        profile_type.observe(ty);
    }
}

fn profile_splat_length(profiler: &mut Profiler, profile: &mut IseqProfile, ci: *const rb_callinfo) {
    let flags = unsafe { rb_vm_ci_flag(ci) };
    // Only call sites with VM_CALL_ARGS_SPLAT have a splat array on the stack.
    if flags & VM_CALL_ARGS_SPLAT == 0 {
        return;
    }

    let kwarg = unsafe { rb_vm_ci_kwarg(ci) };
    let caller_kw_count = if kwarg.is_null() { 0 } else { (unsafe { get_cikw_keyword_len(kwarg) }) as usize };
    // Starting at the top of the stack, skip the block argument, keyword-splat
    // hash, and explicit keyword values to reach the splat array.
    let splat_pos = usize::from(flags & VM_CALL_ARGS_BLOCKARG != 0)
        + usize::from(flags & VM_CALL_KW_SPLAT != 0)
        + caller_kw_count;
    let splat_array = profiler.peek_at_stack(splat_pos as isize);
    let length = if unsafe { RB_TYPE_P(splat_array, RUBY_T_ARRAY) } {
        SplatLength::try_from(unsafe { rb_jit_array_len(splat_array) }).ok()
    } else {
        None
    };
    profile.splat_lengths.entry(profiler.insn_idx)
        .or_insert_with(SplatLengthDistribution::new).observe(length);
}

fn profile_self(profiler: &mut Profiler, profile: &mut IseqProfile) {
    let entry = profile.entry_mut(profiler.insn_idx);
    if entry.opnd_types.is_empty() {
        entry.opnd_types.resize(1, TypeDistribution::new());
    }
    let obj = profiler.peek_at_self();
    // TODO(max): Handle GC-hidden classes like Array, Hash, etc and make them look normal or
    // drop them or something
    let ty = ProfiledType::new(obj);
    VALUE::from(profiler.iseq).write_barrier(ty.class());
    entry.opnd_types[0].observe(ty);
}

fn profile_block_handler(profiler: &mut Profiler, profile: &mut IseqProfile) {
    let entry = profile.entry_mut(profiler.insn_idx);
    if entry.opnd_types.is_empty() {
        entry.opnd_types.resize(1, TypeDistribution::new());
    }
    let obj = profiler.peek_at_block_handler();
    let ty = ProfiledType::block_handler(obj);
    VALUE::from(profiler.iseq).write_barrier(ty.class());
    entry.opnd_types[0].observe(ty);
}

/// How many times a single `yield` site may replace its profile and ask for a recompile.
/// A site whose fallback handlers keep changing family would otherwise be able to bounce the
/// compiled code between dispatches; two is enough for the boot-to-steady-state transition
/// this targets and leaves recompilation budget for the rest of the ISEQ.
const MAX_REPROFILE_RECOMPILES: u8 = 2;

/// The largest re-profiling window, in samples. Each window that decides nothing doubles the
/// next one, so a site keeps looking for a while — long enough to catch a program that only
/// changes what it yields to well after the site was compiled — and then gives up for good
/// rather than sampling every fallback for the rest of the process.
const MAX_REPROFILE_WINDOW: NumProfiles = 1024;

/// A `yield` site's re-profiling window: what its generic `rb_vm_invokeblock` fallback has
/// been handed since the window opened. See [`rb_zjit_invokeblock_reprofile`].
#[derive(Debug)]
struct Reprofile {
    /// Handlers seen so far in this window. Kept apart from the instruction's profile until
    /// the window decides something, so an inconclusive window leaves the profile alone.
    window: TypeDistribution,
    /// Samples still needed to close the window.
    samples_remaining: NumProfiles,
    /// Size of the current window. Doubles every time a window decides nothing.
    window_size: NumProfiles,
    /// Positional arguments the `yield` passes, which decides whether a sampled handler is one
    /// a recompile could dispatch.
    argc: u8,
    /// Recompiles this site may still ask for. Counts down from [`MAX_REPROFILE_RECOMPILES`].
    budget: u8,
}

/// What [`IseqProfile::observe_fallback_block_handler`] wants the caller to do next.
#[derive(PartialEq, Eq, Debug)]
enum Reprofiled {
    /// Recorded, or ignored because the site has stopped re-profiling. Nothing to do.
    Sampled,
    /// This site has no window yet. Open one under the VM lock and start over.
    NeedsWindow,
    /// The window closed on handlers a recompile could dispatch and has been folded into the
    /// instruction's profile. Invalidate the compiled code so it reads the new profile.
    Recompile,
}

/// True if recompiling a `yield` site that passes `argc` arguments against `window` would give
/// most of those samples a dispatch of their own. Mirrors the families `iseq_to_hir` can emit a
/// fast path for, so a window of Procs, of blocks whose parameters do not match the yield, or
/// of nothing in particular is not worth bouncing the compiled code over.
///
/// Every family is held to the chain's coverage bar here, including the two the compiler will
/// emit a tag test for at a smaller share: a recompile is only worth its cost when most of
/// what the fallback is seeing stops going through `rb_vm_invokeblock`, whereas adding an arm
/// to a site that is being compiled anyway is worth it for a fraction of the executions.
fn window_is_dispatchable(window: &TypeDistribution, argc: usize) -> bool {
    use crate::hir::CHAIN_COVERAGE_THRESHOLD;
    let summary = TypeDistributionSummary::new(window);
    let covered = |keep: &dyn Fn(ProfiledType) -> bool| {
        summary.coverage(|_, ty| !ty.is_empty() && keep(ty))
    };
    // IFUNC handlers are profiled by kind, so a site that always yields to a C block reaches
    // full coverage here even though it never sees the same handler object twice.
    if covered(&|ty| ty.is_block_ifunc()) >= CHAIN_COVERAGE_THRESHOLD {
        return true;
    }
    // `yield` with no arguments has no receiver to call the symbol's method on.
    if argc > 0 && covered(&|ty| ty.class().static_sym_p()) >= CHAIN_COVERAGE_THRESHOLD {
        return true;
    }
    covered(&|ty| unsafe { rb_IMEMO_TYPE_P(ty.class(), imemo_iseq) == 1 }
        && crate::hir::block_iseq_dispatchable(ty.class().as_iseq(), argc)) >= CHAIN_COVERAGE_THRESHOLD
}

/// Called from JIT code on the generic `rb_vm_invokeblock` fallback path of a `yield` site.
///
/// The block-handler profile that picked the site's compiled dispatch was collected in the
/// interpreter before the ISEQ was first compiled, and nothing refreshes it afterwards: the
/// fallback is an ordinary join in the compiled code rather than a side exit, so the
/// instruction never runs in the interpreter again and the `done_profiling_at` gate that
/// [`crate::codegen::exit_recompile`] waits on never opens. A site whose handlers changed
/// after boot — the `each` that yielded to C blocks while the program was loading and to Ruby
/// blocks once it is working — keeps calling `rb_vm_invokeblock` for the rest of the process.
///
/// Record what the handler actually is now. Once a window's worth of samples is in, and only
/// if most of them are handlers a recompile could dispatch, adopt the window as the site's
/// profile and invalidate the compiled code. An inconclusive window is thrown away and the
/// next one is twice as long, so a site whose handlers are genuinely chaotic converges on
/// sampling almost none of its fallbacks rather than on recompiling.
///
/// Sampling deliberately runs without the VM lock after the first fallback: a window that
/// decides nothing has to be able to open another one, so this call stays live on the fallback
/// path indefinitely and taking the lock per yield would cost more than the dispatch it is
/// trying to fix. What it writes is profile data into a window this site already owns, so no
/// collection grows here and a Ractor racing on the same site can only miscount buckets. The
/// lock is taken to create that window, and again for the invalidation a decided window asks
/// for.
///
/// `version` is the compiled unit this call lives in, not the ISEQ that owns the instruction:
/// an inlined callee has no compiled code of its own, so it is the outer function that has to
/// be invalidated. It also disarms the call: frames already running the invalidated code keep
/// reaching the fallback until they return, and testing the version they came from is what
/// stops those from sampling into the next version's window.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_invokeblock_reprofile(cfp: CfpPtr, version: *mut IseqVersion, frame_iseq: VALUE, insn_idx: u32, argc: u32) {
    if unsafe { (*version).is_invalidated() } {
        return;
    }
    let insn_idx = insn_idx as YarvInsnIdx;
    let iseq = frame_iseq.as_iseq();
    let handler = unsafe { rb_vm_get_untagged_block_handler(cfp) };
    match get_or_create_iseq_payload(iseq).profile.observe_fallback_block_handler(iseq, insn_idx, handler) {
        Reprofiled::Sampled => {}
        Reprofiled::NeedsWindow => {
            // First time this site has fallen back. Yield sites that never do should not pay
            // for a window, so it is created here rather than at compile time.
            with_vm_lock(src_loc!(), || {
                get_or_create_iseq_payload(iseq).profile.open_reprofile_window(insn_idx, argc as usize);
            });
        }
        Reprofiled::Recompile => {
            // Invalidate the running version so it recompiles and reads the adopted profile. It
            // is the last one: a new version only appears once this one has been invalidated,
            // and the check above returned early in that case.
            incr_counter_by(Counter::invokeblock_reprofile_recompile_count, 1);
            let compiled_iseq = VALUE::from(unsafe { (*version).iseq });
            with_vm_lock(src_loc!(), || {
                let compiled_iseq = compiled_iseq.as_iseq();
                let payload = get_or_create_iseq_payload(compiled_iseq);
                if let Some(version) = payload.versions.last_mut() {
                    let cb = crate::state::ZJITState::get_code_block();
                    crate::codegen::invalidate_iseq_version(cb, compiled_iseq, version);
                    cb.mark_all_executable();
                }
            });
        }
    }
}

fn profile_getblockparamproxy(profiler: &mut Profiler, profile: &mut IseqProfile) {
    let entry = profile.entry_mut(profiler.insn_idx);
    if entry.opnd_types.is_empty() {
        entry.opnd_types.resize(1, TypeDistribution::new());
    }

    let level = profiler.insn_opnd(1).as_u32();
    let ep = unsafe { get_cfp_ep_level(profiler.cfp, level) };
    let block_handler = unsafe { *ep.offset(VM_ENV_DATA_INDEX_SPECVAL as isize) };
    let untagged = unsafe { rb_vm_untag_block_handler(block_handler) };

    let ty = ProfiledType::block_handler(untagged);
    VALUE::from(profiler.iseq).write_barrier(ty.class());
    entry.opnd_types[0].observe(ty);
}

fn profile_invokesuper(profiler: &mut Profiler, profile: &mut IseqProfile) {
    let cme = unsafe { rb_vm_frame_method_entry(profiler.cfp) };
    let cme_value = VALUE(cme as usize);  // CME is a T_IMEMO, which is a VALUE

    profile.super_cme.entry(profiler.insn_idx)
        .or_insert_with(|| TypeDistribution::new()).observe(ProfiledType::object(cme_value));

    unsafe { rb_gc_writebarrier(profiler.iseq.into(), cme_value) };

    let cd: *const rb_call_data = profiler.insn_opnd(0).as_ptr();
    let argc = num_arguments_on_stack(cd);

    // Profile all the arguments and self (+1).
    profile_operands(profiler, profile, (argc + 1) as usize);
    profile_splat_length(profiler, profile, unsafe { (*cd).ci });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flags(u32);

impl Flags {
    const NONE: u32 = 0;
    const IS_IMMEDIATE: u32 = 1 << 0;
    /// Object is embedded and the ivar index lands within the object
    const IS_EMBEDDED: u32 = 1 << 1;
    /// Object is a T_OBJECT
    const IS_T_OBJECT: u32 = 1 << 2;
    /// Object is a struct with embedded fields
    const IS_STRUCT_EMBEDDED: u32 = 1 << 3;
    /// Set if the ProfiledType is used for profiling specific objects, not just classes/shapes
    const IS_OBJECT_PROFILING: u32 = 1 << 4;
    /// Block handler is an IFUNC (a block implemented in C)
    const IS_BLOCK_IFUNC: u32 = 1 << 5;
    /// Block handler is a Proc
    const IS_BLOCK_PROC: u32 = 1 << 6;

    pub fn none() -> Self { Self(Self::NONE) }

    pub fn immediate() -> Self { Self(Self::IS_IMMEDIATE) }
    pub fn is_immediate(self) -> bool { (self.0 & Self::IS_IMMEDIATE) != 0 }
    pub fn is_embedded(self) -> bool { (self.0 & Self::IS_EMBEDDED) != 0 }
    pub fn is_t_object(self) -> bool { (self.0 & Self::IS_T_OBJECT) != 0 }
    pub fn is_struct_embedded(self) -> bool { (self.0 & Self::IS_STRUCT_EMBEDDED) != 0 }
    pub fn is_object_profiling(self) -> bool { (self.0 & Self::IS_OBJECT_PROFILING) != 0 }
    pub fn is_block_ifunc(self) -> bool { (self.0 & Self::IS_BLOCK_IFUNC) != 0 }
    pub fn is_block_proc(self) -> bool { (self.0 & Self::IS_BLOCK_PROC) != 0 }
}

/// opt_send_without_block/opt_plus/... should store:
/// * the class of the receiver, so we can do method lookup
/// * the shape of the receiver, so we can optimize ivar lookup
///
/// with those two, pieces of information, we can also determine when an object is an immediate:
/// * Integer + IS_IMMEDIATE == Fixnum
/// * Float + IS_IMMEDIATE == Flonum
/// * Symbol + IS_IMMEDIATE == StaticSymbol
/// * NilClass == Nil
/// * TrueClass == True
/// * FalseClass == False
#[derive(Debug, Clone, Copy, Eq)]
pub struct ProfiledType {
    class: VALUE,
    shape: ShapeId,
    flags: Flags,
}

impl PartialEq for ProfiledType {
    fn eq(&self, other: &Self) -> bool {
        // IFUNC and Proc block handlers are allocated per call (`rb_vm_ifunc_new`) or per block
        // capture (`rb_vm_make_proc`), so their object identity says nothing about the call site:
        // a `yield` that always runs the same C block sees a different IFUNC every time. Treat all
        // IFUNC handlers as one profiled type and all Proc handlers as another so such sites look
        // monomorphic instead of filling up the distribution with garbage.
        if self.flags.is_block_ifunc() || other.flags.is_block_ifunc() {
            return self.flags.is_block_ifunc() && other.flags.is_block_ifunc();
        }
        if self.flags.is_block_proc() || other.flags.is_block_proc() {
            return self.flags.is_block_proc() && other.flags.is_block_proc();
        }
        self.class == other.class && self.shape == other.shape && self.flags == other.flags
    }
}

impl Default for ProfiledType {
    fn default() -> Self {
        Self::empty()
    }
}

impl ProfiledType {
    /// Profile the object itself
    fn object(obj: VALUE) -> Self {
        let mut flags = Flags::none();
        flags.0 |= Flags::IS_OBJECT_PROFILING;
        Self { class: obj, shape: INVALID_SHAPE_ID, flags }
    }

    /// Profile an untagged block handler (see `rb_vm_untag_block_handler`). ISEQ block handlers
    /// and symbols are recorded by identity because those objects are stable for a given block,
    /// while IFUNC and Proc handlers are only recorded by kind (see [`PartialEq`] above).
    fn block_handler(obj: VALUE) -> Self {
        let mut ty = Self::object(obj);
        if !obj.special_const_p() {
            if unsafe { rb_IMEMO_TYPE_P(obj, imemo_ifunc) == 1 } {
                ty.flags.0 |= Flags::IS_BLOCK_IFUNC;
            } else if unsafe { rb_obj_is_proc(obj).test() } {
                ty.flags.0 |= Flags::IS_BLOCK_PROC;
            }
        }
        ty
    }

    /// Profile the class and shape of the given object
    fn new(obj: VALUE) -> Self {
        // Qundef must never escape the VM internals; rb_class_of(Qundef) is undefined
        debug_assert_ne!(obj, Qundef, "should not profile Qundef");
        if obj.special_const_p() {
            return Self { class: obj.class_of(),
                          shape: INVALID_SHAPE_ID,
                          flags: Flags::immediate() };
        }
        let mut flags = Flags::none();
        let shape = obj.shape_id_of();
        if shape.layout() == ShapeLayout::RObject {
            flags.0 |= Flags::IS_EMBEDDED;
        }
        if obj.struct_embedded_p() {
            flags.0 |= Flags::IS_STRUCT_EMBEDDED;
        }
        if unsafe { RB_TYPE_P(obj, RUBY_T_OBJECT) } {
            flags.0 |= Flags::IS_T_OBJECT;
        }
        Self { class: obj.class_of(), shape, flags }
    }

    pub fn empty() -> Self {
        Self { class: VALUE(0), shape: INVALID_SHAPE_ID, flags: Flags::none() }
    }

    pub fn is_empty(&self) -> bool {
        self.class == VALUE(0)
    }

    pub fn class(&self) -> VALUE {
        self.class
    }

    pub fn shape(&self) -> ShapeId {
        self.shape
    }

    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// True if this profiled a block handler that is an IFUNC (a block implemented in C).
    pub fn is_block_ifunc(&self) -> bool {
        self.flags.is_block_ifunc()
    }

    pub fn is_fixnum(&self) -> bool {
        self.class == unsafe { rb_cInteger } && self.flags.is_immediate()
    }

    pub fn is_string(&self) -> bool {
        if self.flags.is_object_profiling() {
            panic!("should not call is_string on object-profiled ProfiledType");
        }
        // Fast paths for immediates and exact-class
        if self.flags.is_immediate() {
            return false;
        }

        let string = unsafe { rb_cString };
        if self.class == string{
            return true;
        }

        self.class.is_subclass_of(string) == ClassRelationship::Subclass
    }

    pub fn is_flonum(&self) -> bool {
        self.class == unsafe { rb_cFloat } && self.flags.is_immediate()
    }

    pub fn is_static_symbol(&self) -> bool {
        self.class == unsafe { rb_cSymbol } && self.flags.is_immediate()
    }

    pub fn is_nil(&self) -> bool {
        self.class == unsafe { rb_cNilClass } && self.flags.is_immediate()
    }

    pub fn is_true(&self) -> bool {
        self.class == unsafe { rb_cTrueClass } && self.flags.is_immediate()
    }

    pub fn is_false(&self) -> bool {
        self.class == unsafe { rb_cFalseClass } && self.flags.is_immediate()
    }
}

/// Per-instruction profile entry, stored sparsely in a sorted Vec.
#[derive(Debug)]
pub struct ProfileEntry {
    /// YARV instruction index
    insn_idx: u32,
    /// Type information of YARV instruction operands
    opnd_types: Vec<TypeDistribution>,
    /// Number of profiles remaining before recompilation. Counts down from --zjit-num-profiles.
    profiles_remaining: NumProfiles,
}

impl ProfileEntry {
    pub fn set_profiles_remaining(&mut self, num_profiles: NumProfiles) {
        self.profiles_remaining = num_profiles;
    }
}

#[derive(Debug)]
pub struct IseqProfile {
    /// Sparse storage of per-instruction profile data, sorted by instruction index.
    /// Only instructions that have actually been profiled have entries here.
    entries: Vec<ProfileEntry>,

    /// Method entries for `super` calls (stored as VALUE to be GC-safe)
    super_cme: HashMap<YarvInsnIdx, TypeDistribution>,

    /// Observed lengths of caller splat arrays for call instructions.
    splat_lengths: HashMap<YarvInsnIdx, SplatLengthDistribution>,

    /// Re-profiling windows for `yield` sites, keyed by instruction index. Only sites that
    /// have actually taken their compiled fallback have an entry here.
    reprofiles: HashMap<YarvInsnIdx, Reprofile>,
}

impl IseqProfile {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            super_cme: HashMap::new(),
            splat_lengths: HashMap::new(),
            reprofiles: HashMap::new(),
        }
    }

    /// True if the `yield` site at `insn_idx` is still re-profiling itself, i.e. it is worth
    /// emitting the [`rb_zjit_invokeblock_reprofile`] call on its fallback path. A site that
    /// has never fallen back has no window yet and gets the benefit of the doubt.
    pub fn can_reprofile_at(&self, insn_idx: YarvInsnIdx) -> bool {
        self.reprofiles.get(&insn_idx).map_or(true, |reprofile| reprofile.budget > 0)
    }

    /// Open the re-profiling window for the `yield` site at `insn_idx`. Called under the VM
    /// lock the first time the site's compiled fallback runs, which is what lets every later
    /// sample update the window without the lock.
    fn open_reprofile_window(&mut self, insn_idx: YarvInsnIdx, argc: usize) {
        // Size the first window so its samples cannot be outvoted by the profile they are
        // being folded into: the compile-time gates read bucket shares, so a site that yielded
        // to a C block `num_profiles` times while the program booted only stops looking like a
        // pure IFUNC site once at least that many other handlers have landed on top.
        let already_seen = self.entry(insn_idx)
            .and_then(|entry| entry.opnd_types.first())
            .map_or(0, |distribution| TypeDistributionSummary::new(distribution).num_seen());
        let window_size = NumProfiles::try_from(already_seen).unwrap_or(NumProfiles::MAX)
            .max(get_option!(num_profiles)).max(1);
        self.reprofiles.entry(insn_idx).or_insert_with(|| Reprofile {
            window: TypeDistribution::new(),
            samples_remaining: window_size,
            window_size,
            argc: u8::try_from(argc).unwrap_or(u8::MAX),
            budget: MAX_REPROFILE_RECOMPILES,
        });
    }

    /// Record a block handler the JIT fallback was handed, and tell the caller what to do next.
    /// [`Reprofiled::Recompile`] means the window closed on handlers a recompile could dispatch
    /// and has been folded into the instruction's profile.
    ///
    /// The window is merged rather than substituted: it only sees the executions the current
    /// dispatch could not handle, so on its own it would argue for dropping a fast path that is
    /// working. Merging leaves both families in the profile, and the compiler emits an arm for
    /// each of them.
    fn observe_fallback_block_handler(&mut self, iseq: IseqPtr, insn_idx: YarvInsnIdx, handler: VALUE) -> Reprofiled {
        let Some(reprofile) = self.reprofiles.get_mut(&insn_idx) else { return Reprofiled::NeedsWindow };
        if reprofile.budget == 0 {
            return Reprofiled::Sampled;
        }
        incr_counter_by(Counter::invokeblock_reprofile_sample_count, 1);
        let ty = ProfiledType::block_handler(handler);
        VALUE::from(iseq).write_barrier(ty.class());
        reprofile.window.observe(ty);

        reprofile.samples_remaining = reprofile.samples_remaining.saturating_sub(1);
        if reprofile.samples_remaining > 0 {
            return Reprofiled::Sampled;
        }

        let decided = window_is_dispatchable(&reprofile.window, reprofile.argc.into());
        let window = std::mem::replace(&mut reprofile.window, TypeDistribution::new());
        if !decided {
            // Nothing here a recompile could dispatch. Leave the profile alone and look again
            // over twice as many fallbacks, or stop looking once the windows have grown past
            // what a site that is going to settle would need.
            reprofile.window_size = reprofile.window_size.saturating_mul(2);
            if reprofile.window_size > MAX_REPROFILE_WINDOW {
                reprofile.budget = 0;
            }
            reprofile.samples_remaining = reprofile.window_size;
            return Reprofiled::Sampled;
        }
        reprofile.budget -= 1;
        reprofile.samples_remaining = reprofile.window_size;

        let entry = self.entry_mut(insn_idx);
        if entry.opnd_types.is_empty() {
            entry.opnd_types.push(window);
        } else {
            entry.opnd_types[0].merge(&window);
        }
        Reprofiled::Recompile
    }

    /// Get or create a mutable profile entry for the given instruction index.
    pub fn entry_mut(&mut self, insn_idx: YarvInsnIdx) -> &mut ProfileEntry {
        let idx = insn_idx as u32;
        match self.entries.binary_search_by_key(&idx, |e| e.insn_idx) {
            Ok(i) => &mut self.entries[i],
            Err(i) => {
                self.entries.insert(i, ProfileEntry {
                    insn_idx: idx,
                    opnd_types: Vec::new(),
                    profiles_remaining: get_option!(num_profiles),
                });
                &mut self.entries[i]
            }
        }
    }

    /// Get a profile entry for the given instruction index (read-only).
    fn entry(&self, insn_idx: YarvInsnIdx) -> Option<&ProfileEntry> {
        let idx = insn_idx as u32;
        self.entries.binary_search_by_key(&idx, |e| e.insn_idx)
            .ok().map(|i| &self.entries[i])
    }

    /// Check if enough profiles have been gathered for this instruction.
    pub fn done_profiling_at(&self, insn_idx: YarvInsnIdx) -> bool {
        self.entry(insn_idx).map_or(false, |e| e.profiles_remaining == 0)
    }

    /// Get profiled operand types for a given instruction index
    pub fn get_operand_types(&self, insn_idx: YarvInsnIdx) -> Option<&[TypeDistribution]> {
        self.entry(insn_idx).map(|e| e.opnd_types.as_slice()).filter(|s| !s.is_empty())
    }

    pub fn get_splat_length_summary(&self, insn_idx: YarvInsnIdx) -> Option<SplatLengthDistributionSummary> {
        self.splat_lengths.get(&insn_idx)
            .map(SplatLengthDistributionSummary::new)
    }

    pub fn get_super_method_entry(&self, insn_idx: YarvInsnIdx) -> Option<*const rb_callable_method_entry_t> {
        let Some(entry) = self.super_cme.get(&insn_idx) else { return None };
        let summary = TypeDistributionSummary::new(entry);

        if summary.is_monomorphic() {
            Some(summary.bucket(0).class.0 as *const rb_callable_method_entry_t)
        } else {
            None
        }
    }

    /// Run a given callback with every object in IseqProfile
    pub fn each_object(&self, callback: impl Fn(VALUE)) {
        for entry in &self.entries {
            for distribution in &entry.opnd_types {
                for profiled_type in distribution.each_item() {
                    // If the type is a GC object, call the callback
                    callback(profiled_type.class);
                }
            }
        }

        for super_cme_values in self.super_cme.values() {
            for profiled_type in super_cme_values.each_item() {
                callback(profiled_type.class)
            }
        }

        // Handlers sampled by a re-profiling window that has not been adopted yet.
        for reprofile in self.reprofiles.values() {
            for profiled_type in reprofile.window.each_item() {
                callback(profiled_type.class)
            }
        }
    }

    /// Run a given callback with a mutable reference to every object in IseqProfile.
    pub fn each_object_mut(&mut self, callback: impl Fn(&mut VALUE)) {
        for entry in &mut self.entries {
            for distribution in &mut entry.opnd_types {
                for ref mut profiled_type in distribution.each_item_mut() {
                    // If the type is a GC object, call the callback
                    callback(&mut profiled_type.class);
                }
            }
        }

        // Update CME references if they move during compaction.
        for super_cme_values in self.super_cme.values_mut() {
            for ref mut profiled_type in super_cme_values.each_item_mut() {
                callback(&mut profiled_type.class)
            }
        }

        for reprofile in self.reprofiles.values_mut() {
            for ref mut profiled_type in reprofile.window.each_item_mut() {
                callback(&mut profiled_type.class)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cruby::*;

    #[test]
    fn can_profile_block_handler() {
        with_rubyvm(|| eval("
            def foo = yield
            foo rescue 0
            foo rescue 0
        "));
    }
}
