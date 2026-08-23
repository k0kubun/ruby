//! Profiler for runtime information.

// We use the YARV bytecode constants which have a CRuby-style name
#![allow(non_upper_case_globals)]

use crate::fasthash::FastHashMap as HashMap;
use crate::{cruby::*, payload::get_or_create_iseq_payload, options::{get_option, NumProfiles}};
use crate::mem_stats::hash_table_bytes;
use crate::distribution::{Distribution, DistributionSummary, StableBucket};
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
        // The value being destructured is the only operand `expandarray` pops.
        YARVINSN_expandarray => profile_operands(profiler, profile, 1),
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
            profile_send_method_name(profiler, profile, cd);
        }
        YARVINSN_splatarray => profile_operands(profiler, profile, 1),
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
    // Only touches counters, never a distribution, so `marked_objects` stays valid.
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

/// How many profile entries to make room for at a time. See
/// [`IseqProfile::entry_mut`].
const ENTRY_GROWTH_STEP: usize = 8;

pub type TypeDistribution = Distribution<ProfiledType, DISTRIBUTION_SIZE>;

pub type TypeDistributionSummary = DistributionSummary<ProfiledType, DISTRIBUTION_SIZE>;

pub type SplatLength = u32;

/// `None` records an unknown length so this distribution covers the same
/// executions as the operand type profile.
pub type SplatLengthDistribution = Distribution<Option<SplatLength>, DISTRIBUTION_SIZE>;

pub type SplatLengthDistributionSummary = DistributionSummary<Option<SplatLength>, DISTRIBUTION_SIZE>;

/// Allocate exactly `n` empty operand type distributions.
fn new_opnd_types(n: usize) -> Box<[TypeDistribution]> {
    // vec![elem; n] allocates exactly n elements, unlike Vec::resize().
    vec![TypeDistribution::new(); n].into_boxed_slice()
}

/// Profile the Type of top-`n` stack operands
fn profile_operands(profiler: &mut Profiler, profile: &mut IseqProfile, n: usize) {
    let entry = profile.entry_mut(profiler.insn_idx);
    if entry.opnd_types.is_empty() {
        entry.opnd_types = new_opnd_types(n);
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
    profile.splat_lengths_mut()
        .entry(profiler.insn_idx)
        .or_insert_with(SplatLengthDistribution::new).observe(length);
}

/// `recv.send(:name, ...)` picks its callee from the first argument rather than from the
/// call site's method ID, so the ordinary operand type profile (always `Symbol`) says
/// nothing useful. Record the method-name objects themselves; `type_specialize` uses them
/// to turn each observed name into a direct call.
fn profile_send_method_name(profiler: &mut Profiler, profile: &mut IseqProfile, cd: *const rb_call_data) {
    let ci = unsafe { (*cd).ci };
    let mid = unsafe { rb_vm_ci_mid(ci) };
    if mid != ID!(send) && mid != ID!(__send__) {
        return;
    }
    // Only plain positional calls can be specialized; don't spend profile slots on the rest.
    let flags = unsafe { rb_vm_ci_flag(ci) };
    if flags & (VM_CALL_ARGS_SPLAT | VM_CALL_KWARG | VM_CALL_KW_SPLAT | VM_CALL_ARGS_BLOCKARG | VM_CALL_FORWARDING) != 0 {
        return;
    }
    let argc = unsafe { vm_ci_argc(ci) } as usize;
    if argc == 0 {
        return;
    }
    // Stack is [.., recv, name, arg1, .., argN-1] with argN-1 on top.
    let name = profiler.peek_at_stack((argc - 1) as isize);
    // Dynamic symbols and Strings are not stable enough to key a guard on; static symbols
    // are immortal, so recording one keeps no object alive that would not be alive anyway.
    if !name.static_sym_p() {
        return;
    }
    profile.send_mid_mut()
        .entry(profiler.insn_idx)
        .or_insert_with(TypeDistribution::new).observe(ProfiledType::object(name));
}

fn profile_self(profiler: &mut Profiler, profile: &mut IseqProfile) {
    let entry = profile.entry_mut(profiler.insn_idx);
    if entry.opnd_types.is_empty() {
        entry.opnd_types = new_opnd_types(1);
    }
    let obj = profiler.peek_at_self();
    // TODO(max): Handle GC-hidden classes like Array, Hash, etc and make them look normal or
    // drop them or something
    let ty = ProfiledType::new(obj);
    VALUE::from(profiler.iseq).write_barrier(ty.class());
    entry.opnd_types[0].observe(ty);
}

/// Samples an ivar site's fallback path has to see before it decides whether the shapes
/// arriving there are worth a recompile. Small enough that a site which was frozen on the
/// wrong shape recovers almost immediately, large enough that a brief detour through an
/// unusual receiver does not spend a version. Kept under `u8::MAX` so the per-instruction window
/// counters fit in a byte each; [`ProfileEntry`] is allocated once per profiled instruction and
/// its size is asserted on.
const IVAR_REPROFILE_WINDOW: u32 = 64;

/// Share of a window that the shapes a recompile would add an arm for must account for,
/// in percent, before the recompile is worth a version.
///
/// A single missing shape is not evidence that a recompile helps: at a site that is genuinely
/// polymorphic the fallback sees a long tail of shapes, and an arm for one of them removes a
/// few percent of the calls while costing a version and the code for every other arm. What
/// matters is the share the arms a recompile would add cover *together*, which is why this is
/// not a per-shape threshold -- a window split evenly between three missing shapes is a good
/// bet, and a window with one missing shape in it is not.
const IVAR_REPROFILE_MIN_SHARE_PERCENT: u32 = 50;

/// Share of a window that has to be shapes the profile has no bucket left to hold before the
/// cold buckets are dropped to make room. Set to the same share a recompile needs: a window this
/// full of shapes nothing can record is one where the profile, not the dispatch, is what stands
/// between the site and a specialization.
const IVAR_EVICT_MIN_CROWDED_PERCENT: u32 = 50;

/// How many times one site may drop its cold profile buckets. Each eviction is followed by a
/// window that refills them from live traffic and, if the refill holds up, a recompile; two
/// rounds are enough to converge, and more would let a site with a long shape tail trade versions
/// for buckets indefinitely.
const MAX_IVAR_PROFILE_EVICTIONS: u8 = 2;

/// Result of [`IseqProfile::observe_ivar_fallback`].
#[derive(PartialEq, Eq, Debug)]
enum IvarReprofiled {
    /// Recorded, mid-window. Nothing to do.
    Sampled,
    /// The window closed without making the case for a recompile.
    Declined,
    /// The window closed with most of its samples on shapes the compiled dispatch has no arm
    /// for. They are now in the instruction's profile; recompile to pick them up.
    Recompile,
}

impl IseqProfile {
    /// Record the shape of a receiver that reached an ivar site's fallback path, and report
    /// whether a recompile would now take most of that traffic off the fallback.
    ///
    /// The window is kept in the instruction's own profile entry, so a site that falls back
    /// rarely costs one distribution slot and nothing else.
    fn observe_ivar_fallback(&mut self, iseq: IseqPtr, insn_idx: YarvInsnIdx, recv: VALUE) -> IvarReprofiled {
        let entry = self.entry_mut(insn_idx);
        if entry.opnd_types.is_empty() {
            entry.opnd_types = new_opnd_types(1);
        }
        // The dispatch that is running was built from the shapes the profile held when the ISEQ
        // was last compiled. Remember how many that was the first time we sample: because this
        // path only ever grows the distribution through `observe_stable`, every bucket at or
        // past that index is a shape folded in since, which the running code has no arm for.
        let dispatch_shapes = match entry.ivar_dispatch_shapes {
            Some(shapes) => shapes,
            None => {
                let shapes = entry.opnd_types[0].each_item().count() as u8;
                entry.ivar_dispatch_shapes = Some(shapes);
                shapes
            }
        };
        let ty = ProfiledType::new(recv);
        let fixable = if ty.shape().is_complex() {
            // Too-complex shapes keep their ivars in a hash table, so the dispatch would filter
            // this one back out. Recording it would only cost a bucket a fixable shape could use.
            false
        } else {
            // De-duplicate by shape, the way the dispatch itself does.
            let bucket = entry.opnd_types[0].observe_stable(ty, |seen, ty| seen.shape() == ty.shape());
            if let StableBucket::Inserted(_) = bucket {
                // The distribution keeps the class alive for as long as the profile does.
                VALUE::from(iseq).write_barrier(ty.class());
            }
            match bucket {
                StableBucket::Existing(index) | StableBucket::Inserted(index) => index >= dispatch_shapes as usize,
                // No bucket left to record this shape in, so no recompile can specialize it
                // until one is freed below.
                StableBucket::Full => {
                    entry.ivar_fallback_crowded = entry.ivar_fallback_crowded.saturating_add(1);
                    false
                }
            }
        };
        entry.ivar_fallback_samples = entry.ivar_fallback_samples.saturating_add(1);
        if fixable {
            entry.ivar_fallback_fixable = entry.ivar_fallback_fixable.saturating_add(1);
        }
        if u32::from(entry.ivar_fallback_samples) < IVAR_REPROFILE_WINDOW {
            return IvarReprofiled::Sampled;
        }
        entry.ivar_fallback_samples = 0;
        let fixable = u32::from(std::mem::take(&mut entry.ivar_fallback_fixable));
        let crowded = u32::from(std::mem::take(&mut entry.ivar_fallback_crowded));
        if fixable * 100 >= IVAR_REPROFILE_WINDOW * IVAR_REPROFILE_MIN_SHARE_PERCENT {
            // The recompile rebuilds the dispatch from the whole profile, so start the next
            // window measuring against all of it.
            entry.ivar_dispatch_shapes = None;
            IvarReprofiled::Recompile
        } else if crowded * 100 >= IVAR_REPROFILE_WINDOW * IVAR_EVICT_MIN_CROWDED_PERCENT
            && entry.ivar_profile_evictions < MAX_IVAR_PROFILE_EVICTIONS
            && Self::insn_profiles_self_shape(iseq, insn_idx)
        {
            // Most of this window was shapes with nowhere to go. The profile is a sample of the
            // ISEQ's first executions, and on a long-lived process that is the least
            // representative sample there is: on lobsters, ActiveRecord's `@attributes` read is
            // compiled against eight boot-time shapes and then hands three steady-state shapes,
            // none of them recordable, to the fallback 570K times.
            //
            // Drop everything but the most-observed bucket and let the next window refill the
            // rest from live traffic. Keeping bucket 0 keeps the arm most likely to still be
            // taking hits, so this gives up at most the arms the site is demonstrably not
            // hitting -- it only runs at all once a window's worth of receivers has missed
            // every one of them.
            entry.ivar_profile_evictions += 1;
            entry.opnd_types[0].retain_primary();
            entry.ivar_dispatch_shapes = Some(1);
            crate::stats::incr_counter!(ivar_profile_evicted_count);
            IvarReprofiled::Sampled
        } else {
            // Most of what the fallback handles is something a recompile cannot take away: a
            // shape already in the dispatch (whose arm was dropped, or which is unspecializable),
            // or one of so many shapes that the profile has no room left for them. Do not spend a
            // version to move a few percent of it.
            crate::stats::incr_counter!(ivar_respecialize_declined_count);
            IvarReprofiled::Declined
        }
    }

    /// Whether the instruction at `insn_idx` profiles `self` into `opnd_types[0]`, so that
    /// distribution is a shape profile this fallback path may rewrite.
    ///
    /// `dispatch_ivar` also serves an inlined `attr_reader` whose shape guard could not exit, and
    /// there the frame state names the *call site*, whose `opnd_types[0]` is the receiver
    /// distribution the send specialization is built from: evicting buckets out of that would
    /// change which method the call compiles to, not which shapes an ivar chain covers. A
    /// `setinstancevariable` does profile `self`, but its arms have to survive
    /// `prepare_optimized_setivar` as well as a shape match, so a refill is a worse bet there and
    /// measurably lost ground on lobsters.
    fn insn_profiles_self_shape(iseq: IseqPtr, insn_idx: YarvInsnIdx) -> bool {
        let opcode = unsafe {
            let pc = rb_iseq_pc_at_idx(iseq, insn_idx as u32);
            rb_zjit_insn_to_bare_insn(rb_iseq_opcode_at_pc(iseq, pc))
        };
        opcode as u32 == YARVINSN_getinstancevariable
    }
}

/// Called from JIT code on the fallback path of an ivar site that was compiled on the final
/// version, where [`crate::hir::CompilePolicy::no_side_exits`] turned the site's shape guard
/// into a branch with a C-call fallback.
///
/// That dispatch is built from a profile the interpreter collected before the ISEQ was first
/// compiled, and nothing refreshes it: once compiled, the instruction only runs in the
/// interpreter on a side exit, and the exit-free fallback is not one. A site whose receiver
/// changed shape after boot therefore calls `rb_vm_getinstancevariable` for the rest of the
/// process even though the new shape is perfectly specializable.
///
/// Sample what the fallback is actually handed. Once a window's worth of samples names a
/// shape the dispatch has no arm for, the shape is already in the instruction's profile, so
/// grant the ISEQ one extra version and invalidate the compiled unit to spend it. Each such
/// version strictly adds an arm to a dispatch that was falling back, and
/// [`crate::payload::MAX_IVAR_RESPECIALIZATIONS`] bounds how many an ISEQ may earn.
///
/// `version` is the compiled unit this call lives in, not the ISEQ owning the instruction: an
/// inlined callee has no code of its own, so the outer function is what gets invalidated.
/// Testing it also disarms frames still running the invalidated code.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_ivar_reprofile(version: *mut crate::payload::IseqVersion, frame_iseq: VALUE, insn_idx: u32, recv: VALUE) {
    // Both of these are one load off `version`, which keeps the give-up path from paying for the
    // payload lookup and the profile timer below.
    if unsafe { (*version).is_invalidated() || (*version).ivar_reprofile_windows == 0 } {
        return;
    }
    // Immediates have no shape to specialize and would poison the profile with a bucket the
    // dispatch can never use.
    if recv.special_const_p() {
        return;
    }
    let insn_idx = insn_idx as YarvInsnIdx;
    let frame_iseq = frame_iseq.as_iseq();
    let reprofiled = with_time_stat(profile_time_ns, || {
        get_or_create_iseq_payload(frame_iseq).profile.observe_ivar_fallback(frame_iseq, insn_idx, recv)
    });
    match reprofiled {
        IvarReprofiled::Sampled => return,
        IvarReprofiled::Declined => {
            // Spend one of the version's windows. When they run out the sampling call stays in
            // the code, but every execution of it stops at the check above.
            let windows = unsafe { &mut (*version).ivar_reprofile_windows };
            *windows -= 1;
            if *windows == 0 {
                crate::stats::incr_counter!(ivar_respecialize_giveup_count);
                // Leave the sampling out of whatever this ISEQ compiles next. Invalidating just
                // to drop it is not worth a compile -- the call is already a no-op after the
                // check above -- but a version compiled for any other reason should not pay for
                // evidence this ISEQ has already gathered and rejected.
                // The flag belongs to the unit that compiled the call, which is what
                // `Function::emit_ivar_reprofile` tests, not the frame the site came from.
                get_or_create_iseq_payload(unsafe { (*version).iseq }).ivar_reprofile_giveup = true;
            }
            return;
        }
        IvarReprofiled::Recompile => {}
    }
    // Read the compiled unit's ISEQ out before taking the lock: `version` points into the
    // payload, and holding a reference to it across the lock's unwind boundary is not allowed.
    let compiled_iseq = VALUE::from(unsafe { (*version).iseq });
    with_vm_lock(src_loc!(), || {
        let compiled_iseq = compiled_iseq.as_iseq();
        let payload = get_or_create_iseq_payload(compiled_iseq);
        if payload.ivar_respecializations >= crate::payload::MAX_IVAR_RESPECIALIZATIONS {
            return;
        }
        payload.ivar_respecializations += 1;
        crate::stats::incr_counter!(ivar_respecialize_count);
        if let Some(version) = payload.versions.last_mut() {
            let cb = crate::state::ZJITState::get_code_block();
            // The respecialization budget was just raised above, so this is not the
            // grant path in invalidate_iseq_version(): it simply fits under the new limit.
            crate::codegen::invalidate_iseq_version(cb, compiled_iseq, version, crate::codegen::InvalidationCause::Respecialize);
            cb.mark_all_executable();
        }
    });
}

fn profile_block_handler(profiler: &mut Profiler, profile: &mut IseqProfile) {
    let obj = profiler.peek_at_block_handler();
    let ty = ProfiledType::block_handler(obj);
    VALUE::from(profiler.iseq).write_barrier(ty.class());
    let insn_idx = profiler.insn_idx;
    profile.block_handlers_mut().entry(insn_idx)
        .or_insert_with(TypeDistribution::new).observe(ty);

    // The operand slots profile the `yield`ed arguments, not the handler. A Symbol
    // handler turns the `yield` into a send to the first argument, and the class of
    // that argument is what decides whether the send can be compiled directly.
    let cd: *const rb_call_data = profiler.insn_opnd(0).as_ptr();
    let argc = num_arguments_on_stack(cd);
    if argc > 0 {
        profile_operands(profiler, profile, argc);
    }
}

fn profile_getblockparamproxy(profiler: &mut Profiler, profile: &mut IseqProfile) {
    let entry = profile.entry_mut(profiler.insn_idx);
    if entry.opnd_types.is_empty() {
        entry.opnd_types = new_opnd_types(1);
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

    profile.super_cme_mut()
        .entry(profiler.insn_idx)
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
    /// Set if the ProfiledType was synthesized for one arm of a polymorphic dispatch rather
    /// than observed at that program point. The class is guaranteed by the arm's type test,
    /// but the shape is only the shape the profiler happened to see for that class.
    const IS_POLYMORPHIC_ARM: u32 = 1 << 5;
    /// Block handler is an IFUNC (a block implemented in C)
    const IS_BLOCK_IFUNC: u32 = 1 << 6;
    /// Block handler is a Proc
    const IS_BLOCK_PROC: u32 = 1 << 7;

    pub fn none() -> Self { Self(Self::NONE) }

    pub fn immediate() -> Self { Self(Self::IS_IMMEDIATE) }
    pub fn is_immediate(self) -> bool { (self.0 & Self::IS_IMMEDIATE) != 0 }
    pub fn is_embedded(self) -> bool { (self.0 & Self::IS_EMBEDDED) != 0 }
    pub fn is_t_object(self) -> bool { (self.0 & Self::IS_T_OBJECT) != 0 }
    pub fn is_struct_embedded(self) -> bool { (self.0 & Self::IS_STRUCT_EMBEDDED) != 0 }
    pub fn is_object_profiling(self) -> bool { (self.0 & Self::IS_OBJECT_PROFILING) != 0 }
    pub fn is_polymorphic_arm(self) -> bool { (self.0 & Self::IS_POLYMORPHIC_ARM) != 0 }
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

    /// Return a copy marked as belonging to one arm of a polymorphic dispatch.
    pub fn as_polymorphic_arm(mut self) -> Self {
        self.flags.0 |= Flags::IS_POLYMORPHIC_ARM;
        self
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

    /// True when the profiled object was an instance of `Array` itself (not a subclass and not a
    /// singleton), which is what the `expandarray` fast path needs.
    pub fn is_array_exact(&self) -> bool {
        !self.flags.is_immediate() && self.class == unsafe { rb_cArray }
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
    /// Type information of YARV instruction operands. A boxed slice rather
    /// than a `Vec` because it is sized once and never resized: `Vec` would
    /// round the allocation up to `MIN_NON_ZERO_CAP` (4 elements), which for
    /// 80-byte distributions wastes up to 240 bytes on every single-operand
    /// instruction, and would also carry a capacity field we never read.
    opnd_types: Box<[TypeDistribution]>,
    /// Number of profiles remaining before recompilation. Counts down from --zjit-num-profiles.
    profiles_remaining: NumProfiles,
    /// Receivers seen on this ivar site's fallback path in the current re-profiling window.
    /// See [`rb_zjit_ivar_reprofile`].
    ivar_fallback_samples: u8,
    /// How many of those a recompile would give an arm of its own.
    ivar_fallback_fixable: u8,
    /// How many of those named a shape the profile has no bucket left to record.
    ivar_fallback_crowded: u8,
    /// How many times this site has dropped its cold profile buckets to make room for the shapes
    /// its fallback is seeing. Capped at [`MAX_IVAR_PROFILE_EVICTIONS`].
    ivar_profile_evictions: u8,
    /// Number of shapes the dispatch now running was compiled from, i.e. how many buckets of
    /// `opnd_types[0]` it has arms for. `None` until the first sample after a compile.
    ivar_dispatch_shapes: Option<u8>,
}

impl ProfileEntry {
    pub fn set_profiles_remaining(&mut self, num_profiles: NumProfiles) {
        self.profiles_remaining = num_profiles;
    }
}

/// What an [`IseqProfile`]'s heap bytes are spent on. See [`IseqProfile::heap_size`].
#[derive(Default, Debug)]
pub struct ProfileHeapSize {
    /// Total heap bytes owned by the profile.
    pub bytes: usize,
    /// Of those, bytes in the unused tail of the `entries` Vec.
    pub entry_slack_bytes: usize,
    /// Number of per-instruction profile entries.
    pub entry_count: usize,
    /// Number of operand type distributions across all entries.
    pub distribution_count: usize,
    /// How many of those distributions saw at most one type.
    pub monomorphic_distribution_count: usize,
    /// Number of objects in the dense array GC marking walks.
    pub marked_object_count: usize,
}

#[derive(Debug)]
pub struct IseqProfile {
    /// Sparse storage of per-instruction profile data, sorted by instruction index.
    /// Only instructions that have actually been profiled have entries here.
    entries: Vec<ProfileEntry>,

    /// Method entries for `super` calls (stored as VALUE to be GC-safe).
    /// Boxed because only ISEQs containing `invokesuper` ever use it, and an
    /// inline `HashMap` costs every payload 48 bytes to hold nothing.
    super_cme: Option<Box<HashMap<YarvInsnIdx, TypeDistribution>>>,

    /// Method-name symbols observed as the first argument of `send`/`__send__` call sites
    /// (stored as VALUE to be GC-safe). Boxed for the same reason as `super_cme`: only
    /// ISEQs that actually contain a `send`/`__send__` site ever populate it.
    send_mid: Option<Box<HashMap<YarvInsnIdx, TypeDistribution>>>,

    /// Observed lengths of caller splat arrays for call instructions. Boxed for
    /// the same reason as `super_cme`.
    splat_lengths: Option<Box<HashMap<YarvInsnIdx, SplatLengthDistribution>>>,

    /// Block handlers observed at `invokeblock` sites (stored as VALUE to be GC-safe).
    /// Kept out of `opnd_types` so that the entry's operand slots profile the yielded
    /// arguments instead: a `yield` to a Symbol block sends to the first argument, and
    /// that send only specializes if the argument's class was profiled. Boxed for the
    /// same reason as `super_cme`.
    block_handlers: Option<Box<HashMap<YarvInsnIdx, TypeDistribution>>>,

    /// Dense copy of every object the distributions above reference, which is what
    /// GC marking actually walks. See [`IseqProfile::marked_objects`].
    ///
    /// Duplicates are dropped while building it, so this is a set, not a multiset;
    /// marking an object once or twice is the same thing to the GC.
    marked_objects: Vec<VALUE>,

    /// Whether `marked_objects` still describes the distributions. Every method that
    /// can hand out a mutable distribution sets it: `entry_mut`, `super_cme_mut`,
    /// `send_mid_mut` and `each_object_mut`. Nothing else in this file may touch
    /// `entries`, `super_cme` or `send_mid` mutably without going through one of
    /// those -- a mutation that failed to set this flag would leave marking blind to
    /// a newly recorded object, i.e. a use-after-free.
    marked_objects_stale: bool,
}

impl IseqProfile {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            super_cme: None,
            send_mid: None,
            splat_lengths: None,
            block_handlers: None,
            marked_objects: Vec::new(),
            // Nothing recorded yet, so the (empty) dense copy is already accurate.
            marked_objects_stale: false,
        }
    }

    /// Get or create a mutable profile entry for the given instruction index.
    pub fn entry_mut(&mut self, insn_idx: YarvInsnIdx) -> &mut ProfileEntry {
        // The caller may be about to record an object into this entry's
        // distributions. Just a store; the dense copy keeps its allocation.
        self.marked_objects_stale = true;
        let idx = insn_idx as u32;
        match self.entries.binary_search_by_key(&idx, |e| e.insn_idx) {
            Ok(i) => &mut self.entries[i],
            Err(i) => {
                if self.entries.len() == self.entries.capacity() {
                    // Grow by a fixed step rather than doubling. An ISEQ has a
                    // bounded number of profiled instructions and these entries
                    // live as long as the ISEQ, so doubling leaves up to half of
                    // a long-lived table permanently unused.
                    self.entries.reserve_exact(ENTRY_GROWTH_STEP);
                }
                self.entries.insert(i, ProfileEntry {
                    insn_idx: idx,
                    opnd_types: Box::new([]),
                    profiles_remaining: get_option!(num_profiles),
                    ivar_fallback_samples: 0,
                    ivar_fallback_fixable: 0,
                    ivar_fallback_crowded: 0,
                    ivar_profile_evictions: 0,
                    ivar_dispatch_shapes: None,
                });
                &mut self.entries[i]
            }
        }
    }

    /// Mutable access to the `invokesuper` method-entry distributions, creating the
    /// table on first use. Goes through here so that recording a CME cannot skip
    /// invalidating the dense copy GC marking walks.
    fn super_cme_mut(&mut self) -> &mut HashMap<YarvInsnIdx, TypeDistribution> {
        self.marked_objects_stale = true;
        self.super_cme.get_or_insert_with(Default::default)
    }

    /// Mutable access to the `send`/`__send__` method-name distributions, creating
    /// the table on first use. Same reason as [`Self::super_cme_mut`].
    fn send_mid_mut(&mut self) -> &mut HashMap<YarvInsnIdx, TypeDistribution> {
        self.marked_objects_stale = true;
        self.send_mid.get_or_insert_with(Default::default)
    }

    /// Mutable access to the splat length distributions, creating the table on first
    /// use. Unlike the two above this needs no invalidation: a
    /// `SplatLengthDistribution` holds array lengths, not objects, so GC marking
    /// never looks at it.
    fn splat_lengths_mut(&mut self) -> &mut HashMap<YarvInsnIdx, SplatLengthDistribution> {
        self.splat_lengths.get_or_insert_with(Default::default)
    }

    /// Mutable access to the `invokeblock` block-handler distributions, creating the
    /// table on first use. Same reason as [`Self::super_cme_mut`].
    fn block_handlers_mut(&mut self) -> &mut HashMap<YarvInsnIdx, TypeDistribution> {
        self.marked_objects_stale = true;
        self.block_handlers.get_or_insert_with(Default::default)
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
        self.entry(insn_idx).map(|e| &*e.opnd_types).filter(|s| !s.is_empty())
    }

    pub fn get_splat_length_summary(&self, insn_idx: YarvInsnIdx) -> Option<SplatLengthDistributionSummary> {
        self.splat_lengths.as_ref()?.get(&insn_idx)
            .map(SplatLengthDistributionSummary::new)
    }

    pub fn get_super_method_entry(&self, insn_idx: YarvInsnIdx) -> Option<*const rb_callable_method_entry_t> {
        let summary = self.get_super_method_entries(insn_idx)?;

        if summary.is_monomorphic() {
            Some(summary.bucket(0).class.0 as *const rb_callable_method_entry_t)
        } else {
            None
        }
    }

    /// The whole distribution of frame method entries seen at an `invokesuper` site. A site with
    /// more than one is a `super` inside a method body that several classes run, most often a
    /// module method reached through more than one includer; each such method entry resolves
    /// `super` to a different target.
    pub fn get_super_method_entries(&self, insn_idx: YarvInsnIdx) -> Option<TypeDistributionSummary> {
        let entry = self.super_cme.as_ref()?.get(&insn_idx)?;
        Some(TypeDistributionSummary::new(entry))
    }

    /// Get the distribution of method-name symbols seen at a `send`/`__send__` call site.
    pub fn get_send_method_names(&self, insn_idx: YarvInsnIdx) -> Option<TypeDistributionSummary> {
        self.send_mid.as_ref()?.get(&insn_idx).map(TypeDistributionSummary::new)
    }

    /// The distribution of block handlers observed at an `invokeblock` site.
    pub fn get_block_handlers(&self, insn_idx: YarvInsnIdx) -> Option<TypeDistributionSummary> {
        self.block_handlers.as_ref()?.get(&insn_idx).map(TypeDistributionSummary::new)
    }

    /// Bytes this profile owns on the Rust heap, excluding the `IseqProfile`
    /// struct itself (which lives inside the `IseqPayload` allocation), plus
    /// counts describing what those bytes are spent on.
    pub fn heap_size(&self) -> ProfileHeapSize {
        let mut out = ProfileHeapSize::default();
        out.bytes = self.entries.capacity() * size_of::<ProfileEntry>()
            + self.marked_objects.capacity() * size_of::<VALUE>();
        out.marked_object_count = self.marked_objects.len();
        out.entry_slack_bytes = (self.entries.capacity() - self.entries.len()) * size_of::<ProfileEntry>();
        out.entry_count = self.entries.len();
        for entry in &self.entries {
            out.bytes += entry.opnd_types.len() * size_of::<TypeDistribution>();
            out.distribution_count += entry.opnd_types.len();
            for distribution in entry.opnd_types.iter() {
                out.bytes += distribution.heap_size();
                if distribution.num_buckets_used() <= 1 {
                    out.monomorphic_distribution_count += 1;
                }
            }
        }
        if let Some(super_cme) = self.super_cme.as_ref() {
            out.bytes += size_of::<HashMap<YarvInsnIdx, TypeDistribution>>();
            out.bytes += hash_table_bytes::<(YarvInsnIdx, TypeDistribution)>(super_cme.capacity());
            out.bytes += super_cme.values().map(TypeDistribution::heap_size).sum::<usize>();
        }
        if let Some(send_mid) = self.send_mid.as_ref() {
            out.bytes += size_of::<HashMap<YarvInsnIdx, TypeDistribution>>();
            out.bytes += hash_table_bytes::<(YarvInsnIdx, TypeDistribution)>(send_mid.capacity());
            out.bytes += send_mid.values().map(TypeDistribution::heap_size).sum::<usize>();
        }
        if let Some(splat_lengths) = self.splat_lengths.as_ref() {
            out.bytes += size_of::<HashMap<YarvInsnIdx, SplatLengthDistribution>>();
            out.bytes += hash_table_bytes::<(YarvInsnIdx, SplatLengthDistribution)>(splat_lengths.capacity());
            out.bytes += splat_lengths.values().map(SplatLengthDistribution::heap_size).sum::<usize>();
        }
        out
    }

    /// Run a given callback with every object in IseqProfile
    pub fn each_object(&self, mut callback: impl FnMut(VALUE)) {
        for entry in &self.entries {
            for distribution in &entry.opnd_types {
                for profiled_type in distribution.each_item() {
                    // If the type is a GC object, call the callback
                    callback(profiled_type.class);
                }
            }
        }

        for super_cme_values in self.super_cme.iter().flat_map(|map| map.values()) {
            for profiled_type in super_cme_values.each_item() {
                callback(profiled_type.class)
            }
        }

        for send_mid_values in self.send_mid.iter().flat_map(|map| map.values()) {
            for profiled_type in send_mid_values.each_item() {
                callback(profiled_type.class)
            }
        }

        for handler_values in self.block_handlers.iter().flat_map(|map| map.values()) {
            for profiled_type in handler_values.each_item() {
                callback(profiled_type.class)
            }
        }
    }

    /// Every object this profile references, as one dense array the GC can be handed
    /// directly.
    ///
    /// Walking the distributions to find them is what marking used to do on every
    /// major collection, for every live payload. That walk is a pointer chase through
    /// the whole profile -- the `entries` vector, a boxed operand-type slice per
    /// entry, a boxed tail per polymorphic distribution, and two hash tables -- so it
    /// touches tens of megabytes and takes a cache miss per object found, even though
    /// the answer only changes when something is profiled. So the answer is cached
    /// here and rebuilt only after a mutation, which in steady state (every hot ISEQ
    /// compiled, its instructions done profiling) is never.
    ///
    /// The cached array is exactly the set [`Self::each_object`] would yield, so
    /// marking it retains neither more nor less than before.
    ///
    /// A rebuild can allocate, and this runs inside the GC's mark phase. That is
    /// safe: ZJIT's Rust allocator goes to the system allocator, not to
    /// `ruby_xmalloc`, so it cannot re-enter the GC. (`GC.stress` over a workload
    /// with live payloads exercises exactly this.)
    pub fn marked_objects(&mut self) -> &[VALUE] {
        if self.marked_objects_stale {
            self.rebuild_marked_objects();
        }
        &self.marked_objects
    }

    /// Recompute [`Self::marked_objects`] from the distributions.
    #[cold]
    fn rebuild_marked_objects(&mut self) {
        // Reuse the allocation: this runs once per mutation-then-GC, and on a warming
        // application that is once per payload per collection.
        let objects = std::mem::take(&mut self.marked_objects);
        let mut objects = objects;
        objects.clear();
        self.each_object(|object| {
            // Deduplicate by linear scan. These arrays hold a handful of classes in
            // practice, so a scan beats a hash set (which would allocate per
            // payload); past DEDUP_LIMIT the scan would cost more than the duplicate
            // marks it saves, so stop looking and just append.
            const DEDUP_LIMIT: usize = 64;
            if objects.len() < DEDUP_LIMIT && objects.contains(&object) {
                return;
            }
            objects.push(object);
        });
        objects.shrink_to_fit();
        self.marked_objects = objects;
        self.marked_objects_stale = false;
    }

    /// Run a given callback with a mutable reference to every object in IseqProfile.
    pub fn each_object_mut(&mut self, mut callback: impl FnMut(&mut VALUE)) {
        // Compaction rewrites the objects in place, so the dense copy is now stale.
        // Rebuilding lazily rather than rewriting it here keeps the two ways of
        // producing it down to one.
        self.marked_objects_stale = true;
        for entry in &mut self.entries {
            for distribution in &mut entry.opnd_types {
                for ref mut profiled_type in distribution.each_item_mut() {
                    // If the type is a GC object, call the callback
                    callback(&mut profiled_type.class);
                }
            }
        }

        // Update CME references if they move during compaction.
        for super_cme_values in self.super_cme.iter_mut().flat_map(|map| map.values_mut()) {
            for ref mut profiled_type in super_cme_values.each_item_mut() {
                callback(&mut profiled_type.class)
            }
        }

        for send_mid_values in self.send_mid.iter_mut().flat_map(|map| map.values_mut()) {
            for ref mut profiled_type in send_mid_values.each_item_mut() {
                callback(&mut profiled_type.class)
            }
        }

        for handler_values in self.block_handlers.iter_mut().flat_map(|map| map.values_mut()) {
            for ref mut profiled_type in handler_values.each_item_mut() {
                callback(&mut profiled_type.class)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ZJIT retains one of these per profiled operand of every profiled instruction, and
    /// most stay monomorphic for the life of the process, so their size is the single
    /// biggest lever on ZJIT's non-code memory. Pin it: widening the inline part of
    /// `Distribution`, or letting `ProfileEntry` carry a `Vec` again, costs megabytes on a
    /// compile-heavy workload without anything else failing.
    #[test]
    fn profiling_metadata_stays_small() {
        // Bucket 0 inline (16B) + two u16 counts + a null tail pointer.
        assert_eq!(size_of::<TypeDistribution>(), 32);
        assert_eq!(size_of::<ProfileEntry>(), 32,
            "opnd_types must stay a Box<[_]> (16 bytes), not a Vec (24)");

        // What the same distribution would cost with all 8 buckets inline, which is
        // what it cost before the tail was boxed: [ProfiledType; 8] + [u16; 8] + other.
        const INLINE_SIZE: usize = 8 * size_of::<ProfiledType>() + 8 * size_of::<NumProfiles>()
            + size_of::<NumProfiles>();
        assert!(INLINE_SIZE >= 146 && size_of::<TypeDistribution>() * 4 < INLINE_SIZE,
            "the inline layout was {INLINE_SIZE} bytes; boxing has to be worth much more than a small constant");

        // The tail a distribution pays for only once it goes polymorphic.
        let mut dist = TypeDistribution::new();
        let a = ProfiledType::object(VALUE(8));
        let b = ProfiledType::object(VALUE(16));
        dist.observe(a);
        assert_eq!(dist.heap_size(), 0, "a monomorphic distribution owns no heap");
        dist.observe(b);
        assert_eq!(dist.heap_size(), 8 * size_of::<ProfiledType>() + 8 * size_of::<NumProfiles>(),
            "the boxed tail is buckets 1..8 plus their counts, with index 0 left unused so \
             bucket numbering lines up with Distribution's");
    }

    #[test]
    fn can_profile_block_handler() {
        with_rubyvm(|| eval("
            def foo = yield
            foo rescue 0
            foo rescue 0
        "));
    }
}
