//! Profiler for runtime information.

// We use the YARV bytecode constants which have a CRuby-style name
#![allow(non_upper_case_globals)]

use std::collections::HashMap;
use crate::{cruby::*, payload::get_or_create_iseq_payload, options::{get_option, NumProfiles}};
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

pub const DISTRIBUTION_SIZE: usize = 8;

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
            entry.opnd_types.resize(1, TypeDistribution::new());
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
            crate::codegen::invalidate_iseq_version(cb, compiled_iseq, version);
            cb.mark_all_executable();
        }
    });
}

/// Samples a `yield` site's generic fallback path has to see before it decides whether the
/// handlers arriving there are worth a recompile. Same trade as [`IVAR_REPROFILE_WINDOW`]: small
/// enough that an iterator frozen on a boot-time profile recovers within the first requests,
/// large enough that a brief detour through an unusual handler does not spend a version.
const BLOCK_REPROFILE_WINDOW: u16 = 64;

/// Share of a window, in percent, that the handlers a recompile would give an arm to must
/// account for together before the recompile is worth a version. Matches the HIR builder's
/// `CHAIN_COVERAGE_THRESHOLD`, which is the bar the rebuilt dispatch itself has to clear:
/// asking for less here would earn a version and then decline to use it.
const BLOCK_REPROFILE_MIN_SHARE_PERCENT: u32 = 50;

/// How many windows of live traffic one `yield` site may fold into its handler profile.
///
/// This bounds how often the *profile* is rewritten, not how many recompiles it can cause. A
/// site inside a shared iterator is compiled once standalone and again inside every caller that
/// inlines it, and all of those copies read the one profile: the first window to close fixes it
/// for all of them, and the rest only need to notice that their own dispatch is behind. Capping
/// the recompiles here instead would fix whichever compiled units happened to sample first and
/// freeze every other one for the life of the process.
const MAX_BLOCK_PROFILE_REFRESHES: u8 = 3;

/// Live-traffic re-profiling state for one `invokeblock` site. Only sites whose compiled
/// dispatch actually reaches `rb_vm_invokeblock()` ever get one.
#[derive(Debug)]
struct BlockFallbackEntry {
    /// Block handlers seen on the generic fallback path in the current window. Built fresh
    /// rather than added to the site's profile directly: the profile is what the *running* code
    /// was compiled from, and the decision below is about what the fallback alone is seeing.
    dist: TypeDistribution,
    /// Classes of the first `yield`ed argument, recorded only for the samples whose handler was
    /// a Symbol. A Symbol block turns the `yield` into a send to that argument, and that send
    /// only compiles if its receiver's class resolves -- but the site's operand profile holds
    /// the class of everything the iterator ever yielded, which at `Array#each` is everything in
    /// the process. Recorded here it is conditioned on the handler being a Symbol, which is the
    /// only traffic the arm dispatches over.
    symbol_recv: TypeDistribution,
    /// Samples taken in the current window.
    samples: u16,
    /// How many windows this site has already folded into its profile.
    refreshes: u8,
}

/// Result of [`IseqProfile::observe_block_fallback`].
#[derive(PartialEq, Eq, Debug)]
enum BlockReprofiled {
    /// Recorded, mid-window. Nothing to do.
    Sampled,
    /// The window closed without making the case for a recompile.
    Declined,
    /// The window closed with most of its samples on handlers the compiled dispatch has no arm
    /// for. They are now in the site's handler profile; recompile to pick them up.
    Recompile,
}

/// Called from JIT code on the generic `rb_vm_invokeblock()` fallback of a compiled `yield`.
///
/// The shared core iterators -- `Array#each`, `Array#map`, `Array#select` in
/// `<internal:array>` -- are called enough times during boot to be compiled there, from a
/// profile of their first few dozen yields. That profile contains the IFUNC handlers the C
/// runtime passes them and none of the ISEQ blocks or `&:sym` handlers the application will.
/// Nothing refreshes it either: the dispatch those iterators got does not side-exit on a miss,
/// it branches to `rb_vm_invokeblock()`, so no guard ever fails and no recompile is ever
/// triggered. On lobsters that froze 875K yields per run at one call site.
///
/// So sample what the fallback is handed. Once a window's worth of samples is mostly handlers a
/// rebuilt dispatch could serve, fold the window into the site's handler profile -- dropping the
/// cold buckets the boot-time traffic is holding, the way
/// [`IseqProfile::observe_ivar_fallback`] does for shapes -- grant the ISEQ one extra version,
/// and invalidate the compiled unit to spend it.
///
/// `version` is the compiled unit this call lives in, not the ISEQ owning the instruction. See
/// [`rb_zjit_ivar_reprofile`], whose structure this follows exactly.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_block_reprofile(version: *mut crate::payload::IseqVersion, frame_iseq: VALUE, insn_idx: u32, block_handler: VALUE, argc: u32, arg0: VALUE) {
    // JIT code only reaches here once the countdown has run out, so this is the point at which
    // the next one is set. Anything that returns early below has to leave one behind or the site
    // would call in on every fallback for the rest of the process.
    fn set_countdown(version: *mut crate::payload::IseqVersion, value: u32) {
        unsafe { (*version).block_reprofile_countdown = value };
    }
    if unsafe { (*version).is_invalidated() } {
        // Frames still running the invalidated code have nothing to earn.
        set_countdown(version, u32::MAX);
        return;
    }
    // No block was passed at all, so this `yield` is on its way to raising. Nothing to specialize.
    if block_handler == VALUE(VM_BLOCK_HANDLER_NONE as usize) {
        set_countdown(version, crate::payload::BLOCK_REPROFILE_COOLDOWN);
        return;
    }
    let untagged = unsafe { rb_vm_untag_block_handler(block_handler) };
    let insn_idx = insn_idx as YarvInsnIdx;
    let frame_iseq = frame_iseq.as_iseq();
    let reprofiled = with_time_stat(profile_time_ns, || {
        get_or_create_iseq_payload(frame_iseq).profile
            .observe_block_fallback(frame_iseq, insn_idx, untagged, argc as usize, arg0)
    });
    match reprofiled {
        BlockReprofiled::Sampled => {
            set_countdown(version, 0);
            return;
        }
        BlockReprofiled::Declined => {
            // Spend one of the version's windows. When they run out the site goes dormant for a
            // cooldown and then opens a fresh set: what a shared iterator yields to changes
            // completely between boot and steady state, so a verdict is only ever provisional.
            let windows = unsafe { &mut (*version).block_reprofile_windows };
            *windows = windows.saturating_sub(1);
            if *windows == 0 {
                crate::stats::incr_counter!(block_respecialize_giveup_count);
                *windows = crate::payload::MAX_BLOCK_REPROFILE_WINDOWS;
                set_countdown(version, crate::payload::BLOCK_REPROFILE_COOLDOWN);
            } else {
                set_countdown(version, 0);
            }
            return;
        }
        BlockReprofiled::Recompile => set_countdown(version, 0),
    }
    // `version` points into the payload, and holding a reference to it across the lock's unwind
    // boundary is not allowed: read the compiled unit's ISEQ out first, and let the closure
    // report back whether the site has anything left to earn rather than writing it itself.
    let compiled_iseq = VALUE::from(unsafe { (*version).iseq });
    let out_of_versions = with_vm_lock(src_loc!(), || {
        let compiled_iseq = compiled_iseq.as_iseq();
        let payload = get_or_create_iseq_payload(compiled_iseq);
        if payload.block_respecializations >= crate::payload::MAX_BLOCK_RESPECIALIZATIONS {
            // Out of versions: nothing this site reports can be acted on any more. Leave the
            // call out of whatever this ISEQ compiles next, too.
            payload.block_reprofile_giveup = true;
            return true;
        }
        payload.block_respecializations += 1;
        crate::stats::incr_counter!(block_respecialize_count);
        if let Some(version) = payload.versions.last_mut() {
            let cb = crate::state::ZJITState::get_code_block();
            // The respecialization budget was just raised above, so this is not the
            // grant path in invalidate_iseq_version(): it simply fits under the new limit.
            crate::codegen::invalidate_iseq_version(cb, compiled_iseq, version);
            cb.mark_all_executable();
        }
        false
    });
    if out_of_versions {
        set_countdown(version, u32::MAX);
    }
}

fn profile_block_handler(profiler: &mut Profiler, profile: &mut IseqProfile) {
    let obj = profiler.peek_at_block_handler();
    let ty = ProfiledType::object(obj);
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
        entry.opnd_types.resize(1, TypeDistribution::new());
    }

    let level = profiler.insn_opnd(1).as_u32();
    let ep = unsafe { get_cfp_ep_level(profiler.cfp, level) };
    let block_handler = unsafe { *ep.offset(VM_ENV_DATA_INDEX_SPECVAL as isize) };
    let untagged = unsafe { rb_vm_untag_block_handler(block_handler) };

    let ty = ProfiledType::object(untagged);
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

    pub fn none() -> Self { Self(Self::NONE) }

    pub fn immediate() -> Self { Self(Self::IS_IMMEDIATE) }
    pub fn is_immediate(self) -> bool { (self.0 & Self::IS_IMMEDIATE) != 0 }
    pub fn is_embedded(self) -> bool { (self.0 & Self::IS_EMBEDDED) != 0 }
    pub fn is_t_object(self) -> bool { (self.0 & Self::IS_T_OBJECT) != 0 }
    pub fn is_struct_embedded(self) -> bool { (self.0 & Self::IS_STRUCT_EMBEDDED) != 0 }
    pub fn is_object_profiling(self) -> bool { (self.0 & Self::IS_OBJECT_PROFILING) != 0 }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfiledType {
    class: VALUE,
    shape: ShapeId,
    flags: Flags,
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

#[derive(Debug)]
pub struct IseqProfile {
    /// Sparse storage of per-instruction profile data, sorted by instruction index.
    /// Only instructions that have actually been profiled have entries here.
    entries: Vec<ProfileEntry>,

    /// Method entries for `super` calls (stored as VALUE to be GC-safe)
    super_cme: HashMap<YarvInsnIdx, TypeDistribution>,

    /// Observed lengths of caller splat arrays for call instructions.
    splat_lengths: HashMap<YarvInsnIdx, SplatLengthDistribution>,

    /// Block handlers observed at `invokeblock` sites (stored as VALUE to be GC-safe).
    /// Kept out of `opnd_types` so that the entry's operand slots profile the yielded
    /// arguments instead: a `yield` to a Symbol block sends to the first argument, and
    /// that send only specializes if the argument's class was profiled.
    block_handlers: HashMap<YarvInsnIdx, TypeDistribution>,

    /// Live-traffic re-profiling windows for `invokeblock` sites whose compiled dispatch keeps
    /// reaching `rb_vm_invokeblock()`.
    block_fallbacks: HashMap<YarvInsnIdx, BlockFallbackEntry>,
}

impl IseqProfile {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            super_cme: HashMap::new(),
            splat_lengths: HashMap::new(),
            block_handlers: HashMap::new(),
            block_fallbacks: HashMap::new(),
        }
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

    /// Mutable access to the `invokesuper` method-entry distributions.
    fn super_cme_mut(&mut self) -> &mut HashMap<YarvInsnIdx, TypeDistribution> {
        &mut self.super_cme
    }

    /// Mutable access to the splat length distributions.
    fn splat_lengths_mut(&mut self) -> &mut HashMap<YarvInsnIdx, SplatLengthDistribution> {
        &mut self.splat_lengths
    }

    /// Mutable access to the `invokeblock` block-handler distributions.
    fn block_handlers_mut(&mut self) -> &mut HashMap<YarvInsnIdx, TypeDistribution> {
        &mut self.block_handlers
    }

    /// Record one block handler that reached a compiled `yield`'s generic fallback, and say
    /// whether the window it closed makes the case for recompiling the site.
    /// See [`rb_zjit_block_reprofile`].
    fn observe_block_fallback(&mut self, iseq: IseqPtr, insn_idx: YarvInsnIdx, handler: VALUE, argc: usize, arg0: VALUE) -> BlockReprofiled {
        let entry = self.block_fallbacks
            .entry(insn_idx).or_insert_with(|| BlockFallbackEntry {
                dist: TypeDistribution::new(), symbol_recv: TypeDistribution::new(),
                samples: 0, refreshes: 0,
            });
        let ty = ProfiledType::object(handler);
        if unsafe { crate::cruby::rb_IMEMO_TYPE_P(ty.class(), crate::cruby::imemo_iseq) == 1 } {
            crate::stats::incr_counter!(block_fallback_sample_iseq);
        } else if ty.class().static_sym_p() {
            crate::stats::incr_counter!(block_fallback_sample_symbol);
        } else {
            crate::stats::incr_counter!(block_fallback_sample_other);
        }
        VALUE::from(iseq).write_barrier(ty.class());
        entry.dist.observe(ty);
        if handler.static_sym_p() && argc > 0 && !arg0.special_const_p() {
            let recv_ty = ProfiledType::new(arg0);
            VALUE::from(iseq).write_barrier(recv_ty.class());
            entry.symbol_recv.observe(recv_ty);
        }
        entry.samples += 1;
        if entry.samples < BLOCK_REPROFILE_WINDOW {
            return BlockReprofiled::Sampled;
        }
        entry.samples = 0;
        let summary = TypeDistributionSummary::new(&entry.dist);
        let symbol_recv = TypeDistributionSummary::new(&entry.symbol_recv);
        let share = crate::hir::block_fallback_specializable_share(&summary, &symbol_recv, argc);
        if (share * 100.0) < BLOCK_REPROFILE_MIN_SHARE_PERCENT as f64 {
            // Start the next window from scratch rather than accumulating: what matters is what
            // the fallback is seeing now, not what it saw before the last verdict.
            entry.dist = TypeDistribution::new();
            entry.symbol_recv = TypeDistribution::new();
            crate::stats::incr_counter!(block_respecialize_declined_count);
            return BlockReprofiled::Declined;
        }
        let refresh = entry.refreshes < MAX_BLOCK_PROFILE_REFRESHES;
        entry.refreshes = entry.refreshes.saturating_add(1);
        let window = std::mem::replace(&mut entry.dist, TypeDistribution::new());
        let recv_window = std::mem::replace(&mut entry.symbol_recv, TypeDistribution::new());
        if !refresh {
            // The profile already holds what this window would add -- some other compiled unit's
            // window folded it in. The dispatch *running here* still has no arm for it, so this
            // version has just as much reason to recompile; it simply needs no profile change.
            return BlockReprofiled::Recompile;
        }

        // Fold the window into the profile the next compile will read. The buckets already there
        // are what the running dispatch was built from -- for these sites, boot-time traffic
        // that has not been seen since -- and there are only eight of them, so keep the hottest
        // one (the handler kind the dispatch is ordered around) and give the rest to live
        // traffic. Replaying the window's counts rather than assigning them keeps the
        // distribution's "bucket 0 is the most common item" invariant.
        /// Add every bucket of `window` to `dist` with the count it was seen with. Replaying the
        /// counts rather than assigning them keeps the "bucket 0 is the most common item"
        /// invariant that `DistributionSummary::new` asserts on.
        fn replay_into(dist: &mut TypeDistribution, window: &TypeDistribution) {
            for idx in 0..DISTRIBUTION_SIZE {
                let count = window.count(idx);
                if count == 0 { continue; }
                let item = window.bucket(idx);
                for _ in 0..count {
                    dist.observe(item);
                }
            }
        }

        let handlers = self.block_handlers
            .entry(insn_idx).or_insert_with(TypeDistribution::new);
        handlers.retain_primary();
        replay_into(handlers, &window);

        // The Symbol arms the recompile emits send to the first `yield`ed argument and resolve
        // that send from the operand profile. Refresh it from the receivers this window saw
        // while the handler was a Symbol, for the same reason and in the same way: what the
        // iterator yields to its other handlers says nothing about what a `&:sym` block is
        // about to be called on.
        if recv_window.num_observed() > 0 {
            let entry = self.entry_mut(insn_idx);
            if entry.opnd_types.is_empty() {
                entry.opnd_types.resize(argc.max(1), TypeDistribution::new());
            }
            let recv_types = &mut entry.opnd_types[0];
            recv_types.retain_primary();
            replay_into(recv_types, &recv_window);
        }
        BlockReprofiled::Recompile
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

    /// The distribution of block handlers observed at an `invokeblock` site.
    pub fn get_block_handlers(&self, insn_idx: YarvInsnIdx) -> Option<TypeDistributionSummary> {
        self.block_handlers.get(&insn_idx).map(TypeDistributionSummary::new)
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

        for handler_values in self.block_handlers.values() {
            for profiled_type in handler_values.each_item() {
                callback(profiled_type.class)
            }
        }

        for fallback in self.block_fallbacks.values() {
            for profiled_type in fallback.dist.each_item().chain(fallback.symbol_recv.each_item()) {
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

        for handler_values in self.block_handlers.values_mut() {
            for ref mut profiled_type in handler_values.each_item_mut() {
                callback(&mut profiled_type.class)
            }
        }

        for fallback in self.block_fallbacks.values_mut() {
            for ref mut profiled_type in fallback.dist.each_item_mut() {
                callback(&mut profiled_type.class)
            }
            for ref mut profiled_type in fallback.symbol_recv.each_item_mut() {
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
