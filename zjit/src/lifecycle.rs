//! The per-ISEQ function lifecycle of ZJIT as a single state machine.
//!
//! Every decision about *when* an ISEQ is profiled, compiled, invalidated, or
//! recompiled is made in this module, and every such decision reads its tuning
//! knob from [`Knobs`]. Other modules only provide the *mechanisms* (rewriting
//! YARV instructions to `zjit_*` variants, generating code, patching code) and
//! report events here through the `on_*` functions at the bottom of this file.
//!
//! ```text
//!               InterpCall: calls == profile_threshold     InterpCall: calls == call_threshold
//!               StubHit:    calls <  profile_threshold     StubHit:    calls >= call_threshold
//!               [StartProfiling]                           [Compile] if num_versions < max_versions
//!  Interpreted ---------------------------> Profiling ------------------------------> (compiling)
//!                                                                                      |      |
//!                                     CompileSucceeded (HirBuilt: [RearmProfiling]     |      | CompileFailed
//!                                     unless it's the final version)                   |      v
//!                    +-----------------------------------------------------------------+  CantCompile
//!                    v                                                                     (terminal)
//!              Compiled(vN) --- RecompileExit x num_exits_until_invalidate ---+
//!                |   |                                                        |
//!                |   +------------- Invalidate(PatchPoint) -------------------+ if N < max_versions
//!                |                                                            v
//!                |                                                 [Invalidate] (calls = 0)
//!                | Invalidate(EpEscape | TracePoint)                Invalidated(vN)
//!                | at N == max_versions                                       |
//!                v                                    InterpCall: calls == profile_threshold
//!       VersionLimitReached                           [StartProfiling]        v
//!       (interpreter only)                                           Reprofiling(vN)
//!                                                                             | [Compile] as above
//!                                                                             v
//!                                                         (compiling) --> Compiled(vN+1)
//! ```
//!
//! Every transition and the knob it reads:
//!
//! | From                       | Event             | Knob (guard)                          | To / Actions                    |
//! |----------------------------|-------------------|---------------------------------------|---------------------------------|
//! | Interpreted, Invalidated   | InterpCall        | calls == profile_threshold            | Profiling, Reprofiling / StartProfiling |
//! | Profiling, Reprofiling     | InterpCall        | calls == call_threshold               | Compile                         |
//! | no compiled code           | StubHit           | calls < profile_threshold             | calls = profile_threshold / StartProfiling |
//! | no compiled code           | StubHit           | calls >= call_threshold               | Compile                         |
//! | any                        | CompileRequested  | num_versions < max_versions           | Compile, else VersionLimitReached |
//! | (compiling)                | HirBuilt          | num_versions + 1 < max_versions       | RearmProfiling (num_profiles)   |
//! | (compiling)                | CompileSucceeded  | budget = num_exits_until_invalidate   | Compiled(vN+1)                  |
//! | (compiling)                | CompileFailed     |                                       | CantCompile                     |
//! | Compiled(vN)               | RecompileExit     | budget-- == 0, N < max_versions       | RequestInvalidation             |
//! | Compiled(vN)               | Invalidate(PatchPoint, RecompileExit) | N < max_versions      | Invalidated(vN) / Invalidate    |
//! | Compiled(vN)               | Invalidate(EpEscape)  |                                   | Invalidated(vN) / Invalidate    |
//! | any                        | Invalidate(TracePoint)|                                   | calls = 0 / Invalidate          |
//! | (on Invalidate other than TracePoint) | | num_reprofiles = Some(n)                   | calls = call_threshold - n / RearmProfiling (n) |
//! | any                        | OutOfMemory       |                                       | StopCompiling (global)          |
//!
//! `profile_threshold` is `call_threshold - num_profiles` (see [`Knobs::profile_threshold`]).
//! Profiled instructions additionally count down a per-instruction budget of
//! `num_profiles` samples ([`on_insn_profiled`]) before reverting to the bare instruction.
//!
//! Out of executable memory, [`Actions::STOP_COMPILING`] turns off `rb_zjit_compiling_p`
//! globally so the interpreter stops counting calls, until memory is available again.
//!
//! The state is not stored separately: [`State::of`] derives it from the
//! interpreter call counter (`jit_entry_calls`) and the ISEQ's
//! [`IseqPayload::versions`], which are the ground truth the rest of ZJIT
//! reads. [`on_event`] is a pure function over a [`Snapshot`] of those fields,
//! so the whole policy can be unit-tested without a VM.
//!
//! The hot path stays in C: `zjit_compile()` in vm.c increments
//! `jit_entry_calls` on each interpreted call and compares it against
//! [`rb_zjit_profile_threshold`] and [`rb_zjit_call_threshold`], which are
//! published by [`publish_thresholds`] from the knobs. Its [`Event::InterpCall`]
//! rule below must stay in sync with that C code. When a threshold is hit, vm.c
//! calls back into this module ([`rb_zjit_start_profiling`] or the compile
//! entry point).
//!
//! `--zjit-trace-lifecycle` prints every transition that changes the state or
//! takes an action to stderr.

use crate::cruby::*;
use crate::options::{get_option, CallThreshold, NumExits, NumProfiles, OPTIONS, DEFAULT_CALL_THRESHOLD, DEFAULT_MAX_VERSIONS, DEFAULT_NUM_EXITS_UNTIL_INVALIDATE, DEFAULT_NUM_PROFILES};
use crate::payload::{IseqPayload, IseqStatus, IseqVersionRef, get_iseq_payload_ptr};
use crate::profile::reset_profiles_remaining;
use crate::state::rb_zjit_compiling_p;

/// Number of calls to start profiling YARV instructions.
/// They are profiled `rb_zjit_call_threshold - rb_zjit_profile_threshold` times,
/// which is equal to --zjit-num-profiles. Published by [`publish_thresholds`].
#[unsafe(no_mangle)]
#[allow(non_upper_case_globals)]
pub static mut rb_zjit_profile_threshold: CallThreshold = DEFAULT_CALL_THRESHOLD - DEFAULT_NUM_PROFILES as CallThreshold;

/// Number of calls to compile ISEQ with ZJIT at jit_compile() in vm.c.
/// --zjit-call-threshold=1 compiles on first execution without profiling information.
#[unsafe(no_mangle)]
#[allow(non_upper_case_globals)]
pub static mut rb_zjit_call_threshold: CallThreshold = DEFAULT_CALL_THRESHOLD;

/// The tuning knobs of the lifecycle. Each transition in [`on_event`] reads its knob from here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Knobs {
    /// --zjit-call-threshold: interpreter calls before compiling
    pub call_threshold: CallThreshold,
    /// --zjit-num-profiles: calls profiled before compiling, and samples per instruction
    pub num_profiles: NumProfiles,
    /// --zjit-num-reprofiles: calls profiled after an invalidation before recompiling.
    /// None re-runs the whole call_threshold warmup.
    pub num_reprofiles: Option<NumProfiles>,
    /// --zjit-num-exits-until-invalidate: recompile exits before invalidating a version
    pub num_exits_until_invalidate: NumExits,
    /// --zjit-max-versions: maximum number of compiled versions per ISEQ
    pub max_versions: usize,
}

impl Default for Knobs {
    fn default() -> Self {
        Knobs {
            call_threshold: DEFAULT_CALL_THRESHOLD,
            num_profiles: DEFAULT_NUM_PROFILES,
            num_reprofiles: None,
            num_exits_until_invalidate: DEFAULT_NUM_EXITS_UNTIL_INVALIDATE,
            max_versions: DEFAULT_MAX_VERSIONS,
        }
    }
}

impl Knobs {
    /// Read the current knobs from the command-line options
    pub fn get() -> Knobs {
        let call_threshold = unsafe { rb_zjit_call_threshold };
        match unsafe { OPTIONS.as_ref() } {
            Some(opts) => Knobs {
                call_threshold,
                num_profiles: opts.num_profiles,
                num_reprofiles: opts.num_reprofiles,
                num_exits_until_invalidate: opts.num_exits_until_invalidate,
                max_versions: opts.max_versions,
            },
            None => Knobs { call_threshold, ..Knobs::default() },
        }
    }

    /// Number of calls at which the interpreter starts profiling instructions.
    pub fn profile_threshold(&self) -> CallThreshold {
        if self.call_threshold == 1 {
            // If --zjit-call-threshold=1, never rewrite ISEQs to profile instructions.
            0
        } else {
            // Otherwise, profile instructions at least once.
            self.call_threshold.saturating_sub(self.num_profiles.into()).max(1)
        }
    }
}

/// Publish the thresholds that vm.c compares the interpreter call counter against.
/// Must be called whenever a knob they depend on changes.
pub fn publish_thresholds() {
    unsafe { rb_zjit_profile_threshold = Knobs::get().profile_threshold(); }
}

/// Set --zjit-call-threshold and republish the thresholds
pub fn set_call_threshold(call_threshold: CallThreshold) {
    unsafe { rb_zjit_call_threshold = call_threshold; }
    publish_thresholds();
}

/// Maximum number of compiled versions per ISEQ (--zjit-max-versions)
pub fn max_iseq_versions() -> usize {
    unsafe { OPTIONS.as_ref() }.map_or(DEFAULT_MAX_VERSIONS, |opts| opts.max_versions)
}

/// Number of times each YARV instruction is profiled before it is reverted to the bare instruction.
pub fn insn_profile_budget() -> NumProfiles {
    get_option!(num_profiles)
}

/// Number of recompile exits a fresh version tolerates before it is invalidated.
pub fn initial_exit_budget() -> NumExits {
    get_option!(num_exits_until_invalidate)
}

/// Count one profile sample of an instruction. Return true if the instruction
/// has been profiled enough and should be reverted to the bare instruction.
#[inline]
pub fn on_insn_profiled(profiles_remaining: &mut NumProfiles) -> bool {
    *profiles_remaining = profiles_remaining.saturating_sub(1);
    *profiles_remaining == 0
}

/// Status of the latest compiled version of an ISEQ
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LastVersion {
    /// Never compiled
    None,
    /// The latest version is live code
    Compiled,
    /// The latest version failed to compile. Terminal.
    CantCompile,
    /// The latest version was invalidated
    Invalidated,
}

/// The lifecycle-relevant fields of an ISEQ. [`on_event`] reads and updates these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    /// Interpreter call counter (`jit_entry_calls` in `rb_iseq_constant_body`)
    pub calls: CallThreshold,
    /// Number of versions compiled so far, including failed and invalidated ones
    pub num_versions: usize,
    /// Status of the latest version
    pub last_version: LastVersion,
    /// Remaining recompile exits before the live version is invalidated
    pub exits_until_invalidate: NumExits,
}

/// The lifecycle state of an ISEQ. Version numbers are 1-based.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// Counting interpreted calls, not profiling yet
    Interpreted,
    /// Profiling instructions in the interpreter, waiting to be compiled
    Profiling,
    /// Running compiled version N
    Compiled(usize),
    /// Version N was invalidated, counting interpreted calls again
    Invalidated(usize),
    /// Version N was invalidated, profiling again for version N+1
    Reprofiling(usize),
    /// The latest version failed to compile. Never compiled again.
    CantCompile,
    /// No live code and no more versions may be compiled
    VersionLimitReached,
}

impl State {
    /// Derive the state of an ISEQ from its [`Snapshot`]
    pub fn of(snapshot: &Snapshot, knobs: &Knobs) -> State {
        let n = snapshot.num_versions;
        match snapshot.last_version {
            LastVersion::Compiled => State::Compiled(n),
            LastVersion::CantCompile => State::CantCompile,
            LastVersion::None | LastVersion::Invalidated => {
                let profiling = snapshot.calls >= knobs.profile_threshold();
                if n >= knobs.max_versions {
                    State::VersionLimitReached
                } else if n == 0 {
                    if profiling { State::Profiling } else { State::Interpreted }
                } else if profiling {
                    State::Reprofiling(n)
                } else {
                    State::Invalidated(n)
                }
            }
        }
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            State::Compiled(n) => write!(f, "Compiled(v{n})"),
            State::Invalidated(n) => write!(f, "Invalidated(v{n})"),
            State::Reprofiling(n) => write!(f, "Reprofiling(v{n})"),
            _ => write!(f, "{self:?}"),
        }
    }
}

/// Why a version is being invalidated
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidationCause {
    /// An assumption baked into a PatchPoint (BOP, CME, constant, ...) was broken
    PatchPoint,
    /// The version took --zjit-num-exits-until-invalidate recompile exits
    RecompileExit,
    /// The ISEQ's EP escaped. Its code must not run again, even at the version limit.
    EpEscape,
    /// A TracePoint was enabled. All code must stop running.
    TracePoint,
}

impl InvalidationCause {
    /// Whether this invalidation must happen even at the version limit
    fn is_forced(self) -> bool {
        matches!(self, InvalidationCause::EpEscape | InvalidationCause::TracePoint)
    }
}

/// An event that may move an ISEQ to another [`State`]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// The interpreter called the ISEQ while it has no compiled code (vm.c `zjit_compile()`)
    InterpCall,
    /// Compiled code called the ISEQ through a function stub
    StubHit,
    /// A compile is about to create a new version
    CompileRequested,
    /// HIR for the next version has been built and optimized
    HirBuilt,
    /// A new version has been compiled
    CompileSucceeded,
    /// A new version failed to compile
    CompileFailed,
    /// Compiled code took a side exit that asks for recompilation
    RecompileExit,
    /// A version is being invalidated
    Invalidate { cause: InvalidationCause, already_invalidated: bool },
    /// ZJIT ran out of executable memory
    OutOfMemory,
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Event::Invalidate { cause, already_invalidated: false } => write!(f, "Invalidate({cause:?})"),
            Event::Invalidate { cause, already_invalidated: true } => write!(f, "Invalidate({cause:?}, already invalidated)"),
            _ => write!(f, "{self:?}"),
        }
    }
}

/// Set of actions that [`on_event`] asks the caller to perform
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Actions(u8);

impl Actions {
    pub const NOTHING: Actions = Actions(0);
    /// Rewrite the ISEQ's YARV instructions to profiling `zjit_*` instructions
    pub const START_PROFILING: Actions = Actions(1 << 0);
    /// Compile a new version now
    pub const COMPILE: Actions = Actions(1 << 1);
    /// Reset per-instruction profile budgets and keep profiling in the interpreter
    /// (e.g. on side exits) so that the next version gets fresh profiles
    pub const REARM_PROFILING: Actions = Actions(1 << 2);
    /// Take the VM lock and dispatch an [`Event::Invalidate`]
    pub const REQUEST_INVALIDATION: Actions = Actions(1 << 3);
    /// Mark the version invalidated, reset call counters, and re-stub incoming calls
    pub const INVALIDATE: Actions = Actions(1 << 4);
    /// Stop compiling anything
    pub const STOP_COMPILING: Actions = Actions(1 << 5);

    const NAMES: [(Actions, &'static str); 6] = [
        (Actions::START_PROFILING, "StartProfiling"),
        (Actions::COMPILE, "Compile"),
        (Actions::REARM_PROFILING, "RearmProfiling"),
        (Actions::REQUEST_INVALIDATION, "RequestInvalidation"),
        (Actions::INVALIDATE, "Invalidate"),
        (Actions::STOP_COMPILING, "StopCompiling"),
    ];

    pub fn contains(self, other: Actions) -> bool {
        self.0 & other.0 == other.0 && other.0 != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for Actions {
    type Output = Actions;
    fn bitor(self, rhs: Actions) -> Actions {
        Actions(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Actions {
    fn bitor_assign(&mut self, rhs: Actions) {
        self.0 |= rhs.0;
    }
}

impl std::fmt::Debug for Actions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return write!(f, "Nothing");
        }
        let mut sep = "";
        for (action, name) in Actions::NAMES {
            if self.contains(action) {
                write!(f, "{sep}{name}")?;
                sep = "|";
            }
        }
        Ok(())
    }
}

/// The result of [`on_event`]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub from: State,
    pub event: Event,
    pub to: State,
    pub actions: Actions,
}

/// The transition function of the lifecycle. Update `snapshot` for `event` and
/// return the actions the caller has to perform. This is pure: it only reads `knobs`.
pub fn on_event(snapshot: &mut Snapshot, event: Event, knobs: &Knobs) -> Transition {
    let from = State::of(snapshot, knobs);
    let mut actions = Actions::NOTHING;
    match event {
        Event::InterpCall => {
            // Mirrors zjit_compile() and jit_compile_exception() in vm.c
            snapshot.calls = snapshot.calls.wrapping_add(1);
            if snapshot.calls == knobs.profile_threshold() {
                actions |= Actions::START_PROFILING;
            }
            if snapshot.calls == knobs.call_threshold {
                actions |= Actions::COMPILE;
            }
        }
        Event::StubHit => {
            if snapshot.calls < knobs.profile_threshold() {
                // Skip the unprofiled warmup. The compiled caller already establishes
                // that the callee is hot, so go straight to the profiling window.
                snapshot.calls = knobs.profile_threshold();
                actions |= Actions::START_PROFILING;
            } else {
                snapshot.calls = snapshot.calls.wrapping_add(1);
            }
            // Otherwise, exit to the interpreter until the callee collects enough profiles.
            if snapshot.calls >= knobs.call_threshold {
                actions |= Actions::COMPILE;
            }
        }
        Event::CompileRequested => {
            if snapshot.num_versions < knobs.max_versions {
                actions |= Actions::COMPILE;
            }
        }
        Event::HirBuilt => {
            // Unless this is the final version, keep profiling in the interpreter so
            // that side exits of this version collect profiles for the next one.
            if snapshot.num_versions + 1 < knobs.max_versions {
                actions |= Actions::REARM_PROFILING;
            }
        }
        Event::CompileSucceeded => {
            snapshot.num_versions += 1;
            snapshot.last_version = LastVersion::Compiled;
            // Give the new version a fresh budget of recompile exits.
            snapshot.exits_until_invalidate = knobs.num_exits_until_invalidate;
        }
        Event::CompileFailed => {
            snapshot.num_versions += 1;
            snapshot.last_version = LastVersion::CantCompile;
        }
        Event::RecompileExit => {
            // Skip if the version is already invalidated or it's the final version.
            let already_done = snapshot.last_version == LastVersion::Invalidated
                || snapshot.num_versions >= knobs.max_versions;
            if !already_done {
                // Wait for the configured number of recompile exits before invalidation.
                snapshot.exits_until_invalidate = snapshot.exits_until_invalidate.saturating_sub(1);
                if snapshot.exits_until_invalidate == 0 {
                    actions |= Actions::REQUEST_INVALIDATION;
                }
            }
        }
        Event::Invalidate { cause, already_invalidated } => {
            let allowed = if cause == InvalidationCause::TracePoint {
                // TracePoint resets every ISEQ, whatever its state.
                true
            } else if cause.is_forced() {
                !already_invalidated
            } else {
                // Do not invalidate the final version to prevent unbounded recompilation.
                !already_invalidated && snapshot.num_versions < knobs.max_versions
            };
            if allowed {
                actions |= Actions::INVALIDATE;
                if snapshot.num_versions > 0 {
                    snapshot.last_version = LastVersion::Invalidated;
                }
                match knobs.num_reprofiles {
                    // Profile the next num_reprofiles calls right away, then recompile.
                    Some(num_reprofiles) if cause != InvalidationCause::TracePoint && num_reprofiles > 0 => {
                        snapshot.calls = knobs.call_threshold.saturating_sub(num_reprofiles.into());
                        actions |= Actions::REARM_PROFILING;
                    }
                    // Count calls from 0 again, which re-runs the warmup and the
                    // profiling window before the next version is compiled.
                    _ => snapshot.calls = 0,
                }
            }
        }
        Event::OutOfMemory => {
            actions |= Actions::STOP_COMPILING;
        }
    }
    let to = State::of(snapshot, knobs);
    Transition { from, event, to, actions }
}

// ---------------------------------------------------------------------------
// Glue between the VM and the transition function. Each `on_*` function loads
// a Snapshot, runs on_event(), performs the actions that belong to the
// lifecycle (counters, profiling, invalidation marks), stores the counters back,
// and returns what the caller still has to do.
// ---------------------------------------------------------------------------

fn last_version_of(payload: &IseqPayload) -> LastVersion {
    match payload.versions.last().map(|version| &unsafe { version.as_ref() }.status) {
        None | Some(IseqStatus::NotCompiled) => LastVersion::None,
        Some(IseqStatus::Compiled(_)) => LastVersion::Compiled,
        Some(IseqStatus::CantCompile(_)) => LastVersion::CantCompile,
        Some(IseqStatus::Invalidated) => LastVersion::Invalidated,
    }
}

impl Snapshot {
    /// Load the snapshot of an ISEQ. `payload` is None if the ISEQ doesn't have one yet.
    fn load(iseq: IseqPtr, payload: Option<&IseqPayload>, jit_exception: bool) -> Snapshot {
        let calls = unsafe { rb_zjit_iseq_entry_calls(iseq, jit_exception) };
        match payload {
            Some(payload) => Snapshot {
                calls,
                num_versions: payload.versions.len(),
                last_version: last_version_of(payload),
                exits_until_invalidate: payload.num_exits_until_invalidate,
            },
            None => Snapshot {
                calls,
                num_versions: 0,
                last_version: LastVersion::None,
                exits_until_invalidate: initial_exit_budget(),
            },
        }
    }
}

/// Print a transition for --zjit-trace-lifecycle
fn trace(iseq: IseqPtr, transition: &Transition, jit_exception: bool) {
    if !get_option!(trace_lifecycle) {
        return;
    }
    if transition.from == transition.to && transition.actions.is_empty() {
        return;
    }
    let Transition { from, event, to, actions } = transition;
    let entry = if jit_exception { " (exception entry)" } else { "" };
    let actions = if actions.is_empty() { String::new() } else { format!(" [{actions:?}]") };
    eprintln!("ZJIT lifecycle: {}{entry}: {from} --{event}--> {to}{actions}", iseq_get_location(iseq, 0));
}

/// Run `event` on an ISEQ with a payload, writing the counters back
fn dispatch(iseq: IseqPtr, payload: &mut IseqPayload, event: Event) -> Transition {
    let knobs = Knobs::get();
    let mut snapshot = Snapshot::load(iseq, Some(payload), false);
    let transition = on_event(&mut snapshot, event, &knobs);
    payload.num_exits_until_invalidate = snapshot.exits_until_invalidate;
    if snapshot.calls != unsafe { rb_zjit_iseq_entry_calls(iseq, false) } {
        unsafe { rb_zjit_iseq_set_entry_calls(iseq, false, snapshot.calls) };
    }
    trace(iseq, &transition, false);
    transition
}

/// Called by vm.c when the interpreter call counter reaches `rb_zjit_profile_threshold`.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_start_profiling(iseq: IseqPtr, jit_exception: bool) {
    let actions = on_interp_threshold(iseq, jit_exception);
    if actions.contains(Actions::START_PROFILING) {
        unsafe { rb_zjit_profile_enable(iseq) };
    }
}

/// Replay the interpreter call that vm.c just counted to get its transition.
/// vm.c has already incremented the counter, so this doesn't store it back.
///
/// vm.c calls this when the counter it incremented equals `rb_zjit_profile_threshold`.
/// The counter isn't atomic, so with multiple Ractors running the ISEQ it may have moved
/// again by the time we get here. Replay from the value vm.c saw rather than re-reading it.
pub fn on_interp_threshold(iseq: IseqPtr, jit_exception: bool) -> Actions {
    let knobs = Knobs::get();
    let payload = unsafe { get_iseq_payload_ptr(iseq).as_ref() };
    let mut snapshot = Snapshot::load(iseq, payload, jit_exception);
    snapshot.calls = unsafe { rb_zjit_profile_threshold }.wrapping_sub(1);
    let transition = on_event(&mut snapshot, Event::InterpCall, &knobs);
    trace(iseq, &transition, jit_exception);
    transition.actions
}

/// A function stub was hit. Return true if the callee should be compiled now;
/// otherwise the caller exits to the interpreter to keep profiling.
pub fn on_stub_hit(iseq: IseqPtr, payload: &mut IseqPayload) -> bool {
    let transition = dispatch(iseq, payload, Event::StubHit);
    if transition.actions.contains(Actions::START_PROFILING) {
        unsafe { rb_zjit_profile_enable(iseq) };
    }
    transition.actions.contains(Actions::COMPILE)
}

/// Return true if a new version may be compiled, i.e. the version limit isn't reached.
pub fn on_compile_requested(iseq: IseqPtr, payload: &mut IseqPayload) -> bool {
    let knobs = Knobs::get();
    let mut snapshot = Snapshot::load(iseq, Some(payload), false);
    let transition = on_event(&mut snapshot, Event::CompileRequested, &knobs);
    let allowed = transition.actions.contains(Actions::COMPILE);
    if !allowed {
        trace(iseq, &Transition { to: State::VersionLimitReached, ..transition }, false);
    }
    allowed
}

/// HIR for a new version has been built. Re-arm interpreter profiling if needed.
pub fn on_hir_built(iseq: IseqPtr, payload: &mut IseqPayload) {
    let transition = dispatch(iseq, payload, Event::HirBuilt);
    if transition.actions.contains(Actions::REARM_PROFILING) {
        reset_profiles_remaining(iseq, insn_profile_budget());
    }
}

/// Called right before a newly compiled version (`ok` is whether it succeeded) is
/// pushed to `payload.versions`.
pub fn on_compile_finished(iseq: IseqPtr, payload: &mut IseqPayload, ok: bool) {
    dispatch(iseq, payload, if ok { Event::CompileSucceeded } else { Event::CompileFailed });
}

/// Compiled code took a recompile exit. Return true if the caller should take
/// the VM lock and invalidate the latest version with [`on_invalidate`].
/// This runs on every such exit, so it doesn't take the VM lock.
pub fn on_recompile_exit(iseq: IseqPtr, payload: &mut IseqPayload) -> bool {
    dispatch(iseq, payload, Event::RecompileExit).actions.contains(Actions::REQUEST_INVALIDATION)
}

/// Invalidate `version` of `iseq` if the lifecycle allows it for `cause`. If this
/// returns true, the version has been marked invalidated and the ISEQ's call
/// counters have been reset, and the caller must re-stub the version's incoming calls.
pub fn on_invalidate(iseq: IseqPtr, payload: &mut IseqPayload, version: &mut IseqVersionRef, cause: InvalidationCause) -> bool {
    let already_invalidated = unsafe { version.as_ref() }.is_invalidated();
    let event = Event::Invalidate { cause, already_invalidated };
    let knobs = Knobs::get();
    let mut snapshot = Snapshot::load(iseq, Some(payload), false);
    let transition = on_event(&mut snapshot, event, &knobs);
    if transition.actions.contains(Actions::INVALIDATE) {
        unsafe { version.as_mut() }.status = IseqStatus::Invalidated;
        // Reset jit_entry, jit_exception, and both call counters.
        unsafe { rb_iseq_reset_jit_func(iseq) };
        if snapshot.calls != 0 {
            unsafe { rb_zjit_iseq_set_entry_calls(iseq, false, snapshot.calls) };
        }
        if transition.actions.contains(Actions::REARM_PROFILING) {
            let num_reprofiles = knobs.num_reprofiles.unwrap_or(knobs.num_profiles);
            reset_profiles_remaining(iseq, num_reprofiles);
        }
        trace(iseq, &transition, false);
        true
    } else {
        false
    }
}

/// A TracePoint was enabled: invalidate the latest version (if any) of `iseq` and reset its counters.
pub fn on_tracepoint_enabled(iseq: IseqPtr, payload: &mut IseqPayload) {
    let event = Event::Invalidate { cause: InvalidationCause::TracePoint, already_invalidated: false };
    let knobs = Knobs::get();
    let mut snapshot = Snapshot::load(iseq, Some(payload), false);
    let transition = on_event(&mut snapshot, event, &knobs);
    debug_assert!(transition.actions.contains(Actions::INVALIDATE));
    if let Some(version) = payload.versions.last_mut() {
        unsafe { version.as_mut() }.status = IseqStatus::Invalidated;
        trace(iseq, &transition, false);
    }
    unsafe { rb_iseq_reset_jit_func(iseq) };
}

/// Compiling `iseq` ran out of executable memory. Stop compiling so that the
/// interpreter stops incrementing ISEQ call counters. It is set back to true in
/// update_dropped_bytes() if memory becomes available again.
pub fn on_out_of_memory(iseq: IseqPtr) {
    let knobs = Knobs::get();
    let payload = unsafe { get_iseq_payload_ptr(iseq).as_ref() };
    let mut snapshot = Snapshot::load(iseq, payload, false);
    let transition = on_event(&mut snapshot, Event::OutOfMemory, &knobs);
    if transition.actions.contains(Actions::STOP_COMPILING) {
        unsafe { rb_zjit_compiling_p = false; }
    }
    trace(iseq, &transition, false);
}

/// `iseq` is being inlined into a caller. Its frames may be materialized by side
/// exits, so re-arm its interpreter profiling to collect fresh profiles there too.
pub fn on_inlined(iseq: IseqPtr) {
    reset_profiles_remaining(iseq, insn_profile_budget());
}

/// Return true if the version being compiled for `payload` is the final one,
/// i.e. it will never be invalidated for recompilation.
pub fn compiling_final_version(payload: &IseqPayload) -> bool {
    payload.versions.len() + 1 >= max_iseq_versions()
}

/// Return true if the version being compiled for `payload` should avoid speculative
/// optimizations that may side-exit: a previous version was invalidated and the
/// new one is the final version.
pub fn avoid_side_exits(payload: &IseqPayload) -> bool {
    payload.versions.iter().any(|v| unsafe { v.as_ref() }.is_invalidated())
        && compiling_final_version(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knobs() -> Knobs {
        Knobs { call_threshold: 30, num_profiles: 5, num_reprofiles: None, num_exits_until_invalidate: 5, max_versions: 4 }
    }

    fn fresh() -> Snapshot {
        Snapshot { calls: 0, num_versions: 0, last_version: LastVersion::None, exits_until_invalidate: 5 }
    }

    #[test]
    fn profile_threshold() {
        assert_eq!(knobs().profile_threshold(), 25);
        assert_eq!(Knobs { call_threshold: 1, ..knobs() }.profile_threshold(), 0);
        assert_eq!(Knobs { call_threshold: 2, ..knobs() }.profile_threshold(), 1);
        assert_eq!(Knobs { call_threshold: 30, num_profiles: 0, ..knobs() }.profile_threshold(), 30);
        assert_eq!(Knobs { call_threshold: 0, ..knobs() }.profile_threshold(), 1);
    }

    #[test]
    fn interp_calls_profile_then_compile() {
        let k = knobs();
        let mut s = fresh();
        for _ in 0..24 {
            let t = on_event(&mut s, Event::InterpCall, &k);
            assert_eq!((t.from, t.to, t.actions), (State::Interpreted, State::Interpreted, Actions::NOTHING));
        }
        let t = on_event(&mut s, Event::InterpCall, &k);
        assert_eq!((t.from, t.to, t.actions), (State::Interpreted, State::Profiling, Actions::START_PROFILING));
        for _ in 26..30 {
            assert_eq!(on_event(&mut s, Event::InterpCall, &k).actions, Actions::NOTHING);
        }
        let t = on_event(&mut s, Event::InterpCall, &k);
        assert_eq!((t.from, t.to, t.actions), (State::Profiling, State::Profiling, Actions::COMPILE));
        assert_eq!(s.calls, 30);
        // Past the threshold (e.g. the compile was skipped), the interpreter doesn't retry.
        assert_eq!(on_event(&mut s, Event::InterpCall, &k).actions, Actions::NOTHING);
    }

    #[test]
    fn interp_call_threshold_one_compiles_without_profiling() {
        let k = Knobs { call_threshold: 1, ..knobs() };
        let mut s = fresh();
        assert_eq!(on_event(&mut s, Event::InterpCall, &k).actions, Actions::COMPILE);
    }

    #[test]
    fn interp_call_zero_profiles_profiles_and_compiles_at_once() {
        let k = Knobs { num_profiles: 0, ..knobs() };
        let mut s = Snapshot { calls: 29, ..fresh() };
        assert_eq!(on_event(&mut s, Event::InterpCall, &k).actions, Actions::START_PROFILING | Actions::COMPILE);
    }

    #[test]
    fn stub_hit_skips_warmup() {
        let k = knobs();
        let mut s = Snapshot { calls: 3, ..fresh() };
        let t = on_event(&mut s, Event::StubHit, &k);
        assert_eq!((t.from, t.to, t.actions), (State::Interpreted, State::Profiling, Actions::START_PROFILING));
        assert_eq!(s.calls, 25);
        for calls in 26..30 {
            assert_eq!(on_event(&mut s, Event::StubHit, &k).actions, Actions::NOTHING);
            assert_eq!(s.calls, calls);
        }
        assert_eq!(on_event(&mut s, Event::StubHit, &k).actions, Actions::COMPILE);
        // Keep asking to compile on further hits, e.g. after a transient failure.
        assert_eq!(on_event(&mut s, Event::StubHit, &k).actions, Actions::COMPILE);
    }

    #[test]
    fn stub_hit_small_thresholds() {
        let k = Knobs { call_threshold: 2, ..knobs() };
        let mut s = fresh();
        assert_eq!(on_event(&mut s, Event::StubHit, &k).actions, Actions::START_PROFILING);
        assert_eq!(on_event(&mut s, Event::StubHit, &k).actions, Actions::COMPILE);

        let k = Knobs { call_threshold: 1, ..knobs() };
        let mut s = fresh();
        assert_eq!(on_event(&mut s, Event::StubHit, &k).actions, Actions::COMPILE);

        let k = Knobs { num_profiles: 0, ..knobs() };
        let mut s = fresh();
        assert_eq!(on_event(&mut s, Event::StubHit, &k).actions, Actions::START_PROFILING | Actions::COMPILE);
    }

    #[test]
    fn compile_outcomes() {
        let k = knobs();
        let mut s = Snapshot { calls: 30, exits_until_invalidate: 1, ..fresh() };
        assert_eq!(on_event(&mut s, Event::CompileRequested, &k).actions, Actions::COMPILE);
        assert_eq!(on_event(&mut s, Event::HirBuilt, &k).actions, Actions::REARM_PROFILING);
        let t = on_event(&mut s, Event::CompileSucceeded, &k);
        assert_eq!((t.from, t.to), (State::Profiling, State::Compiled(1)));
        assert_eq!(s.exits_until_invalidate, 5);

        let mut s = Snapshot { calls: 30, ..fresh() };
        let t = on_event(&mut s, Event::CompileFailed, &k);
        assert_eq!((t.from, t.to), (State::Profiling, State::CantCompile));
    }

    #[test]
    fn final_version_does_not_rearm_profiling() {
        let k = knobs();
        let mut s = Snapshot { num_versions: 3, last_version: LastVersion::Invalidated, ..fresh() };
        assert_eq!(on_event(&mut s, Event::HirBuilt, &k).actions, Actions::NOTHING);
        assert_eq!(on_event(&mut s, Event::CompileRequested, &k).actions, Actions::COMPILE);
        let mut s = Snapshot { num_versions: 4, last_version: LastVersion::Invalidated, ..fresh() };
        let t = on_event(&mut s, Event::CompileRequested, &k);
        assert_eq!((t.from, t.actions), (State::VersionLimitReached, Actions::NOTHING));
    }

    #[test]
    fn recompile_exits_until_invalidate() {
        let k = knobs();
        let mut s = Snapshot { calls: 30, num_versions: 1, last_version: LastVersion::Compiled, exits_until_invalidate: 5, ..fresh() };
        for remaining in (1..5).rev() {
            assert_eq!(on_event(&mut s, Event::RecompileExit, &k).actions, Actions::NOTHING);
            assert_eq!(s.exits_until_invalidate, remaining);
        }
        let t = on_event(&mut s, Event::RecompileExit, &k);
        assert_eq!((t.from, t.to, t.actions), (State::Compiled(1), State::Compiled(1), Actions::REQUEST_INVALIDATION));

        let event = Event::Invalidate { cause: InvalidationCause::RecompileExit, already_invalidated: false };
        let t = on_event(&mut s, event, &k);
        assert_eq!((t.from, t.to, t.actions), (State::Compiled(1), State::Invalidated(1), Actions::INVALIDATE));
        assert_eq!(s.calls, 0);

        // Exits from frames still running the invalidated code are ignored
        assert_eq!(on_event(&mut s, Event::RecompileExit, &k).actions, Actions::NOTHING);
        assert_eq!(s.exits_until_invalidate, 0);
    }

    #[test]
    fn recompile_exit_on_final_version_is_ignored() {
        let k = knobs();
        let mut s = Snapshot { num_versions: 4, last_version: LastVersion::Compiled, exits_until_invalidate: 1, ..fresh() };
        assert_eq!(on_event(&mut s, Event::RecompileExit, &k).actions, Actions::NOTHING);
        assert_eq!(s.exits_until_invalidate, 1);
    }

    #[test]
    fn reprofile_after_invalidation() {
        let k = knobs();
        let mut s = Snapshot { num_versions: 1, last_version: LastVersion::Invalidated, ..fresh() };
        assert_eq!(State::of(&s, &k), State::Invalidated(1));
        for _ in 0..24 {
            on_event(&mut s, Event::InterpCall, &k);
        }
        let t = on_event(&mut s, Event::InterpCall, &k);
        assert_eq!((t.from, t.to, t.actions), (State::Invalidated(1), State::Reprofiling(1), Actions::START_PROFILING));
        for _ in 26..30 {
            on_event(&mut s, Event::InterpCall, &k);
        }
        assert_eq!(on_event(&mut s, Event::InterpCall, &k).actions, Actions::COMPILE);
        let t = on_event(&mut s, Event::CompileSucceeded, &k);
        assert_eq!((t.from, t.to), (State::Reprofiling(1), State::Compiled(2)));
    }

    #[test]
    fn num_reprofiles_shortens_reprofiling() {
        let k = Knobs { num_reprofiles: Some(3), ..knobs() };
        let mut s = Snapshot { calls: 30, num_versions: 1, last_version: LastVersion::Compiled, ..fresh() };
        let event = Event::Invalidate { cause: InvalidationCause::RecompileExit, already_invalidated: false };
        let t = on_event(&mut s, event, &k);
        assert_eq!((t.from, t.to, t.actions), (State::Compiled(1), State::Reprofiling(1), Actions::INVALIDATE | Actions::REARM_PROFILING));
        assert_eq!(s.calls, 27);
        assert_eq!(on_event(&mut s, Event::InterpCall, &k).actions, Actions::NOTHING);
        assert_eq!(on_event(&mut s, Event::InterpCall, &k).actions, Actions::NOTHING);
        assert_eq!(on_event(&mut s, Event::InterpCall, &k).actions, Actions::COMPILE);

        // A longer window than num_profiles still starts profiling right away and
        // re-installs profiling instructions at profile_threshold.
        let k = Knobs { num_reprofiles: Some(10), ..knobs() };
        let mut s = Snapshot { calls: 30, num_versions: 1, last_version: LastVersion::Compiled, ..fresh() };
        on_event(&mut s, event, &k);
        assert_eq!(s.calls, 20);
        for _ in 21..25 {
            assert_eq!(on_event(&mut s, Event::InterpCall, &k).actions, Actions::NOTHING);
        }
        assert_eq!(on_event(&mut s, Event::InterpCall, &k).actions, Actions::START_PROFILING);

        // TracePoint always counts from 0
        let event = Event::Invalidate { cause: InvalidationCause::TracePoint, already_invalidated: false };
        let mut s = Snapshot { calls: 30, num_versions: 1, last_version: LastVersion::Compiled, ..fresh() };
        assert_eq!(on_event(&mut s, event, &k).actions, Actions::INVALIDATE);
        assert_eq!(s.calls, 0);
    }

    #[test]
    fn patch_point_invalidation_respects_version_limit() {
        let k = knobs();
        let event = Event::Invalidate { cause: InvalidationCause::PatchPoint, already_invalidated: false };
        let mut s = Snapshot { calls: 30, num_versions: 3, last_version: LastVersion::Compiled, ..fresh() };
        assert_eq!(on_event(&mut s, event, &k).actions, Actions::INVALIDATE);

        let mut s = Snapshot { calls: 30, num_versions: 4, last_version: LastVersion::Compiled, ..fresh() };
        let t = on_event(&mut s, event, &k);
        assert_eq!((t.from, t.to, t.actions), (State::Compiled(4), State::Compiled(4), Actions::NOTHING));
        assert_eq!(s.calls, 30);

        let event = Event::Invalidate { cause: InvalidationCause::PatchPoint, already_invalidated: true };
        let mut s = Snapshot { calls: 30, num_versions: 1, last_version: LastVersion::Compiled, ..fresh() };
        assert_eq!(on_event(&mut s, event, &k).actions, Actions::NOTHING);
    }

    #[test]
    fn forced_invalidation_ignores_version_limit() {
        let k = knobs();
        let event = Event::Invalidate { cause: InvalidationCause::EpEscape, already_invalidated: false };
        let mut s = Snapshot { calls: 30, num_versions: 4, last_version: LastVersion::Compiled, ..fresh() };
        let t = on_event(&mut s, event, &k);
        assert_eq!((t.from, t.to, t.actions), (State::Compiled(4), State::VersionLimitReached, Actions::INVALIDATE));

        let event = Event::Invalidate { cause: InvalidationCause::EpEscape, already_invalidated: true };
        assert_eq!(on_event(&mut s, event, &k).actions, Actions::NOTHING);

        // TracePoint always resets, even ISEQs that were never compiled or failed to compile
        let event = Event::Invalidate { cause: InvalidationCause::TracePoint, already_invalidated: false };
        let mut s = Snapshot { calls: 27, ..fresh() };
        let t = on_event(&mut s, event, &k);
        assert_eq!((t.from, t.to, t.actions), (State::Profiling, State::Interpreted, Actions::INVALIDATE));
        let mut s = Snapshot { calls: 30, num_versions: 1, last_version: LastVersion::CantCompile, ..fresh() };
        let t = on_event(&mut s, event, &k);
        assert_eq!((t.from, t.to), (State::CantCompile, State::Invalidated(1)));
    }

    #[test]
    fn out_of_memory_stops_compiling() {
        let mut s = fresh();
        assert_eq!(on_event(&mut s, Event::OutOfMemory, &knobs()).actions, Actions::STOP_COMPILING);
    }

    #[test]
    fn insn_profile_budget_counts_down() {
        let mut remaining: NumProfiles = 2;
        assert!(!on_insn_profiled(&mut remaining));
        assert!(on_insn_profiled(&mut remaining));
        assert!(on_insn_profiled(&mut remaining));
        assert_eq!(remaining, 0);
    }

    #[test]
    fn event_display() {
        let event = Event::Invalidate { cause: InvalidationCause::EpEscape, already_invalidated: false };
        assert_eq!(event.to_string(), "Invalidate(EpEscape)");
        assert_eq!(Event::StubHit.to_string(), "StubHit");
        assert_eq!(State::Reprofiling(2).to_string(), "Reprofiling(v2)");
    }

    #[test]
    fn actions_debug() {
        assert_eq!(format!("{:?}", Actions::NOTHING), "Nothing");
        assert_eq!(format!("{:?}", Actions::START_PROFILING | Actions::COMPILE), "StartProfiling|Compile");
    }
}
