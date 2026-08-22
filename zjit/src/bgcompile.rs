//! Optional background compilation: move ISEQ compilation off the thread that
//! tripped the call threshold and onto a dedicated Ruby thread, so application
//! threads never pay compile latency -- and, for the phases that permit it, off
//! the GVL entirely, so the compile genuinely runs in parallel with the
//! application rather than in the slices the scheduler takes away from it.
//!
//! # What runs where
//!
//! The compile thread is an ordinary Ruby thread created with `rb_thread_create`.
//!
//! ZJIT's compiler state (`ZJITState`, the `CodeBlock`, `Invariants`, every
//! `IseqPayload`) is reached through `&mut` references derived from `static mut`s.
//! Two threads touching that concurrently would be UB. The synchronous path is
//! already reachable from *any* Ruby thread: whichever thread happens to make the
//! threshold-crossing call is the one that compiles. What keeps that sound is not
//! the VM lock -- `rb_vm_lock_enter` tracks ownership per *ractor*, so a second
//! thread in the same ractor re-enters it recursively rather than blocking -- but
//! the GVL: only one thread of a ractor executes Ruby or CRuby internals at a
//! time. The VM lock plus barrier is what stops *other ractors* from running JIT
//! code while we flip page permissions.
//!
//! A compilation is therefore split into three phases, of which only the middle
//! one may run without the GVL:
//!
//! | phase | what | GVL |
//! |---|---|---|
//! | 1 | HIR construction and optimization, HIR to LIR lowering | held |
//! | 2 | LIR splitting, register allocation, side-exit compilation, scratch-register split | released |
//! | 3 | machine code emission, patch point registration, JIT-to-JIT stubs, `jit_entry` store | held |
//!
//! Phase 1 is everything that reads the VM, and there is a lot of it: bytecode,
//! profiles, method lookups for inlining and specialization, class hierarchies,
//! shapes, constant caches, frozen-string checks. None of it can move. Phase 3 is
//! everything that writes state the interpreter shares: the code region, the
//! invariant tables, the ISEQ payloads, the side-exit metadata table. Phase 2 is
//! what is left, and what is left is 42% of compile time in an optimized build --
//! register allocation alone is a third of it, side-exit compilation another fifth.
//!
//! Phase 2's precondition is stated as a type: [`crate::backend::lir::Assembler::prepare`]
//! takes and returns an `Assembler` and touches nothing else. Getting side-exit
//! compilation in there took moving its two shared-state writes out: interning
//! side-exit metadata and reopening profile windows now happen in
//! `Assembler::adopt_exit_metas` at the start of emission. Debug builds also
//! enforce the precondition at runtime -- see [`assert_gvl_held`], which the
//! getters for every piece of shared ZJIT state call.
//!
//! # Why a batch
//!
//! The phases run over a *batch* of queued ISEQs rather than one at a time, and
//! that is not an optimization but the difference between the split working and
//! being much worse than not splitting at all.
//!
//! Giving up the GVL is easy; getting it back is not. A thread waiting for the GVL
//! gets it when the holder yields, and a CPU-bound application thread yields once
//! per 10ms timeslice. One ISEQ per release means one 10ms wait per ISEQ, which
//! caps compilation at a few dozen ISEQs a second -- so the ISEQs stay interpreted,
//! and every benchmark gets slower. Phase 3 of one batch and phase 1 of the next
//! run in the same GVL acquisition, so a batch costs exactly one wait however many
//! ISEQs it holds.
//!
//! The batch is bounded by *time*, not by count: phase 1 stops accepting ISEQs
//! after `--zjit-background-compile-batch-ms` (default 10ms, one timeslice) and
//! puts the rest back at the front of the queue. Bounding by time is what keeps the
//! knob meaningful across ISEQs that differ in size by orders of magnitude, and one
//! timeslice is the natural default: it makes the compile thread no greedier with
//! the GVL than an application thread is.
//!
//! Raising it trades latency for throughput and lowering it does the reverse, and
//! both directions are measurable. The floor is not "as small as possible": too
//! small and compilation never finishes, which is the pathology above.
//!
//! # Staleness
//!
//! Releasing the GVL between reading the VM and installing the code opens a window
//! the old design did not have: the application thread runs arbitrary Ruby, and
//! anything phase 1 assumed can stop being true before phase 3 gets to arm the
//! patch point that was supposed to guard it.
//!
//! The set of assumptions is closed and already enumerated -- it is exactly what
//! ZJIT tracks patch points for -- so phase 1 records it and the invalidation
//! hooks report against it. See [`crate::bg_assume`] for why that is both sound and
//! precise enough to be useful. Phase 3 checks one boolean: a poisoned compilation
//! is thrown away and counted in `bg_compile_stale_discard_count`, and the ISEQ's
//! threshold is re-armed so a later call offers it again.
//!
//! Not backed by a patch point, and so handled separately:
//!
//! * **`body->jit_entry` already set.** Another path may have compiled the ISEQ
//!   while we were preparing. Re-checked under the lock, as before.
//! * **GC compaction.** It moves the ISEQs, classes and objects the snapshot holds
//!   raw pointers to. [`note_compaction`] poisons unconditionally; compaction is
//!   rare enough that keying it would buy nothing.
//! * **A new version.** Phase 3 re-runs the payload's version-limit and
//!   already-compiled checks before emitting.
//!
//! What phase 2 does *not* need protecting from is object death: [`mark`] marks the
//! ISEQ being compiled, and everything the snapshot reads is reachable from it --
//! its literal pool, the ISEQs its bytecode names, and its ZJIT payload, whose
//! profile marking covers every class the compiler specialized on. Objects reached
//! by another route (a CME, an inlined callee's ISEQ, a class from a constant) have
//! no such root, so they are recorded as assumptions and their free hooks poison.
//!
//! `--zjit-background-compile-hold-gvl` puts the whole pipeline back under one
//! critical section, for A/B measurement. Debug and tracing options do the same
//! implicitly: see [`nogvl_usable`].
//!
//! # What is not backgrounded
//!
//! * Exception-handler entries (`body->jit_exception`, and
//!   `rb_zjit_exception_entry_miss`). These compile *for the live control frame*:
//!   the continuation PC and VM stack depth come from `cfp`, and the frame is
//!   gone by the time a background thread would run. They stay synchronous.
//! * The stub patch in `function_stub_hit`. A stub hit enqueues the callee for an
//!   ordinary entry-point compile and side-exits to the interpreter for this
//!   call (reusing the `materialize_exit_trampoline` shape the profile-stub
//!   machinery already needed). The next stub hit finds `IseqStatus::Compiled`,
//!   so `gen_iseq` returns the existing code and only the one-instruction stub
//!   rewrite happens on the request thread.
//! * Compiles requested from a non-main ractor. The compile thread lives in the
//!   main ractor, which outlives every other ractor; queueing work from a ractor
//!   that may terminate first would strand it.
//!
//! # Liveness
//!
//! The thread starts lazily on the first enqueue and parks in
//! `rb_thread_sleep_deadly()` when the queue drains. Parking deadlockably (rather
//! than in a native blocking region) keeps CRuby's deadlock detection working:
//! the thread is counted as a sleeper, so a program whose only other thread
//! sleeps forever still reports "No live threads left" exactly as it did before.
//!
//! Because the enqueuer and the thread both hold the GVL, and neither releases it
//! between "queue looks empty" and "park", there is no lost-wakeup race and the
//! queue itself needs no mutex.
//!
//! A lost compile thread (`Thread#kill`, or `fork`, which leaves the child with
//! none) is detected on the next enqueue by reading the `Thread`'s status, and a
//! fresh one is started, which drains whatever was left queued. The check is
//! exact rather than inferred from queue depth: a full queue means only that an
//! enqueuing thread outran the compile thread, which a tight loop over thousands
//! of fresh methods does legitimately.
//!
//! Phase 2 changes the `fork` story. It used to be that no atfork hook was needed
//! because a `fork` can only be issued by a thread holding the GVL and a compile
//! held the GVL throughout, so a fork could never land inside one. Now it can, and
//! the child inherits a batch of half-prepared compilations with no thread to
//! finish them. [`forget_lost_batch`] drops them, on the same
//! next-enqueue liveness check, and re-arms their ISEQs.
//!
//! # Where the win is, and where it is not
//!
//! Moving the compile off the requesting thread is a latency win wherever the
//! application yields the GVL, which a request server does constantly: on a
//! 400-handler request loop with a GVL-releasing gap between requests, the worst
//! single request drops from ~19ms to ~0.2ms -- the compile spike disappears rather
//! than moving -- with steady-state request time at parity.
//!
//! Phase 2 is what makes it a *throughput* win as well, and only for the part of
//! the pipeline that fits there. More than half of a compilation still needs the
//! GVL, so an application thread that never yields still waits for that half. The
//! honest summary is that pinning a compile-bound run to two cores recovers roughly
//! the phase-2 share of compile time and no more.
//!
//! What that is worth depends on how much faster the compiled code is than the
//! interpreter, which decides whether the wall clock is dominated by *interference*
//! (what phase 2 removes) or by *compile latency* (which deferring makes worse, and
//! which the phase split makes worse again). On a request-shaped loop whose handlers
//! spend their time in C functions -- the realistic case, where ZJIT is a few tens
//! of percent rather than an order of magnitude -- 43% of compile time runs off the
//! GVL, which halves the interference (565ms to 292ms over the run) and takes 12%
//! off the wall clock of the whole thing. On a loop whose
//! handlers are pure integer arithmetic, where compiled code is 25x the
//! interpreter's speed, the same change *loses*: every millisecond an ISEQ waits to
//! be installed costs 25 milliseconds of interpretation, and that swamps anything
//! interference does. Background compilation already had that property; the split
//! sharpens it.
//!
//! Emission is the next candidate and is not done: it writes into the shared code
//! region, so it would need its destination reserved before the GVL is dropped, and
//! the size of a function is not known until it is emitted. It is worth about
//! another 10% of compile time.
//!
//! One caveat when A/B-testing this at a low `--zjit-call-threshold`: deferring
//! a compile also lengthens that ISEQ's profiling window, so the background run
//! may compile from more samples than the synchronous run did, and produce
//! different code. Pass `--zjit-num-profiles=1` to take that out of the
//! comparison.

use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::c_void;

use crate::bg_assume::{Assumption, Assumptions};
use crate::codegen::{PendingCompile, Snapshot, finish_entry_point, snapshot_entry_point};
use crate::cruby::{
    EcPtr, IseqPtr, Qnil, VALUE, get_ec_cfp, iseq_self_is_heap_object, rb_ec_stack_check,
    rb_gc_location, rb_gc_mark_movable, rb_protect, rb_vm_frame_method_entry, src_loc,
    with_vm_lock,
};
use crate::options::{debug, get_option, get_option_ref};
use crate::payload::get_or_create_iseq_payload;
use crate::state::{ZJITState, zjit_enabled_p};
use crate::stats::{Counter, CounterSink, incr_counter, incr_counter_by};

/// Most ISEQs that may be waiting to compile at once. Overflow drops the request
/// and re-arms the ISEQ's threshold, so the work is not lost, only deferred; we
/// never block a request thread to make room.
const MAX_QUEUE_LEN: usize = 1024;

/// Most ISEQs one batch takes off the queue. Not usually the binding limit -- that
/// is `--zjit-background-compile-batch-ms`, which stops phase 1 well short of this
/// for any realistic ISEQ size -- but it does bound how many prepared compilations
/// can be alive at once, and each one holds a whole `Assembler`. See
/// [`compile_batch`].
const MAX_BATCH_LEN: usize = MAX_QUEUE_LEN;

unsafe extern "C" {
    /// `GET_EC()`. Defined in zjit.c.
    fn rb_zjit_current_ec() -> EcPtr;
    /// Whether the calling thread belongs to the main ractor. Defined in zjit.c.
    fn rb_zjit_main_ractor_p() -> bool;
    /// `ISEQ_BODY(iseq)->jit_entry != NULL`. Defined in zjit.c.
    fn rb_zjit_iseq_has_jit_entry(iseq: IseqPtr) -> bool;
    /// Store `body->jit_entry`. Defined in zjit.c.
    fn rb_zjit_iseq_set_jit_entry(iseq: IseqPtr, code_ptr: *const u8);
    /// Put `body->jit_entry_calls` one short of the call threshold so the next
    /// call trips it again. Defined in zjit.c.
    fn rb_zjit_iseq_rearm_threshold(iseq: IseqPtr);
    /// Whether a Thread has not been killed. A plain field read: unlike
    /// `rb_thread_wakeup_alive` it touches no scheduler state, so it is safe to
    /// call under the VM lock. Defined in zjit.c.
    fn rb_zjit_thread_alive_p(thread: VALUE) -> bool;
    /// Move a thread out of `ThreadGroup::Default`. Defined in zjit.c.
    fn rb_zjit_thread_group_isolate(thread: VALUE);

    /// Run `func(data)` with the GVL released and interrupts deferred. Defined in
    /// zjit.c.
    fn rb_zjit_compile_without_gvl(func: extern "C" fn(*mut c_void), data: *mut c_void);

    fn rb_thread_create(func: extern "C" fn(*mut c_void) -> VALUE, arg: *mut c_void) -> VALUE;
    fn rb_thread_wakeup_alive(thread: VALUE) -> VALUE;
    fn rb_thread_sleep_deadly();
    fn rb_thread_schedule();
}

// Whether this thread is inside phase 2 of a background compilation. Debug builds
// use it to catch a phase-2 pass reaching for state only a GVL holder may touch.
thread_local! {
    static IN_NOGVL_PHASE: Cell<bool> = const { Cell::new(false) };
}

/// Panic if `what` was reached from the GVL-free phase of a background
/// compilation. Called from the accessors for every piece of shared ZJIT state, so
/// it has to cost nothing in release builds -- and it does: with
/// `debug_assertions` off the whole body folds away.
///
/// This is the enforcement half of phase 2's contract. The reasoning half is the
/// audit in the module docs; this is what stops the next pass added to
/// [`Assembler::prepare`] from quietly breaking it.
#[inline(always)]
pub fn assert_gvl_held(what: &'static str) {
    if cfg!(debug_assertions) && in_nogvl_phase() {
        panic!("ZJIT: {what} reached from the GVL-free background compile phase");
    }
}

/// Whether this thread is inside phase 2 of a background compilation. For the few
/// places that have to *behave* differently there rather than just assert, which
/// means they are checked in release builds too.
#[inline(always)]
pub fn in_nogvl_phase() -> bool {
    IN_NOGVL_PHASE.with(|flag| flag.get())
}

/// Whether phase 2 may run without the GVL. Options that make the compiler read
/// the VM or write shared state outside phase 1 and 3 turn it off:
///
/// * `dump_disasm`, `dump_lir` and (on dev builds) `debug` enable `asm_comment!`,
///   whose formatting calls into the VM for ISEQ and method names.
/// * `trace_side_exits` makes side-exit compilation render a `SideExitReason`,
///   which does the same.
/// * `trace_compiles` has every pass write into the shared Perfetto tracer.
/// * `perf` is only read during emission, but the symbol names it builds come from
///   the VM, so keep it in the same bucket rather than reasoning about it.
fn nogvl_usable() -> bool {
    !get_option!(background_compile_hold_gvl)
        && get_option_ref!(dump_disasm).is_none()
        && get_option!(dump_lir).is_none()
        && get_option!(trace_side_exits).is_none()
        && !get_option!(trace_compiles)
        && get_option!(perf).is_none()
        && !(cfg!(debug_assertions) && get_option!(debug))
}

/// Queue of ISEQs waiting to be compiled, plus the state of the thread that
/// drains it. Only ever touched while holding the GVL, which is what makes a
/// plain `static mut` sound here -- see the module docs.
struct BgCompiler {
    /// ISEQs waiting to compile, oldest first.
    queue: VecDeque<IseqPtr>,

    /// The ISEQs the compile thread has taken off the queue but not finished with.
    /// Held here rather than only in a local so that GC marking keeps them alive
    /// for the window between the pop and the install.
    current: Vec<IseqPtr>,

    /// The compile thread, or nil before the first enqueue.
    thread: VALUE,

    /// Whether `thread` is a thread we started and have not seen die.
    started: bool,

    /// Whether the compile thread is parked in `rb_thread_sleep_deadly`.
    parked: bool,

    /// Set when the thread should return instead of parking again. Nothing sets
    /// it today: CRuby's ordinary shutdown (`rb_thread_terminate_all`) kills the
    /// compile thread while it is parked, which unwinds it cleanly.
    shutdown: bool,

    /// An enqueue that happened under the VM lock wants the compile thread
    /// woken. See [`PostLock`] and [`flush_deferred_wake`].
    wake_deferred: bool,

    /// Give up on background compilation for the rest of the process.
    disabled: bool,

    /// Compiles the thread has finished.
    completed: u64,

    /// What the compilation currently in phase 2 assumed about the VM, while the
    /// GVL is not held. Written and read only by GVL holders -- the compile thread
    /// installs it before releasing the GVL and takes it back after reacquiring,
    /// and the invalidation hooks that poison it all run on GVL-holding threads --
    /// so the compile thread itself never touches it during phase 2.
    /// One entry per compilation currently in phase 2, in the order they will be
    /// installed. Written and read only by GVL holders -- the compile thread fills
    /// it before releasing the GVL and drains it after reacquiring, and the
    /// invalidation hooks that poison entries all run on GVL-holding threads -- so
    /// the compile thread itself never touches it during phase 2.
    inflight: Vec<Inflight>,
}

impl BgCompiler {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            current: Vec::new(),
            thread: Qnil,
            started: false,
            parked: false,
            shutdown: false,
            wake_deferred: false,
            disabled: false,
            completed: 0,
            inflight: Vec::new(),
        }
    }
}

/// A compilation in phase 2, as the GVL-holding side sees it.
struct Inflight {
    /// The ISEQ being compiled. Duplicated from the `PendingCompile` on the compile
    /// thread's stack because *this* copy is reachable from
    /// [`update_references`] and so survives compaction, while the one inside the
    /// paused compilation cannot be found or fixed. Phase 3 has to touch the ISEQ
    /// (to re-check `jit_entry`, to re-arm the threshold) before it knows whether
    /// the compilation is stale, and doing that through a pointer a compaction
    /// moved is a segfault.
    iseq: IseqPtr,

    /// What the compilation assumed, and whether it still holds.
    assumptions: Assumptions,
}

static mut BG_COMPILER: Option<BgCompiler> = None;

/// Get the background compiler state, creating it on first use.
///
/// SAFETY: every caller holds the GVL, so there is only ever one `&mut` alive.
fn bg() -> &'static mut BgCompiler {
    unsafe {
        if BG_COMPILER.is_none() {
            BG_COMPILER = Some(BgCompiler::new());
        }
        BG_COMPILER.as_mut().unwrap()
    }
}

/// Whether the state has been created. Used by GC callbacks, which must not
/// allocate it just to walk it.
fn bg_opt() -> Option<&'static mut BgCompiler> {
    unsafe { BG_COMPILER.as_mut() }
}

/// Tell the compilation in phase 2, if any, that `assumption` no longer holds.
///
/// Called from every ZJIT invalidation hook, which is what makes the check
/// complete: an assumption that broke without one of those hooks firing would
/// already be a bug in the *installed* code's guard, not in this window. Cheap
/// enough for the hottest of them (`rb_zjit_method_lookup_changed` runs on every
/// method definition) because the common case is that no compile is in flight.
///
/// Must be called with the GVL held, which every invalidation hook is.
pub fn note_invalidation(assumption: Assumption) {
    let Some(bg) = bg_opt() else { return };
    for inflight in bg.inflight.iter_mut() {
        inflight.assumptions.note(assumption);
    }
}

/// Discard every compilation in phase 2, whatever it assumed.
///
/// For the invalidations that are global rather than keyed -- TracePoint being
/// enabled, a second ractor starting, a non-root box appearing, a whole-method-cache
/// flush. Each one is rare, and each one is broad enough that deciding it does not
/// apply to a paused compilation would mean trusting that the compilation carries
/// the matching patch point, which is not something the hook can check. They cost a
/// discard instead.
pub fn note_invalidation_all() {
    note_compaction();
}

/// Discard any compilation in phase 2, because GC compaction has moved objects it
/// holds raw pointers to. Called from `rb_zjit_root_update_references`.
///
/// Unconditional: the snapshot points at the ISEQ, its callees, classes, CMEs and
/// baked-in objects, and compaction can move all of them. Nothing keyed would be
/// cheaper -- checking whether *this* compaction moved *these* objects means
/// reading them, which is the thing that is no longer safe.
pub fn note_compaction() {
    let Some(bg) = bg_opt() else { return };
    for inflight in bg.inflight.iter_mut() {
        inflight.assumptions.poison();
    }
}

/// Whether `--zjit-background-compile` is on and background compilation is
/// usable right now.
pub fn enabled() -> bool {
    if !get_option!(background_compile) {
        return false;
    }
    // `RubyVM::ZJIT.assert_compiles` asserts that compilation has happened by the
    // time the block returns, which deferring it would break. Also, a compile
    // failure under it panics, and a panic on the compile thread aborts the
    // process instead of failing a test.
    if ZJITState::assert_compiles_enabled() {
        return false;
    }
    !bg_opt().is_some_and(|bg| bg.disabled)
}

/// Hand an ISEQ to the compile thread instead of compiling it here. Returns
/// false if the caller should compile synchronously after all.
///
/// Called from `rb_zjit_compile_iseq()` for ordinary entry points only.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_bg_enqueue(iseq: IseqPtr, ec: EcPtr) -> bool {
    if !zjit_enabled_p() || !enabled() {
        return false;
    }
    // The compile thread lives in the main ractor. Work queued from a ractor that
    // may terminate before the compile runs would be stranded, and a thread
    // created from such a ractor would die with it.
    if !unsafe { rb_zjit_main_ractor_p() } {
        return false;
    }

    let cfp = unsafe { get_ec_cfp(ec) };
    let post = with_vm_lock(src_loc!(), || enqueue_locked(iseq, Some(cfp)));
    if post == PostLock::Declined {
        return false;
    }
    run_post_lock(post);

    if get_option!(background_compile_block) {
        block_until_drained();
    }
    true
}

/// Yield the GVL until the compile thread has drained the queue. Only used by
/// `--zjit-background-compile-block`; see that option's docs. Deliberately keyed
/// on the queue being empty rather than on a particular ISEQ: `rb_thread_schedule`
/// can let a compacting GC run, which would move an `IseqPtr` held in a local.
fn block_until_drained() {
    unsafe extern "C" {
        fn rb_thread_schedule();
    }
    // Bounded so that a wedged compile thread degrades to "these ISEQs stay
    // interpreted" rather than hanging the process.
    for _ in 0..100_000 {
        let bg = bg();
        if bg.disabled || !bg.started {
            return;
        }
        if bg.queue.is_empty() && bg.current.is_empty() {
            return;
        }
        // Releases the GVL, so the compile thread can make progress. May raise a
        // pending interrupt, which would otherwise be raised a few instructions
        // later inside the ISEQ we are about to enter.
        unsafe { rb_thread_schedule() };
    }
}

/// What [`enqueue_locked`] needs its caller to do once the VM lock is released.
///
/// Nothing that touches CRuby's thread scheduler may run inside `with_vm_lock`.
/// When the lock was taken with a barrier -- which it is whenever a second
/// ractor exists -- `rb_ractor_sched_barrier_start` deliberately keeps
/// `vm->ractor.sched.lock` held for the whole critical section ("do not release
/// ractor_sched_lock"), and both waking a thread (`rb_thread_wakeup_alive` ->
/// `ubf_waiting` -> `thread_sched_setup_running_threads`) and creating one
/// re-enter that same non-recursive native mutex. Doing either under the lock
/// self-deadlocks. Both are safe immediately after: the GVL is not released
/// between the push and the action, so no wakeup can be lost.
#[derive(PartialEq, Eq)]
enum PostLock {
    /// Declined: the caller should compile synchronously.
    Declined,
    /// Enqueued, and the compile thread does not need poking.
    Nothing,
    /// Enqueued; wake the parked compile thread.
    Wake,
    /// Enqueued; there is no compile thread yet, so start one.
    Start,
}

/// Push `iseq` onto the queue. Must be called with the VM lock held, because it
/// touches the ISEQ payload. Never allocates a Ruby object and never reaches a
/// check-interrupts point, so it is safe to call with GC disabled.
///
/// `cfp` is the frame the threshold was tripped in, used to decide whether
/// `self` is always a heap object. `None` when the caller has already recorded
/// that (the function stub path does it from the callee frame).
fn enqueue_locked(iseq: IseqPtr, cfp: Option<crate::cruby::CfpPtr>) -> PostLock {
    let bg = bg();
    if bg.disabled {
        return PostLock::Declined;
    }

    let payload = get_or_create_iseq_payload(iseq);
    if payload.bg_queued {
        // Already waiting. Nothing to do, and definitely don't compile here.
        return PostLock::Nothing;
    }

    if bg.queue.len() >= MAX_QUEUE_LEN {
        incr_counter!(bg_compile_overflow_count);
        // Don't block and don't lose the ISEQ: put its counter back one short of
        // the threshold so a later call re-offers it.
        unsafe { rb_zjit_iseq_rearm_threshold(iseq) };
        // A full queue is not evidence of anything wrong: an enqueuing thread
        // that never yields the GVL (a tight loop over thousands of fresh
        // methods) legitimately outruns the compile thread. Liveness is checked
        // below instead, exactly, rather than inferred from the depth.
        return check_liveness(bg);
    }

    // Record whether `self` is always a heap object while we still have the frame
    // that tripped the threshold; the compile thread's own frames say nothing
    // about this ISEQ. Deliberately the same computation as
    // `codegen::update_self_is_heap_object`, which the synchronous path does at
    // compile time.
    if let Some(cfp) = cfp {
        let cme = unsafe { rb_vm_frame_method_entry(cfp) };
        payload.self_is_heap_object =
            !cme.is_null() && iseq_self_is_heap_object(iseq, unsafe { (*cme).owner });
    }

    payload.bg_queued = true;
    bg.queue.push_back(iseq);
    incr_counter!(bg_compile_enqueue_count);
    let depth = (bg.queue.len() + bg.current.len()) as u64;
    let counters = ZJITState::get_counters();
    if depth > counters.bg_compile_queue_high_water {
        counters.bg_compile_queue_high_water = depth;
    }

    check_liveness(bg)
}

/// Drop what a vanished compile thread left mid-compilation.
///
/// Before the phase split this could not happen: a compilation held the GVL
/// throughout, and both `fork` and `Thread#kill` need the GVL, so neither could
/// land inside one. Phase 2 gives up the GVL, so now a `fork` can be issued in the
/// middle of a batch -- and the child inherits the batch with no thread to finish
/// it, and the assumption records that batch published with nothing to answer them.
///
/// The batch's ISEQs are re-armed rather than re-queued: their compilations are
/// half-built Rust objects reachable only from the dead thread's stack, so the work
/// has to start over anyway.
///
/// Must be called with the VM lock held.
fn forget_lost_batch(bg: &mut BgCompiler) {
    bg.inflight.clear();
    for &iseq in bg.current.iter() {
        get_or_create_iseq_payload(iseq).bg_queued = false;
        unsafe { rb_zjit_iseq_rearm_threshold(iseq) };
    }
    bg.current.clear();
}

/// Decide what the caller has to do to make sure the compile thread will pick up
/// what is queued: start one, wake a parked one, or nothing.
///
/// The liveness check is what keeps a lost compile thread from silently stopping
/// ZJIT from ever compiling again. `fork` and `Thread#kill` both leave a dead
/// `Thread` behind, and if it died while draining rather than parked, no wakeup
/// would ever be attempted and nothing else would notice.
fn check_liveness(bg: &mut BgCompiler) -> PostLock {
    if bg.started && !unsafe { rb_zjit_thread_alive_p(bg.thread) } {
        bg.started = false;
        bg.parked = false;
        bg.thread = Qnil;
        forget_lost_batch(bg);
        incr_counter!(bg_compile_thread_restart_count);
    }
    if !bg.started {
        PostLock::Start
    } else if bg.parked {
        PostLock::Wake
    } else {
        PostLock::Nothing
    }
}

/// Carry out what [`enqueue_locked`] deferred. Must run with the GVL held and
/// the VM lock released; see [`PostLock`].
fn run_post_lock(post: PostLock) {
    match post {
        PostLock::Declined | PostLock::Nothing => {}
        PostLock::Start => start_thread(),
        PostLock::Wake => {
            let bg = bg();
            if bg.started && unsafe { rb_thread_wakeup_alive(bg.thread) }.nil_p() {
                // Killed, or we are in a forked child. Start a fresh one, which
                // will drain everything still queued.
                bg.started = false;
                bg.parked = false;
                bg.thread = Qnil;
                forget_lost_batch(bg);
                incr_counter!(bg_compile_thread_restart_count);
                start_thread();
            }
        }
    }
}

/// Give up on background compilation. Everything still queued goes back to
/// tripping its threshold and compiling synchronously.
fn disable_locked(bg: &mut BgCompiler) {
    bg.disabled = true;
    for &iseq in bg.queue.iter() {
        get_or_create_iseq_payload(iseq).bg_queued = false;
        unsafe { rb_zjit_iseq_rearm_threshold(iseq) };
    }
    bg.queue.clear();
    incr_counter!(bg_compile_disabled_count);
}

/// Enqueue the callee of a JIT-to-JIT function stub. Called from
/// `function_stub_hit`, which is already inside `with_vm_lock`, so waking the
/// compile thread has to wait until that lock is released -- see [`PostLock`].
/// The caller must call [`flush_deferred_wake`] once it is out of the lock.
///
/// Declines when there is no compile thread yet, because starting one from
/// inside the lock is exactly what [`PostLock`] forbids; the entry-point path
/// starts it, and until then stub hits compile synchronously as before.
///
/// Returns true when the caller should side-exit to the interpreter for this call.
pub fn enqueue_from_stub(iseq: IseqPtr) -> bool {
    if !enabled() || !unsafe { rb_zjit_main_ractor_p() } {
        return false;
    }
    if !bg().started {
        return false;
    }
    // `function_stub_hit` has already set `payload.self_is_heap_object` from the
    // callee frame, which is a better source than anything we could pass here.
    match enqueue_locked(iseq, None) {
        // Starting a thread is what [`PostLock`] forbids under the lock, so let
        // this call compile synchronously; the entry-point path restarts it.
        PostLock::Declined | PostLock::Start => false,
        PostLock::Nothing => true,
        PostLock::Wake => {
            bg().wake_deferred = true;
            true
        }
    }
}

/// Wake the compile thread if a call made under the VM lock asked for it. Must
/// run with the GVL held and the VM lock released. Cheap and safe to call
/// unconditionally.
pub fn flush_deferred_wake() {
    let Some(bg) = bg_opt() else { return };
    if !bg.wake_deferred {
        return;
    }
    bg.wake_deferred = false;
    run_post_lock(PostLock::Wake);
}

/// Start the compile thread. Runs outside the VM lock.
fn start_thread() {
    debug_assert!(!bg().started);

    let mut state: std::os::raw::c_int = 0;
    // rb_thread_create can raise (ThreadError, NoMemoryError). We are called from
    // deep inside the interpreter's dispatch of an ordinary call, so swallow the
    // exception rather than letting it surface as if the Ruby program raised it.
    let thread = unsafe { rb_protect(Some(create_thread_body), Qnil, &mut state) };
    if state != 0 || thread.nil_p() {
        debug!("ZJIT: failed to start the background compile thread");
        with_vm_lock(src_loc!(), || disable_locked(bg()));
        return;
    }

    let bg = bg();
    bg.thread = thread;
    bg.started = true;
    bg.parked = false;

    // Cosmetic, and separately protected: a failure here must not lose the
    // thread we just recorded.
    let mut ignored: std::os::raw::c_int = 0;
    unsafe { rb_protect(Some(isolate_thread_group_body), thread, &mut ignored) };
}

unsafe extern "C" fn create_thread_body(_arg: VALUE) -> VALUE {
    unsafe { rb_thread_create(bg_thread_main, std::ptr::null_mut()) }
}

/// The compile thread is ZJIT's, not the application's. It cannot be hidden from
/// `Thread.list`, but it does not belong in the default ThreadGroup.
unsafe extern "C" fn isolate_thread_group_body(thread: VALUE) -> VALUE {
    unsafe { rb_zjit_thread_group_isolate(thread) };
    Qnil
}

/// Body of the compile thread. Drains the queue, parks, repeat.
extern "C" fn bg_thread_main(_arg: *mut c_void) -> VALUE {
    // A Rust panic must not unwind into CRuby. Compilation itself is wrapped by
    // with_vm_lock, which aborts on panic; this catches anything else.
    let result = std::panic::catch_unwind(|| {
        loop {
            // Drain in batches. `current` holds the batch for the whole of its
            // three phases, so a GC in between still marks every ISEQ in it.
            loop {
                {
                    let state = bg();
                    let take = state.queue.len().min(MAX_BATCH_LEN);
                    if take == 0 {
                        break;
                    }
                    state.current.extend(state.queue.drain(..take));
                }
                // A copy, not a move: `current` has to stay populated for the
                // whole batch so that a GC during any of its phases marks every
                // ISEQ in it.
                let batch = bg().current.clone();
                let completed = compile_batch(&batch);
                let state = bg();
                state.current.clear();
                state.completed += completed;
            }

            {
                let state = bg();
                if state.shutdown || state.disabled {
                    state.started = false;
                    state.parked = false;
                    return;
                }

                // No GVL release between the empty-queue check above and parking,
                // so an enqueue either happened before it (and we loop) or lands
                // after we are parked (and wakes us).
                state.parked = true;
            }
            unsafe { rb_thread_sleep_deadly() };
            bg().parked = false;
        }
    });
    if result.is_err() {
        eprintln!("ZJIT: background compile thread panicked");
        std::process::abort();
    }
    Qnil
}

/// Compile a batch of queued ISEQs, and return how many were installed.
///
/// The batch exists to amortize GVL handoffs. Splitting a compilation means the
/// compile thread has to reacquire the GVL to finish, and it only gets it when the
/// application yields -- which a CPU-bound application does once per 10ms
/// timeslice. One ISEQ per handoff caps compilation at a few dozen ISEQs a second,
/// which is worse than not splitting at all: the ISEQs stay interpreted. Batching
/// puts many compilations' worth of work into each handoff, so the split costs
/// throughput nothing and buys the phase-2 share of the interference.
fn compile_batch(batch: &[IseqPtr]) -> u64 {
    if !nogvl_usable() {
        let ec = unsafe { rb_zjit_current_ec() };
        let mut installed = 0;
        for &iseq in batch {
            if prepare_to_compile(iseq) {
                installed += compile_one_holding_gvl(iseq, ec) as u64;
            }
        }
        return installed;
    }

    // Phase 1, under the VM lock: build HIR and lower it for as much of the batch
    // as fits in the time budget. The rest goes back on the queue -- it is not
    // dropped, just deferred to the next batch.
    let mut pendings: Vec<PendingCompile> = Vec::new();
    let mut deferred: usize = 0;
    let budget = std::time::Duration::from_millis(get_option!(background_compile_batch_ms));
    with_vm_lock(src_loc!(), std::panic::AssertUnwindSafe(|| {
        let start = std::time::Instant::now();
        for (idx, &iseq) in batch.iter().enumerate() {
            // Bound the stretch the application is blocked for by *time* rather
            // than by ISEQ count, since ISEQs differ in size by orders of
            // magnitude. One timeslice is the natural unit: hold the GVL for
            // longer than that and we are the latency spike we set out to remove.
            if !pendings.is_empty() && start.elapsed() >= budget {
                deferred = batch.len() - idx;
                break;
            }
            if !prepare_to_compile(iseq) {
                continue;
            }
            let cb = ZJITState::get_code_block();
            match crate::stats::with_time_stat(Counter::compile_time_ns, || snapshot_entry_point(cb, iseq, true)) {
                Ok(Snapshot::Pending(pending)) => pendings.push(pending),
                Ok(Snapshot::AlreadyCompiled(_)) => incr_counter!(bg_compile_discard_count),
                Err(err) => {
                    // Record the failure the way the synchronous path does,
                    // including the `--zjit-stats` failure-counter entry point.
                    let code_ptr = finish_entry_point(iseq, false, Err(err));
                    unsafe { rb_zjit_iseq_set_jit_entry(iseq, code_ptr) };
                }
            }
        }

        // Publish what phase 1 assumed, so an invalidation landing while the GVL
        // is released can poison the compilation it applies to. Still under the
        // GVL, and the GVL is not released between here and the release, so
        // nothing can be missed.
        let inflight = &mut bg().inflight;
        // Normally already empty: phase 3 drains it. It is not empty if a
        // previous batch's owner vanished between phases, which `fork` can now do
        // -- see [`forget_lost_batch`]. Those compilations no longer exist, so
        // their records are not ours to answer for.
        inflight.clear();
        for pending in pendings.iter_mut() {
            inflight.push(Inflight {
                iseq: pending.iseq(),
                assumptions: std::mem::take(&mut pending.assumptions),
            });
        }

        if deferred > 0 {
            // Put the tail back at the *front* of the queue, so a long queue still
            // makes progress in order rather than starving its head.
            let bg = bg();
            for &iseq in batch[batch.len() - deferred..].iter().rev() {
                get_or_create_iseq_payload(iseq).bg_queued = true;
                bg.queue.push_front(iseq);
            }
        }
    }));

    if pendings.is_empty() {
        return 0;
    }

    // Phase 2, without the GVL: the whole point of the split.
    let nogvl_ns = run_prepare_without_gvl(&mut pendings);
    incr_counter_by(Counter::bg_compile_nogvl_time_ns, nogvl_ns);
    incr_counter_by(Counter::compile_time_ns, nogvl_ns);

    // Phase 3, under the VM lock. Emission, patch point registration and the
    // `jit_entry` store are one critical section per ISEQ, exactly as they are on
    // the synchronous path, so an invalidation cannot land between arming a patch
    // point and publishing the code that depends on it. `jit_entry` is a naturally
    // aligned pointer, so an interpreter on another thread sees either NULL or the
    // new code.
    with_vm_lock(src_loc!(), std::panic::AssertUnwindSafe(|| {
        let mut inflight = std::mem::take(&mut bg().inflight).into_iter();
        let mut installed = 0;
        for pending in pendings {
            let inflight = inflight.next().expect("one per pending, pushed in order");
            installed += install_one(pending, inflight) as u64;
        }
        installed
    }))
}

/// Clear an ISEQ's queued flag and decide whether there is still anything to do
/// for it. Must be called with the GVL held.
fn prepare_to_compile(iseq: IseqPtr) -> bool {
    get_or_create_iseq_payload(iseq).bg_queued = false;

    // Another path may have compiled this ISEQ while it sat in the queue: a
    // function stub hit, or a nested `RubyVM::ZJIT` call. Nothing is wrong with
    // the queued request, there is just nothing left to do.
    if unsafe { rb_zjit_iseq_has_jit_entry(iseq) } {
        incr_counter!(bg_compile_discard_count);
        return false;
    }

    let ec = unsafe { rb_zjit_current_ec() };
    if unsafe { rb_ec_stack_check(ec as _) } != 0 {
        incr_counter!(skipped_native_stack_full);
        return false;
    }
    true
}

/// Phase 3 for one compilation. Must be called with the VM lock held.
fn install_one(pending: PendingCompile, inflight: Inflight) -> bool {
    // Deliberately `inflight.iseq` and not `pending.iseq()`: a compaction during
    // phase 2 moves the ISEQ, and the copy inside the paused compilation is not
    // reachable from the GC hooks that would have fixed it. Reading a moved ISEQ is
    // a segfault, so the staleness check has to come before anything touches it and
    // has to use the pointer that was updated.
    let iseq = inflight.iseq;

    if inflight.assumptions.is_poisoned() {
        debug!("ZJIT: discarding a stale background compile: {}",
               crate::cruby::iseq_get_location(iseq, 0));
        incr_counter!(bg_compile_stale_discard_count);
        pending.discard();
        // The work is not lost, only deferred: put the counter back so a later
        // call offers the ISEQ again, and it recompiles against the VM as it now
        // is.
        unsafe { rb_zjit_iseq_rearm_threshold(iseq) };
        return false;
    }
    debug_assert_eq!(iseq, pending.iseq(), "an unpoisoned compilation saw no compaction");

    // Anything else got there first: nothing wrong, just nothing to install.
    if unsafe { rb_zjit_iseq_has_jit_entry(iseq) } {
        incr_counter!(bg_compile_discard_count);
        pending.discard();
        return false;
    }

    let code_ptr = crate::stats::with_time_stat(Counter::compile_time_ns, || {
        let cb = ZJITState::get_code_block();
        finish_entry_point(iseq, false, pending.install(cb).map(|ptrs| ptrs.start_ptr))
    });
    unsafe { rb_zjit_iseq_set_jit_entry(iseq, code_ptr) };
    incr_counter!(bg_compile_count);
    true
}

/// Compile without ever letting go of the GVL, as the compile thread did before the
/// phase split. Used by `--zjit-background-compile-hold-gvl` and by the debug
/// options that make phase 2 unsafe; see [`nogvl_usable`].
fn compile_one_holding_gvl(iseq: IseqPtr, ec: EcPtr) -> bool {
    let installed = with_vm_lock(src_loc!(), || {
        if unsafe { rb_zjit_iseq_has_jit_entry(iseq) } {
            incr_counter!(bg_compile_discard_count);
            return false;
        }
        let code_ptr = crate::codegen::gen_entry_point_locked(iseq, ec, false);
        unsafe { rb_zjit_iseq_set_jit_entry(iseq, code_ptr) };
        true
    });
    if installed {
        incr_counter!(bg_compile_count);
    }
    installed
}

/// Run phase 2 for a batch with the GVL released, and return the nanoseconds spent
/// inside the region.
///
/// Three things have to be arranged around the release. Counter updates are
/// redirected into a private list, because the global `Counters` is plain memory
/// that only a GVL holder may write; the caller folds it back in. `IN_NOGVL_PHASE`
/// is set so that a debug build catches a pass reaching for shared state -- see
/// [`assert_gvl_held`]. And the clock is read *inside* the region: the call does not
/// return until the GVL has been reacquired, and the wait for it is the
/// application's time, not the compiler's.
fn run_prepare_without_gvl(pendings: &mut Vec<PendingCompile>) -> u64 {
    /// Everything the phase needs, passed through the C shim as one pointer.
    struct PrepareArgs<'a> {
        pendings: &'a mut Vec<PendingCompile>,
        stall_ms: u64,
        elapsed_ns: u64,
    }

    extern "C" fn prepare_body(arg: *mut c_void) {
        // SAFETY: the pointer is a `&mut PrepareArgs` borrowed for the duration of
        // the call below, and nothing else aliases it.
        let args = unsafe { &mut *(arg as *mut PrepareArgs) };
        IN_NOGVL_PHASE.with(|flag| flag.set(true));
        let start = std::time::Instant::now();

        // `--zjit-background-compile-stall-ms`: hold the window open so a test can
        // land an invalidation in it. Deliberately here rather than under the GVL,
        // so the application thread is free to run during the stall.
        if args.stall_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(args.stall_ms));
        }

        for pending in args.pendings.iter_mut() {
            pending.prepare();
        }

        args.elapsed_ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        IN_NOGVL_PHASE.with(|flag| flag.set(false));
    }

    // Deliberately a local: nothing in `BgCompiler` may be borrowed across the
    // release, since a GVL holder can reach the whole struct while we are gone. An
    // empty Vec costs no allocation, and only `--zjit-stats` ever fills it.
    let mut sink: CounterSink = Vec::new();
    let previous = crate::stats::redirect_counters(&mut sink);

    let mut args = PrepareArgs { pendings, stall_ms: get_option!(background_compile_stall_ms), elapsed_ns: 0 };
    unsafe {
        rb_zjit_compile_without_gvl(prepare_body, &mut args as *mut PrepareArgs as *mut c_void);
    }
    let elapsed_ns = args.elapsed_ns;

    crate::stats::restore_counter_sink(previous);
    // Back under the GVL, so the global counters are writable again.
    crate::stats::flush_counter_sink(&mut sink);

    elapsed_ns
}


/// Mark the ISEQs and the thread the queue keeps alive. Called from
/// `rb_zjit_root_mark`.
pub fn mark() {
    let Some(bg) = bg_opt() else { return };
    for &iseq in bg.queue.iter().chain(bg.current.iter()) {
        unsafe { rb_gc_mark_movable(VALUE::from(iseq)) };
    }
    if !bg.thread.nil_p() {
        unsafe { rb_gc_mark_movable(bg.thread) };
    }
}

/// Mirror of [`mark`] for compaction. Called from
/// `rb_zjit_root_update_references`.
pub fn update_references() {
    let Some(bg) = bg_opt() else { return };
    // A compilation paused in phase 2 gets discarded rather than fixed up (see
    // [`note_compaction`]), but phase 3 still has to reach its ISEQ to re-arm the
    // threshold, so this one pointer is kept current.
    for inflight in bg.inflight.iter_mut() {
        inflight.iseq = unsafe { rb_gc_location(VALUE::from(inflight.iseq)) }.as_iseq();
    }
    for iseq in bg.queue.iter_mut().chain(bg.current.iter_mut()) {
        *iseq = unsafe { rb_gc_location(VALUE::from(*iseq)) }.as_iseq();
    }
    if !bg.thread.nil_p() {
        bg.thread = unsafe { rb_gc_location(bg.thread) };
    }
}
