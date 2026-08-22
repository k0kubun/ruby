//! Optional background compilation: move ISEQ compilation off the thread that
//! tripped the call threshold and onto a dedicated Ruby thread, so application
//! threads never pay compile latency.
//!
//! # What runs where
//!
//! The compile thread is an ordinary Ruby thread created with `rb_thread_create`.
//! That is the whole safety story, and it is worth spelling out why it is enough.
//!
//! ZJIT's compiler state (`ZJITState`, the `CodeBlock`, `Invariants`, every
//! `IseqPayload`) is reached through `&mut` references derived from `static mut`s.
//! Two threads touching that concurrently would be UB. Today's synchronous path
//! is already reachable from *any* Ruby thread: whichever thread happens to make
//! the threshold-crossing call is the one that compiles. What keeps that sound is
//! not the VM lock -- `rb_vm_lock_enter` tracks ownership per *ractor*, so a
//! second thread in the same ractor re-enters it recursively rather than blocking
//! -- but the GVL: only one thread of a ractor executes Ruby or CRuby internals
//! at a time. The VM lock plus barrier is what stops *other ractors* from running
//! JIT code while we flip page permissions.
//!
//! The compile thread inherits exactly those guarantees. It holds the GVL for the
//! whole of each compilation, and it takes the VM lock and barrier around it via
//! the same `with_vm_lock` the synchronous path uses. Nothing in a compilation
//! releases the GVL: `with_vm_lock` disables GC, compilation allocates only on
//! the Rust heap, and it never reaches a `RUBY_VM_CHECK_INTS` point. So a
//! compilation on this thread is indivisible with respect to every other Ruby
//! thread, which is what makes the *install staleness* question go away:
//!
//! * HIR build, patch-point registration, code emission and the `jit_entry`
//!   store all happen inside one GVL-atomic critical section.
//! * Every invalidation hook (`rb_zjit_cme_invalidate`,
//!   `rb_zjit_method_lookup_changed`, `rb_zjit_bop_redefined`, ...) runs on a
//!   Ruby thread holding the GVL.
//!
//! There is therefore no window in which an invalidation can land between a
//! compile's read of the VM and the registration of the patch point that guards
//! it. An epoch/serial check at install time would have nothing to detect. What
//! *can* change between the moment an ISEQ is enqueued and the moment it is
//! compiled is caught by re-reading everything at compile time (the compile reads
//! fresh profiles, CMEs and class hierarchies) plus the `jit_entry` re-check in
//! [`compile_one`], which counts a discard when another path got there first.
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
//! of fresh methods does legitimately. That is also the entire `fork` story --
//! no atfork hook, because a `fork` can only be issued by a thread holding the
//! GVL, so it can never interrupt a compilation.
//!
//! # Where the win is, and where it is not
//!
//! The compile thread needs the GVL, so compilation overlaps application work
//! only while the application releases it. A request server does that constantly
//! (on every database call), and there the effect is the whole point: on a
//! 400-handler request loop with a GVL-releasing gap between requests, the worst
//! single request drops from ~19ms to ~0.2ms -- the compile spike disappears
//! rather than moving -- with steady-state request time at parity.
//!
//! A CPU-bound single-threaded benchmark releases the GVL only when the timer
//! thread preempts it, so there the same compilation is merely moved into slices
//! taken from the mutator, and the ISEQ stays interpreted until it lands. On
//! `30k_methods` (30,000 ISEQs to compile, pinned to one core) that costs about
//! 3%. Making the win unconditional would need the next step: splitting
//! compilation so the phases that touch no VM state (LIR, register allocation,
//! emission) run without the GVL.
//!
//! One caveat when A/B-testing this at a low `--zjit-call-threshold`: deferring
//! a compile also lengthens that ISEQ's profiling window, so the background run
//! may compile from more samples than the synchronous run did, and produce
//! different code. Pass `--zjit-num-profiles=1` to take that out of the
//! comparison.

use std::collections::VecDeque;
use std::ffi::c_void;

use crate::cruby::{
    EcPtr, IseqPtr, Qnil, VALUE, get_ec_cfp, iseq_self_is_heap_object, rb_ec_stack_check,
    rb_gc_location, rb_gc_mark_movable, rb_protect, rb_vm_frame_method_entry, src_loc,
    with_vm_lock,
};
use crate::options::{debug, get_option};
use crate::payload::get_or_create_iseq_payload;
use crate::state::{ZJITState, zjit_enabled_p};
use crate::stats::incr_counter;

/// Most ISEQs that may be waiting to compile at once. Overflow drops the request
/// and re-arms the ISEQ's threshold, so the work is not lost, only deferred; we
/// never block a request thread to make room.
const MAX_QUEUE_LEN: usize = 1024;

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

    fn rb_thread_create(func: extern "C" fn(*mut c_void) -> VALUE, arg: *mut c_void) -> VALUE;
    fn rb_thread_wakeup_alive(thread: VALUE) -> VALUE;
    fn rb_thread_sleep_deadly();
    fn rb_thread_schedule();
}

/// Queue of ISEQs waiting to be compiled, plus the state of the thread that
/// drains it. Only ever touched while holding the GVL, which is what makes a
/// plain `static mut` sound here -- see the module docs.
struct BgCompiler {
    /// ISEQs waiting to compile, oldest first.
    queue: VecDeque<IseqPtr>,

    /// The ISEQ the compile thread has taken off the queue but not finished with.
    /// Held here rather than only in a local so that GC marking keeps it alive
    /// for the window between the pop and the compile.
    current: Option<IseqPtr>,

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
}

impl BgCompiler {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            current: None,
            thread: Qnil,
            started: false,
            parked: false,
            shutdown: false,
            wake_deferred: false,
            disabled: false,
            completed: 0,
        }
    }
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
        if bg.queue.is_empty() && bg.current.is_none() {
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
    let depth = (bg.queue.len() + bg.current.is_some() as usize) as u64;
    let counters = ZJITState::get_counters();
    if depth > counters.bg_compile_queue_high_water {
        counters.bg_compile_queue_high_water = depth;
    }

    check_liveness(bg)
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
            // Drain. `current` is set before the compile and cleared after, so a
            // GC in between still marks the ISEQ.
            loop {
                let Some(iseq) = ({
                    let state = bg();
                    let popped = state.queue.pop_front();
                    state.current = popped;
                    popped
                }) else {
                    break;
                };
                compile_one(iseq);
                let state = bg();
                state.current = None;
                state.completed += 1;
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

/// Compile one queued ISEQ and install it, mirroring
/// `rb_zjit_iseq_gen_entry_point` for the `jit_exception: false` case.
fn compile_one(iseq: IseqPtr) {
    get_or_create_iseq_payload(iseq).bg_queued = false;

    // Another path may have compiled this ISEQ while it sat in the queue: a
    // function stub hit, or a nested `RubyVM::ZJIT` call. Nothing is wrong with
    // the queued request, there is just nothing left to do.
    if unsafe { rb_zjit_iseq_has_jit_entry(iseq) } {
        incr_counter!(bg_compile_discard_count);
        return;
    }

    let ec = unsafe { rb_zjit_current_ec() };
    if unsafe { rb_ec_stack_check(ec as _) } != 0 {
        incr_counter!(skipped_native_stack_full);
        return;
    }

    // Compile and install in one critical section, so that the whole of it --
    // HIR build, patch point registration, code emission, and the `jit_entry`
    // store -- is indivisible with respect to every other Ruby thread, exactly
    // as it is on the synchronous path. `jit_entry` is a naturally aligned
    // pointer, so an interpreter on another thread sees either NULL or the new
    // code, never a torn value.
    let installed = with_vm_lock(src_loc!(), || {
        // Re-check now that nothing else can run.
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
    for iseq in bg.queue.iter_mut().chain(bg.current.iter_mut()) {
        *iseq = unsafe { rb_gc_location(VALUE::from(*iseq)) }.as_iseq();
    }
    if !bg.thread.nil_p() {
        bg.thread = unsafe { rb_gc_location(bg.thread) };
    }
}
