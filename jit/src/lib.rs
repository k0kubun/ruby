//! Shared code between YJIT and ZJIT.
#![warn(unsafe_op_in_unsafe_fn)] // Adopt 2024 edition default when targeting 2021 editions

use std::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

#[global_allocator]
pub static GLOBAL_ALLOCATOR: StatsAlloc = StatsAlloc { alloc_size: AtomicUsize::new(0) };

/// Per-phase allocation accounting for `--zjit-alloc-stats`. Compiled out
/// unless the `alloc_stats` feature is on: the counters live in the global
/// allocator's hot path, and leaving them in costs ~0.3% of compile
/// instructions on allocation-heavy workloads even with the option off.
#[cfg(feature = "alloc_stats")]
pub mod alloc_stats {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Number of buckets allocations can be attributed to. Indexed by the
    /// `Counter` discriminant of the enclosing compile phase, so this has to
    /// stay above ZJIT's counter count; [`set_phase`] falls back to
    /// [`PHASE_NONE`] rather than panicking if that stops being true.
    pub const PHASES: usize = 1024;

    /// Bucket for everything allocated outside a phase that named itself.
    pub const PHASE_NONE: usize = 0;

    /// Whether the allocator should attribute allocations at all. Off until
    /// `--zjit-alloc-stats` turns it on.
    pub static TRACKING: AtomicBool = AtomicBool::new(false);

    /// Bucket allocations are attributed to right now. Phases set this on entry
    /// and restore the enclosing value on exit, so an allocation lands in the
    /// innermost phase that claimed it.
    pub static PHASE: AtomicUsize = AtomicUsize::new(PHASE_NONE);

    /// Number of allocations attributed to each bucket.
    pub static COUNT: [AtomicUsize; PHASES] = [const { AtomicUsize::new(0) }; PHASES];

    /// Bytes requested by the allocations attributed to each bucket. This counts
    /// every request, including memory later freed, unlike `alloc_size` which
    /// tracks the live total.
    pub static BYTES: [AtomicUsize; PHASES] = [const { AtomicUsize::new(0) }; PHASES];

    /// Turn attribution on. Never undone: the counters are read once at exit.
    pub fn enable() {
        TRACKING.store(true, Ordering::Relaxed);
    }

    /// Point subsequent allocations at `phase`, returning the bucket it displaced.
    #[inline]
    pub fn set_phase(phase: usize) -> usize {
        if !TRACKING.load(Ordering::Relaxed) {
            return PHASE_NONE;
        }
        let phase = if phase < PHASES { phase } else { PHASE_NONE };
        PHASE.swap(phase, Ordering::Relaxed)
    }

    /// Restore the bucket [`set_phase`] displaced.
    #[inline]
    pub fn restore_phase(phase: usize) {
        if TRACKING.load(Ordering::Relaxed) {
            PHASE.store(phase, Ordering::Relaxed);
        }
    }

    /// Attribute one allocation of `size` bytes to the current phase.
    #[inline(always)]
    pub fn record(size: usize) {
        if TRACKING.load(Ordering::Relaxed) {
            let phase = PHASE.load(Ordering::Relaxed);
            COUNT[phase].fetch_add(1, Ordering::Relaxed);
            BYTES[phase].fetch_add(size, Ordering::Relaxed);
        }
    }
}

/// No-op stand-in so the allocator body does not need its own `cfg`.
#[cfg(not(feature = "alloc_stats"))]
mod alloc_stats {
    #[inline(always)]
    pub fn record(_size: usize) {}
}

pub struct StatsAlloc {
    pub alloc_size: AtomicUsize,
}

unsafe impl GlobalAlloc for StatsAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.alloc_size.fetch_add(layout.size(), Ordering::SeqCst);
        alloc_stats::record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.alloc_size.fetch_sub(layout.size(), Ordering::SeqCst);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.alloc_size.fetch_add(layout.size(), Ordering::SeqCst);
        alloc_stats::record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            self.alloc_size.fetch_add(new_size - layout.size(), Ordering::SeqCst);
        } else if new_size < layout.size() {
            self.alloc_size.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
        }
        alloc_stats::record(new_size.saturating_sub(layout.size()));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
