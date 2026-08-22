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

/// Per-thread recycling cache for the small, short-lived blocks the JIT
/// compilers churn through.
///
/// Compiling one ISEQ is a burst of millions of small `Vec`/`Box` allocations
/// that are all released again when the compile finishes: HIR snapshots, LIR
/// instruction lists, live-interval range lists, parallel-move edge vectors.
/// Measured with `--zjit-alloc-stats`, a single compile-heavy workload asks for
/// ~18M blocks, and glibc's `malloc`/`free` (bin search, chunk splitting,
/// coalescing, `malloc_consolidate`) was ~13% of all instructions retired.
///
/// A block that is freed and then immediately requested again at the same size
/// does not need any of that work, so freed blocks are parked on a per-thread,
/// per-size-class free list and handed straight back out. The effect is a
/// compile-scoped arena without any lifetime plumbing through the compiler: the
/// arena is the cache, and it refills itself from whatever the previous compile
/// released.
///
/// Invariants that keep this sound:
///
/// * Only layouts with `align <= GRANULE` and `size <= MAX_RECYCLE_SIZE`
///   participate. Everything else goes straight to the system allocator with
///   its original layout, exactly as before.
/// * A participating layout is *always* turned into its canonical layout
///   (`size` rounded up to a multiple of `GRANULE`, `align == GRANULE`) before
///   it reaches the system allocator, on both the allocate and the free side.
///   So every cached block is exactly its class's size and alignment and can
///   serve any request in that class, and the layout `System` sees on free is
///   the one it saw on allocate.
/// * The cache is thread-local and holds no locks. A block allocated on one
///   thread and freed on another simply lands in the freeing thread's cache;
///   the two JIT compile threads (the Ruby thread and the background compile
///   thread) never share a free list.
/// * Retention is capped, so an unusual allocation pattern cannot turn the
///   cache into a leak.
mod recycle {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::ptr;

    /// Size and alignment granularity of the cache. 16 bytes is `malloc`'s
    /// natural alignment on the platforms that matter, so canonicalizing to it
    /// costs no extra memory, and it leaves room in every block for the free
    /// list link.
    const GRANULE: usize = 16;

    /// Largest request the cache serves. The compiler's churn is overwhelmingly
    /// small blocks; above this the per-block allocator cost is small relative
    /// to the work done with the block, and capping the range keeps worst-case
    /// retention low.
    const MAX_RECYCLE_SIZE: usize = 1024;

    const NUM_CLASSES: usize = MAX_RECYCLE_SIZE / GRANULE;

    /// Upper bound on bytes parked in one thread's cache. Sized to cover the
    /// working set of a single large compile; past it, frees go back to the
    /// system allocator so that a workload which allocates in a lopsided
    /// pattern cannot grow the cache without bound.
    const MAX_RETAINED: usize = 4 * 1024 * 1024;

    thread_local! {
        /// Head of each size class's free list. The link to the next free block
        /// is stored in the block's own first 8 bytes, which is always in
        /// bounds: the smallest class is `GRANULE` bytes.
        static FREE_LISTS: [Cell<*mut u8>; NUM_CLASSES] =
            const { [const { Cell::new(ptr::null_mut()) }; NUM_CLASSES] };

        /// Bytes currently parked across all of this thread's free lists.
        static RETAINED: Cell<usize> = const { Cell::new(0) };
    }

    /// The size class serving `layout`, or `None` if the cache does not handle it.
    #[inline(always)]
    fn class_of(layout: Layout) -> Option<usize> {
        let size = layout.size();
        if layout.align() <= GRANULE && size.wrapping_sub(1) < MAX_RECYCLE_SIZE {
            Some((size - 1) / GRANULE)
        } else {
            None
        }
    }

    /// The layout every block in `class` is allocated and freed with.
    #[inline(always)]
    fn canonical_layout(class: usize) -> Layout {
        // SAFETY: GRANULE is a non-zero power of two and the size is a multiple
        // of it, well below `isize::MAX`.
        unsafe { Layout::from_size_align_unchecked((class + 1) * GRANULE, GRANULE) }
    }

    /// Take a block out of `class`'s free list, or `None` if it is empty.
    #[inline(always)]
    fn pop(class: usize) -> Option<*mut u8> {
        FREE_LISTS.with(|lists| {
            let head = lists[class].get();
            if head.is_null() {
                return None;
            }
            // SAFETY: every block on the list is at least GRANULE bytes and had
            // its next pointer written into its first 8 bytes by `push`.
            let next = unsafe { ptr::read(head.cast::<*mut u8>()) };
            lists[class].set(next);
            RETAINED.with(|retained| {
                retained.set(retained.get() - canonical_layout(class).size())
            });
            Some(head)
        })
    }

    /// Park `ptr` on `class`'s free list. Returns false if the cache is full and
    /// the caller should free the block for real.
    #[inline(always)]
    fn push(class: usize, ptr: *mut u8) -> bool {
        let size = canonical_layout(class).size();
        let held = RETAINED.with(|retained| retained.get());
        if held + size > MAX_RETAINED {
            return false;
        }
        FREE_LISTS.with(|lists| {
            // SAFETY: `ptr` is a live block of at least GRANULE bytes that the
            // caller is giving up, so writing the link into it is fine.
            unsafe { std::ptr::write(ptr.cast::<*mut u8>(), lists[class].get()) };
            lists[class].set(ptr);
        });
        RETAINED.with(|retained| retained.set(held + size));
        true
    }

    /// Allocate `layout`, preferring a cached block. Returns the block and
    /// whether it came from the system allocator (and so needs accounting).
    #[inline(always)]
    pub fn alloc(layout: Layout) -> (*mut u8, bool) {
        match class_of(layout) {
            Some(class) => match pop(class) {
                Some(ptr) => (ptr, false),
                // SAFETY: canonical_layout has non-zero size.
                None => (unsafe { System.alloc(canonical_layout(class)) }, true),
            },
            // SAFETY: forwarding a layout the caller already promised is valid.
            None => (unsafe { System.alloc(layout) }, true),
        }
    }

    /// As [`alloc`], for `alloc_zeroed`.
    #[inline(always)]
    pub fn alloc_zeroed(layout: Layout) -> (*mut u8, bool) {
        match class_of(layout) {
            Some(class) => match pop(class) {
                Some(ptr) => {
                    // SAFETY: `ptr` owns the whole class-sized block.
                    unsafe { ptr::write_bytes(ptr, 0, canonical_layout(class).size()) };
                    (ptr, false)
                }
                // SAFETY: canonical_layout has non-zero size.
                None => (unsafe { System.alloc_zeroed(canonical_layout(class)) }, true),
            },
            // SAFETY: forwarding a layout the caller already promised is valid.
            None => (unsafe { System.alloc_zeroed(layout) }, true),
        }
    }

    /// Free `ptr`, parking it in the cache when possible. Returns whether the
    /// block went back to the system allocator (and so needs accounting).
    ///
    /// # Safety
    /// `ptr` must have come from [`alloc`]/[`alloc_zeroed`] with `layout`.
    #[inline(always)]
    pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) -> bool {
        match class_of(layout) {
            Some(class) => {
                if push(class, ptr) {
                    false
                } else {
                    unsafe { System.dealloc(ptr, canonical_layout(class)) };
                    true
                }
            }
            None => {
                unsafe { System.dealloc(ptr, layout) };
                true
            }
        }
    }

    /// Result of a `realloc` that the cache handled itself.
    pub enum Realloc {
        /// The block already sits in the class that serves `new_size`, so the
        /// same pointer is returned untouched. This is the common case for the
        /// first few pushes into a small `Vec`.
        InPlace(*mut u8),
        /// A new block was taken from the cache or the system allocator and the
        /// old one released. `system_alloc`/`system_free` say which side of that
        /// exchange touched the system allocator.
        Moved { ptr: *mut u8, system_alloc: bool, system_free: bool },
        /// Neither the old nor the new size is cacheable; the caller should fall
        /// through to `System.realloc`, which can often grow in place.
        Passthrough,
    }

    /// # Safety
    /// `ptr` must have come from [`alloc`]/[`alloc_zeroed`] with `layout`, and
    /// `new_size` must be a valid size for `layout.align()`.
    #[inline(always)]
    pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> Realloc {
        // SAFETY: the caller promises `new_size` is valid for this alignment.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let old_class = class_of(layout);
        let new_class = class_of(new_layout);
        if old_class.is_none() && new_class.is_none() {
            return Realloc::Passthrough;
        }
        if old_class == new_class {
            return Realloc::InPlace(ptr);
        }
        let (new_ptr, system_alloc) = alloc(new_layout);
        if new_ptr.is_null() {
            return Realloc::Moved { ptr: new_ptr, system_alloc: false, system_free: false };
        }
        // SAFETY: both blocks are live and at least `copied` bytes long, and
        // they do not overlap: `new_ptr` is a fresh block.
        let copied = layout.size().min(new_size);
        unsafe { ptr::copy_nonoverlapping(ptr, new_ptr, copied) };
        // SAFETY: `ptr` came from `alloc` with `layout`, as the caller promised.
        let system_free = unsafe { dealloc(ptr, layout) };
        Realloc::Moved { ptr: new_ptr, system_alloc, system_free }
    }

    /// Bytes the cache asks the system allocator for on behalf of `layout`,
    /// which is what the live-bytes accounting has to add and subtract.
    #[inline(always)]
    pub fn system_size(layout: Layout) -> usize {
        match class_of(layout) {
            Some(class) => canonical_layout(class).size(),
            None => layout.size(),
        }
    }
}

pub struct StatsAlloc {
    pub alloc_size: AtomicUsize,
}

unsafe impl GlobalAlloc for StatsAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        alloc_stats::record(layout.size());
        let (ptr, from_system) = recycle::alloc(layout);
        if from_system {
            self.alloc_size.fetch_add(recycle::system_size(layout), Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller promises `ptr`/`layout` came from this allocator.
        if unsafe { recycle::dealloc(ptr, layout) } {
            self.alloc_size.fetch_sub(recycle::system_size(layout), Ordering::SeqCst);
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        alloc_stats::record(layout.size());
        let (ptr, from_system) = recycle::alloc_zeroed(layout);
        if from_system {
            self.alloc_size.fetch_add(recycle::system_size(layout), Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        alloc_stats::record(new_size.saturating_sub(layout.size()));
        // SAFETY: the caller promises `ptr`/`layout` came from this allocator
        // and that `new_size` is valid for `layout.align()`.
        match unsafe { recycle::realloc(ptr, layout, new_size) } {
            recycle::Realloc::InPlace(ptr) => ptr,
            recycle::Realloc::Moved { ptr, system_alloc, system_free } => {
                // SAFETY: the caller promises `new_size` is valid for this alignment.
                let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
                if system_alloc {
                    self.alloc_size.fetch_add(recycle::system_size(new_layout), Ordering::SeqCst);
                }
                if system_free {
                    self.alloc_size.fetch_sub(recycle::system_size(layout), Ordering::SeqCst);
                }
                ptr
            }
            recycle::Realloc::Passthrough => {
                if new_size > layout.size() {
                    self.alloc_size.fetch_add(new_size - layout.size(), Ordering::SeqCst);
                } else if new_size < layout.size() {
                    self.alloc_size.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
                }
                // SAFETY: forwarding what the caller promised is valid.
                unsafe { System.realloc(ptr, layout, new_size) }
            }
        }
    }
}
