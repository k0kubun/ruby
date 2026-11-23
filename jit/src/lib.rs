//! Shared code between YJIT and ZJIT.
#![warn(unsafe_op_in_unsafe_fn)] // Adopt 2024 edition default when targeting 2021 editions

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

#[global_allocator]
pub static GLOBAL_ALLOCATOR: StatsAlloc = StatsAlloc { alloc_size: AtomicUsize::new(0) };

pub struct StatsAlloc {
    /// Bytes allocated without [`SMALL_CACHE`]. Small blocks are counted in [`FreeLists::alloc_size`].
    alloc_size: AtomicUsize,
}

impl StatsAlloc {
    /// The number of bytes allocated and not freed yet
    pub fn alloc_size(&self) -> usize {
        let small_size = loop {
            if let Some(size) = SMALL_CACHE.try_with(|lists| lists.alloc_size) {
                break size;
            }
            std::hint::spin_loop();
        };
        // The sum is non-negative, but either part alone may not be: a block can be allocated
        // while the cache is locked and freed into it.
        self.alloc_size.load(Ordering::SeqCst).wrapping_add(small_size)
    }

    fn add_alloc_size(&self, size: usize) {
        self.alloc_size.fetch_add(size, Ordering::SeqCst);
    }

    fn sub_alloc_size(&self, size: usize) {
        self.alloc_size.fetch_sub(size, Ordering::SeqCst);
    }
}

/// Small blocks are spaced this far apart in size. It's also their alignment, which malloc()
/// guarantees on the 64-bit platforms YJIT and ZJIT support.
const SMALL_ALIGN: usize = 16;
/// The smallest block size. Size classes are the usable sizes of glibc's chunks (a multiple of
/// 16 bytes minus an 8-byte header) so that a block isn't bigger than malloc(size) would give.
const SMALL_MIN: usize = 24;
const NUM_CLASSES: usize = 64;
/// Blocks of up to this many bytes are recycled through [`SmallCache`]. It's glibc's largest
/// tcache size.
const SMALL_MAX: usize = SMALL_MIN + (NUM_CLASSES - 1) * SMALL_ALIGN;
/// Free bytes each size class may keep before blocks go back to the system allocator.
const CLASS_CACHE_BYTES: usize = 32 * 1024;

/// Free lists of small blocks, one per size class.
///
/// A compile allocates and frees millions of small Vecs and Boxes. glibc's tcache keeps only 7
/// free blocks per size, so most of them went through glibc's slower bins instead. Blocks in a
/// free list are linked through their first word. Every block in class `c` was allocated from
/// the system allocator with [`class_size`]`(c)` bytes, so it can serve any request of that class.
///
/// It's process-wide rather than per-thread: a global allocator can't use a thread-local with a
/// destructor on every platform, and without one, an exiting thread would leak its blocks.
/// A thread that finds the cache locked uses the system allocator instead of waiting.
struct SmallCache {
    locked: AtomicBool,
    lists: UnsafeCell<FreeLists>,
}

struct FreeLists {
    heads: [*mut u8; NUM_CLASSES],
    free_bytes: [usize; NUM_CLASSES],
    /// Bytes of small blocks allocated and not freed yet, counted here to save an atomic
    /// operation on [`StatsAlloc::alloc_size`]. Wraps around like that counter.
    alloc_size: usize,
}

// SAFETY: lists is only accessed while holding locked
unsafe impl Sync for SmallCache {}

static SMALL_CACHE: SmallCache = SmallCache::new();

impl SmallCache {
    const fn new() -> Self {
        SmallCache {
            locked: AtomicBool::new(false),
            lists: UnsafeCell::new(FreeLists { heads: [std::ptr::null_mut(); NUM_CLASSES], free_bytes: [0; NUM_CLASSES], alloc_size: 0 }),
        }
    }

    /// Run func with the free lists, or return None if another thread is using them
    fn try_with<R>(&self, func: impl FnOnce(&mut FreeLists) -> R) -> Option<R> {
        if self.locked.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return None;
        }
        // SAFETY: we hold the lock, and func doesn't reenter the allocator
        let ret = func(unsafe { &mut *self.lists.get() });
        self.locked.store(false, Ordering::Release);
        Some(ret)
    }

    /// Pop a block of a size class, or allocate one from the system allocator.
    /// `size` is the requested size, for [`StatsAlloc::alloc_size`].
    unsafe fn alloc(&self, class: usize, size: usize) -> *mut u8 {
        let cached = self.try_with(|lists| {
            lists.alloc_size = lists.alloc_size.wrapping_add(size);
            let block = lists.heads[class];
            if !block.is_null() {
                lists.heads[class] = unsafe { *(block as *mut *mut u8) };
                lists.free_bytes[class] -= class_size(class);
            }
            block
        });
        match cached {
            Some(block) if !block.is_null() => block,
            Some(_) => unsafe { System.alloc(class_layout(class)) },
            None => {
                GLOBAL_ALLOCATOR.add_alloc_size(size);
                unsafe { System.alloc(class_layout(class)) }
            }
        }
    }

    /// Push a block of a size class, or free it if the class has cached enough.
    /// `size` is the size it was requested with, for [`StatsAlloc::alloc_size`].
    unsafe fn dealloc(&self, ptr: *mut u8, class: usize, size: usize) {
        let cached = self.try_with(|lists| {
            lists.alloc_size = lists.alloc_size.wrapping_sub(size);
            if lists.free_bytes[class] + class_size(class) > CLASS_CACHE_BYTES {
                return false;
            }
            unsafe { *(ptr as *mut *mut u8) = lists.heads[class] };
            lists.heads[class] = ptr;
            lists.free_bytes[class] += class_size(class);
            true
        });
        if cached.is_none() {
            GLOBAL_ALLOCATOR.sub_alloc_size(size);
        }
        if cached != Some(true) {
            unsafe { System.dealloc(ptr, class_layout(class)) };
        }
    }
}

/// The size class of a layout if it's served by [`SmallCache`]
fn small_class(layout: Layout) -> Option<usize> {
    if layout.size() != 0 && layout.size() <= SMALL_MAX && layout.align() <= SMALL_ALIGN {
        // The smallest class whose class_size() fits: ceil((size - SMALL_MIN) / SMALL_ALIGN), or 0
        Some(layout.size().saturating_sub(SMALL_MIN - SMALL_ALIGN + 1) / SMALL_ALIGN)
    } else {
        None
    }
}

fn class_size(class: usize) -> usize {
    SMALL_MIN + class * SMALL_ALIGN
}

fn class_layout(class: usize) -> Layout {
    // SAFETY: the alignment is a power of two, and the size is at least the alignment,
    // so System.alloc() uses malloc(), whose blocks have that alignment.
    unsafe { Layout::from_size_align_unchecked(class_size(class), SMALL_ALIGN) }
}

unsafe impl GlobalAlloc for StatsAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match small_class(layout) {
            Some(class) => unsafe { SMALL_CACHE.alloc(class, layout.size()) },
            None => {
                self.add_alloc_size(layout.size());
                unsafe { System.alloc(layout) }
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        match small_class(layout) {
            Some(class) => unsafe { SMALL_CACHE.dealloc(ptr, class, layout.size()) },
            None => {
                self.sub_alloc_size(layout.size());
                unsafe { System.dealloc(ptr, layout) }
            }
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        match small_class(layout) {
            Some(class) => {
                let ptr = unsafe { SMALL_CACHE.alloc(class, layout.size()) };
                if !ptr.is_null() {
                    unsafe { ptr.write_bytes(0, layout.size()) };
                }
                ptr
            }
            None => {
                self.add_alloc_size(layout.size());
                unsafe { System.alloc_zeroed(layout) }
            }
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees that new_size, rounded up to the alignment, fits in isize
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let old_class = small_class(layout);
        let new_class = small_class(new_layout);
        if old_class.is_none() && new_class.is_none() {
            if new_size > layout.size() {
                self.add_alloc_size(new_size - layout.size());
            } else if new_size < layout.size() {
                self.sub_alloc_size(layout.size() - new_size);
            }
            return unsafe { System.realloc(ptr, layout, new_size) };
        }
        if old_class.is_some() && old_class == new_class {
            // The block already has class_size() bytes
            let counted = SMALL_CACHE.try_with(|lists| {
                lists.alloc_size = lists.alloc_size.wrapping_add(new_size).wrapping_sub(layout.size());
            });
            if counted.is_none() {
                self.add_alloc_size(new_size);
                self.sub_alloc_size(layout.size());
            }
            return ptr;
        }
        // Move between a small block and another block. Small blocks must come from SmallCache::alloc()
        // to have their full class size, so this doesn't use System.realloc().
        let new_ptr = unsafe { self.alloc(new_layout) };
        if !new_ptr.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size)) };
            unsafe { self.dealloc(ptr, layout) };
        }
        new_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    #[test]
    fn small_class_boundaries() {
        assert_eq!(small_class(layout(0, 1)), None);
        assert_eq!(small_class(layout(1, 1)), Some(0));
        assert_eq!(small_class(layout(24, 8)), Some(0));
        assert_eq!(small_class(layout(25, 8)), Some(1));
        assert_eq!(small_class(layout(40, 16)), Some(1));
        assert_eq!(small_class(layout(41, 8)), Some(2));
        assert_eq!(SMALL_MAX, 1032);
        assert_eq!(small_class(layout(SMALL_MAX, 16)), Some(NUM_CLASSES - 1));
        assert_eq!(small_class(layout(SMALL_MAX + 1, 8)), None);
        assert_eq!(small_class(layout(64, 32)), None);
    }

    #[test]
    fn freed_block_is_reused_by_its_class() {
        let cache = SmallCache::new();
        let class = small_class(layout(40, 8)).unwrap();
        unsafe {
            let p = cache.alloc(class, 40);
            assert_eq!(p as usize % SMALL_ALIGN, 0);
            // A block of the class has class_size() bytes
            p.write_bytes(0xff, class_size(class));
            cache.dealloc(p, class, 40);
            // 25..=40 bytes share a class
            assert_eq!(small_class(layout(25, 16)), Some(class));
            let q = cache.alloc(class, 40);
            assert_eq!(p, q);
            assert_ne!(cache.alloc(class, 40), p);
        }
    }

    #[test]
    fn cache_is_bounded_per_class() {
        let cache = SmallCache::new();
        let class = NUM_CLASSES - 1;
        let n = 2 * CLASS_CACHE_BYTES / class_size(class);
        unsafe {
            let blocks: Vec<*mut u8> = (0..n).map(|_| cache.alloc(class, 40)).collect();
            for &p in &blocks {
                cache.dealloc(p, class, 40);
            }
        }
        let free_bytes = cache.try_with(|lists| lists.free_bytes[class]).unwrap();
        assert!(free_bytes <= CLASS_CACHE_BYTES && free_bytes + class_size(class) > CLASS_CACHE_BYTES);
    }

    #[test]
    fn counts_requested_bytes() {
        let cache = SmallCache::new();
        let alloc_size = || cache.try_with(|lists| lists.alloc_size).unwrap();
        unsafe {
            let p = cache.alloc(2, 40);
            let q = cache.alloc(0, 3);
            assert_eq!(alloc_size(), 43);
            cache.dealloc(p, 2, 40);
            assert_eq!(alloc_size(), 3);
            cache.dealloc(q, 0, 3);
            assert_eq!(alloc_size(), 0);
        }
    }

    #[test]
    fn locked_cache_falls_back_to_system_allocator() {
        let cache = SmallCache::new();
        unsafe {
            let p = cache.alloc(0, 1);
            cache.dealloc(p, 0, 1);
            let q = cache.try_with(|_| cache.alloc(0, 1)).unwrap();
            assert_ne!(p, q);
            cache.try_with(|_| cache.dealloc(q, 0, 1)).unwrap();
            assert_eq!(cache.try_with(|lists| lists.free_bytes[0]), Some(class_size(0)));
        }
    }

    #[test]
    fn alloc_zeroed_returns_zeroed_small_blocks() {
        let a = layout(100, 4);
        unsafe {
            for _ in 0..10 {
                let p = GLOBAL_ALLOCATOR.alloc(a);
                p.write_bytes(0xab, 100);
                GLOBAL_ALLOCATOR.dealloc(p, a);
                let q = GLOBAL_ALLOCATOR.alloc_zeroed(a);
                assert!(std::slice::from_raw_parts(q, 100).iter().all(|&b| b == 0));
                GLOBAL_ALLOCATOR.dealloc(q, a);
            }
        }
    }

    #[test]
    fn realloc_keeps_contents_across_classes() {
        let mut v: Vec<u8> = Vec::with_capacity(1);
        for i in 0..5000 {
            v.push(i as u8);
        }
        v.truncate(10);
        v.shrink_to_fit();
        assert_eq!(v, (0..10u8).collect::<Vec<_>>());
        unsafe {
            let p = GLOBAL_ALLOCATOR.alloc(layout(20, 8));
            p.write_bytes(7, 20);
            // Same class: the block already has 24 bytes
            assert_eq!(GLOBAL_ALLOCATOR.realloc(p, layout(20, 8), 24), p);
            let big = GLOBAL_ALLOCATOR.realloc(p, layout(24, 8), 4000);
            assert!(std::slice::from_raw_parts(big, 20).iter().all(|&b| b == 7));
            let small = GLOBAL_ALLOCATOR.realloc(big, layout(4000, 8), 8);
            assert!(std::slice::from_raw_parts(small, 8).iter().all(|&b| b == 7));
            GLOBAL_ALLOCATOR.dealloc(small, layout(8, 8));
        }
    }

    #[test]
    fn blocks_move_between_threads() {
        let v: Vec<u64> = (0..50).collect();
        let v = std::thread::spawn(move || {
            // Blocks allocated on one thread can be freed into the cache on another
            let w: Vec<u64> = v.iter().map(|x| x * 2).collect();
            drop(v);
            w
        }).join().unwrap();
        assert_eq!(v[49], 98);
    }
}
