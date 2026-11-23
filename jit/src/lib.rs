//! Shared code between YJIT and ZJIT.
#![warn(unsafe_op_in_unsafe_fn)] // Adopt 2024 edition default when targeting 2021 editions

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

#[global_allocator]
pub static GLOBAL_ALLOCATOR: StatsAlloc = StatsAlloc { alloc_size: AtomicUsize::new(0) };

pub struct StatsAlloc {
    pub alloc_size: AtomicUsize,
}

/// Blocks of up to this many bytes are recycled through [`SmallCache`].
const SMALL_MAX: usize = 1024;
/// Small blocks are rounded up to a multiple of this, which is also their alignment.
const SMALL_ALIGN: usize = 16;
const NUM_CLASSES: usize = SMALL_MAX / SMALL_ALIGN;
/// Free bytes each size class may keep before blocks go back to the system allocator.
const CLASS_CACHE_BYTES: usize = 32 * 1024;

/// Per-thread free lists of small blocks, one per size class.
///
/// A compile allocates and frees millions of small Vecs and Boxes. glibc's tcache keeps only 7
/// free blocks per size, so most of them went through glibc's slower bins instead. Blocks in a
/// free list are linked through their first word. Every block in class `c` was allocated from
/// the system allocator with [`class_size`]`(c)` bytes, so it can serve any request of that class.
struct SmallCache {
    heads: [*mut u8; NUM_CLASSES],
    free_bytes: [usize; NUM_CLASSES],
}

impl Drop for SmallCache {
    fn drop(&mut self) {
        // Give the blocks back when the thread exits
        for class in 0..NUM_CLASSES {
            let layout = class_layout(class);
            let mut block = self.heads[class];
            while !block.is_null() {
                let next = unsafe { *(block as *mut *mut u8) };
                unsafe { System.dealloc(block, layout) };
                block = next;
            }
            self.heads[class] = std::ptr::null_mut();
            self.free_bytes[class] = 0;
        }
    }
}

thread_local! {
    static SMALL_CACHE: UnsafeCell<SmallCache> = const {
        UnsafeCell::new(SmallCache { heads: [std::ptr::null_mut(); NUM_CLASSES], free_bytes: [0; NUM_CLASSES] })
    };
}

/// The size class of a layout if it's served by [`SmallCache`]
fn small_class(layout: Layout) -> Option<usize> {
    if layout.size() != 0 && layout.size() <= SMALL_MAX && layout.align() <= SMALL_ALIGN {
        Some((layout.size() - 1) / SMALL_ALIGN)
    } else {
        None
    }
}

fn class_size(class: usize) -> usize {
    (class + 1) * SMALL_ALIGN
}

fn class_layout(class: usize) -> Layout {
    // SAFETY: the size is a non-zero multiple of the alignment, which is a power of two
    unsafe { Layout::from_size_align_unchecked(class_size(class), SMALL_ALIGN) }
}

/// Pop a block of a size class, or allocate one from the system allocator
unsafe fn small_alloc(class: usize) -> *mut u8 {
    let cached = SMALL_CACHE.try_with(|cache| {
        // SAFETY: the cache is only borrowed within this closure, and the allocator isn't reentered
        let cache = unsafe { &mut *cache.get() };
        let block = cache.heads[class];
        if !block.is_null() {
            cache.heads[class] = unsafe { *(block as *mut *mut u8) };
            cache.free_bytes[class] -= class_size(class);
        }
        block
    }).unwrap_or(std::ptr::null_mut());
    if !cached.is_null() {
        return cached;
    }
    unsafe { System.alloc(class_layout(class)) }
}

/// Push a block of a size class, or free it if the class has cached enough
unsafe fn small_dealloc(ptr: *mut u8, class: usize) {
    let cached = SMALL_CACHE.try_with(|cache| {
        // SAFETY: the cache is only borrowed within this closure, and the allocator isn't reentered
        let cache = unsafe { &mut *cache.get() };
        if cache.free_bytes[class] + class_size(class) > CLASS_CACHE_BYTES {
            return false;
        }
        unsafe { *(ptr as *mut *mut u8) = cache.heads[class] };
        cache.heads[class] = ptr;
        cache.free_bytes[class] += class_size(class);
        true
    }).unwrap_or(false);
    if !cached {
        unsafe { System.dealloc(ptr, class_layout(class)) };
    }
}

unsafe impl GlobalAlloc for StatsAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.alloc_size.fetch_add(layout.size(), Ordering::SeqCst);
        match small_class(layout) {
            Some(class) => unsafe { small_alloc(class) },
            None => unsafe { System.alloc(layout) },
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.alloc_size.fetch_sub(layout.size(), Ordering::SeqCst);
        match small_class(layout) {
            Some(class) => unsafe { small_dealloc(ptr, class) },
            None => unsafe { System.dealloc(ptr, layout) },
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.alloc_size.fetch_add(layout.size(), Ordering::SeqCst);
        match small_class(layout) {
            Some(class) => {
                let ptr = unsafe { small_alloc(class) };
                if !ptr.is_null() {
                    unsafe { ptr.write_bytes(0, layout.size()) };
                }
                ptr
            }
            None => unsafe { System.alloc_zeroed(layout) },
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            self.alloc_size.fetch_add(new_size - layout.size(), Ordering::SeqCst);
        } else if new_size < layout.size() {
            self.alloc_size.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
        }
        // SAFETY: the caller guarantees that new_size, rounded up to the alignment, fits in isize
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let old_class = small_class(layout);
        let new_class = small_class(new_layout);
        if old_class.is_none() && new_class.is_none() {
            return unsafe { System.realloc(ptr, layout, new_size) };
        }
        if old_class.is_some() && old_class == new_class {
            // The block already has class_size() bytes
            return ptr;
        }
        // Move between a small block and another block. Small blocks must come from small_alloc()
        // to have their full class size, so this doesn't use System.realloc().
        let new_ptr = match new_class {
            Some(class) => unsafe { small_alloc(class) },
            None => unsafe { System.alloc(new_layout) },
        };
        if !new_ptr.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size)) };
            match old_class {
                Some(class) => unsafe { small_dealloc(ptr, class) },
                None => unsafe { System.dealloc(ptr, layout) },
            }
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

    fn cached_bytes(class: usize) -> usize {
        SMALL_CACHE.with(|cache| unsafe { (*cache.get()).free_bytes[class] })
    }

    #[test]
    fn small_class_boundaries() {
        assert_eq!(small_class(layout(0, 1)), None);
        assert_eq!(small_class(layout(1, 1)), Some(0));
        assert_eq!(small_class(layout(16, 8)), Some(0));
        assert_eq!(small_class(layout(17, 8)), Some(1));
        assert_eq!(small_class(layout(SMALL_MAX, 16)), Some(NUM_CLASSES - 1));
        assert_eq!(small_class(layout(SMALL_MAX + 1, 8)), None);
        assert_eq!(small_class(layout(64, 32)), None);
    }

    #[test]
    fn freed_small_block_is_reused_by_its_class() {
        let a = layout(40, 8);
        unsafe {
            let p = GLOBAL_ALLOCATOR.alloc(a);
            assert_eq!(p as usize % SMALL_ALIGN, 0);
            GLOBAL_ALLOCATOR.dealloc(p, a);
            // 33..=48 bytes share a class
            let q = GLOBAL_ALLOCATOR.alloc(layout(48, 16));
            assert_eq!(p, q);
            q.write_bytes(0xff, 48);
            GLOBAL_ALLOCATOR.dealloc(q, layout(48, 16));
        }
    }

    #[test]
    fn alloc_zeroed_clears_a_reused_block() {
        let a = layout(100, 4);
        unsafe {
            let p = GLOBAL_ALLOCATOR.alloc(a);
            p.write_bytes(0xab, 100);
            GLOBAL_ALLOCATOR.dealloc(p, a);
            let q = GLOBAL_ALLOCATOR.alloc_zeroed(a);
            assert_eq!(p, q);
            assert!(std::slice::from_raw_parts(q, 100).iter().all(|&b| b == 0));
            GLOBAL_ALLOCATOR.dealloc(q, a);
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
            // Same class: the block already has 32 bytes
            assert_eq!(GLOBAL_ALLOCATOR.realloc(p, layout(20, 8), 30), p);
            let big = GLOBAL_ALLOCATOR.realloc(p, layout(30, 8), 4000);
            assert!(std::slice::from_raw_parts(big, 20).iter().all(|&b| b == 7));
            let small = GLOBAL_ALLOCATOR.realloc(big, layout(4000, 8), 8);
            assert!(std::slice::from_raw_parts(small, 8).iter().all(|&b| b == 7));
            GLOBAL_ALLOCATOR.dealloc(small, layout(8, 8));
        }
    }

    #[test]
    fn cache_is_bounded_per_class() {
        let a = layout(SMALL_MAX, 8);
        let class = small_class(a).unwrap();
        let n = 2 * CLASS_CACHE_BYTES / SMALL_MAX;
        unsafe {
            let blocks: Vec<*mut u8> = (0..n).map(|_| GLOBAL_ALLOCATOR.alloc(a)).collect();
            for &p in &blocks {
                GLOBAL_ALLOCATOR.dealloc(p, a);
            }
        }
        assert_eq!(cached_bytes(class), CLASS_CACHE_BYTES);
    }

    #[test]
    fn blocks_move_between_threads() {
        let v: Vec<u64> = (0..50).collect();
        let v = std::thread::spawn(move || {
            // Freed into this thread's cache, which goes back to the system when the thread exits
            let w: Vec<u64> = v.iter().map(|x| x * 2).collect();
            drop(v);
            w
        }).join().unwrap();
        assert_eq!(v[49], 98);
    }
}
