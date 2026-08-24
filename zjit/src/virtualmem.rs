//! Memory management stuff for ZJIT's code storage. Deals with virtual memory.
// I'm aware that there is an experiment in Rust Nightly right now for to see if banning
// usize->pointer casts is viable. It seems like a lot of work for us to participate for not much
// benefit.

use std::ptr::NonNull;
use crate::cruby::*;
use crate::stats::zjit_alloc_bytes;

pub type VirtualMem = VirtualMemory<sys::SystemAllocator>;

/// Memory for generated executable machine code. When not testing, we reserve address space for
/// the entire region upfront and map physical memory into the reserved address space as needed. On
/// Linux, this is basically done using an `mmap` with `PROT_NONE` upfront and gradually using
/// `mprotect` with `PROT_READ|PROT_WRITE` as needed. The WIN32 equivalent seems to be
/// `VirtualAlloc` with `MEM_RESERVE` then later with `MEM_COMMIT`.
///
/// This handles ["W^X"](https://en.wikipedia.org/wiki/W%5EX) semi-automatically. Writes
/// are always accepted and once writes are done a call to [Self::mark_all_executable] makes
/// the code in the region executable.
pub struct VirtualMemory<A: Allocator> {
    /// Location of the virtual memory region.
    region_start: NonNull<u8>,

    /// Size of this virtual memory region in bytes.
    region_size_bytes: usize,

    /// Offset at which the outlined (cold) arena starts. The region is one
    /// reservation split into two independently growing arenas: hot code is
    /// written into `[0, outlined_start_bytes)` and cold code -- side exits,
    /// function stubs -- into `[outlined_start_bytes, region_size_bytes)`.
    /// Keeping them in one reservation is what lets a [`CodePtr`] stay a single
    /// offset from one base, and what keeps every hot-to-cold branch inside
    /// rel32 range.
    ///
    /// Equal to `region_size_bytes` when the region is not split, which makes
    /// the outlined arena empty and everything behave as it did before.
    outlined_start_bytes: usize,

    /// mapped_region_bytes + zjit_alloc_size may not increase beyond this limit.
    memory_limit_bytes: Option<usize>,

    /// Number of bytes per "page", memory protection permission can only be controlled at this
    /// granularity.
    page_size_bytes: usize,

    /// Number of bytes that have we have allocated physical memory for starting at
    /// [Self::region_start]. Covers the inlined arena only.
    mapped_region_bytes: usize,

    /// Number of bytes we have allocated physical memory for starting at
    /// `region_start + outlined_start_bytes`. The two arenas map pages
    /// independently, so the address space between them stays unmapped.
    outlined_mapped_bytes: usize,

    /// Keep track of the address of the last written to page.
    /// Used for changing protection to implement W^X.
    current_write_page: Option<usize>,

    /// Zero size member for making syscalls to get physical memory during normal operation.
    /// When testing this owns some memory.
    allocator: A,
}

/// Groups together the two syscalls to get get new physical memory and to change
/// memory protection. See [VirtualMemory] for details.
pub trait Allocator {
    #[must_use]
    fn mark_writable(&mut self, ptr: *const u8, size: u32) -> bool;

    fn mark_executable(&mut self, ptr: *const u8, size: u32);

    fn mark_unused(&mut self, ptr: *const u8, size: u32) -> bool;
}

/// Pointer into a [VirtualMemory] represented as an offset from the base.
/// Note: there is no NULL constant for [CodePtr]. You should use `Option<CodePtr>` instead.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Debug)]
#[repr(C, packed)]
pub struct CodePtr(u32);

impl CodePtr {
    /// Advance the CodePtr. Can return a dangling pointer.
    pub fn add_bytes(self, bytes: usize) -> Self {
        let CodePtr(raw) = self;
        let bytes: u32 = bytes.try_into().unwrap();
        CodePtr(raw + bytes)
    }

    /// Subtract bytes from the CodePtr
    pub fn sub_bytes(self, bytes: usize) -> Self {
        let CodePtr(raw) = self;
        let bytes: u32 = bytes.try_into().unwrap();
        CodePtr(raw.saturating_sub(bytes))
    }

    /// Note that the raw pointer might be dangling if there hasn't
    /// been any writes to it through the [VirtualMemory] yet.
    pub fn raw_ptr(self, base: &impl CodePtrBase) -> *const u8 {
        let CodePtr(offset) = self;
        base.base_ptr().as_ptr().wrapping_add(offset as usize)
    }

    /// Get the address of the code pointer.
    pub fn raw_addr(self, base: &impl CodePtrBase) -> usize {
        self.raw_ptr(base).addr()
    }

    /// Get the offset component for the code pointer. Useful finding the distance between two
    /// code pointers that share the same [VirtualMem].
    pub fn as_offset(self) -> i64 {
        let CodePtr(offset) = self;
        offset.into()
    }
}

/// Errors that can happen when writing to [VirtualMemory]
#[derive(Debug, PartialEq)]
pub enum WriteError {
    OutOfBounds,
    FailedPageMapping,
}

use WriteError::*;

impl VirtualMem {
    /// Allocate a VirtualMem instance with a requested size, split into an inlined
    /// (hot) arena and an outlined (cold) arena of equal size. See
    /// [`VirtualMemory::outlined_start_bytes`].
    pub fn alloc_split(exec_mem_bytes: usize, mem_bytes: Option<usize>) -> Self {
        // The requested size is a whole number of MiB, so half of it is a whole
        // number of 512 KiB and page-aligned for any page size we support.
        let outlined_start_bytes = exec_mem_bytes / 2;
        Self::alloc_with_outlined_start(exec_mem_bytes, mem_bytes, outlined_start_bytes)
    }

    /// Allocate a VirtualMem instance with a requested size and no outlined arena.
    /// Used by tests, which want the old single-arena layout.
    pub fn alloc(exec_mem_bytes: usize, mem_bytes: Option<usize>) -> Self {
        Self::alloc_with_outlined_start(exec_mem_bytes, mem_bytes, exec_mem_bytes)
    }

    fn alloc_with_outlined_start(exec_mem_bytes: usize, mem_bytes: Option<usize>, outlined_start_bytes: usize) -> Self {
        let virt_block: *mut u8 = unsafe { rb_jit_reserve_addr_space(exec_mem_bytes as u32) };

        // Memory protection syscalls need page-aligned addresses, so check it here. Assuming
        // `virt_block` is page-aligned, `second_half` should be page-aligned as long as the
        // page size in bytes is a power of two 2^19 or smaller. This is because the user
        // requested size is half of mem_option * 2^20 as it's in MiB.
        //
        // Basically, we don't support x86-64 2MiB and 1GiB pages. ARMv8 can do up to 64KiB
        // (2^16 bytes) pages, which should be fine. 4KiB pages seem to be the most popular though.
        let page_size = unsafe { rb_jit_get_page_size() };
        assert_eq!(
            virt_block as usize % page_size as usize, 0,
            "Start of virtual address block should be page-aligned",
        );

        Self::new(sys::SystemAllocator {}, page_size, NonNull::new(virt_block).unwrap(), exec_mem_bytes, mem_bytes, outlined_start_bytes)
    }

    /// Reserve `size` bytes of address space below `INT32_MAX` for JITFrame
    pub fn alloc_low(size: usize) -> Option<Self> {
        let virt_block = unsafe { rb_zjit_reserve_low_addr_space(size) } as *mut u8;
        let virt_block = NonNull::new(virt_block)?;
        let page_size = unsafe { rb_jit_get_page_size() };
        Some(Self::new(sys::SystemAllocator {}, page_size, virt_block, size, None, size))
    }
}

impl<A: Allocator> VirtualMemory<A> {
    /// Bring a part of the address space under management.
    pub fn new(
        allocator: A,
        page_size: u32,
        virt_region_start: NonNull<u8>,
        region_size_bytes: usize,
        memory_limit_bytes: Option<usize>,
        outlined_start_bytes: usize,
    ) -> Self {
        assert_ne!(0, page_size);
        let page_size_bytes = page_size as usize;
        assert!(outlined_start_bytes <= region_size_bytes, "outlined arena should start inside the region");
        if outlined_start_bytes < region_size_bytes {
            // An unsplit region names the end of the region here, which need not be
            // page-aligned (some tests ask for an eight-byte region). A real split
            // must be, so that the two arenas never share a page and so can have
            // their protections changed independently.
            assert_eq!(0, outlined_start_bytes % page_size_bytes,
                "the arena split should be page-aligned so the two arenas never share a page");
        }

        Self {
            region_start: virt_region_start,
            region_size_bytes,
            outlined_start_bytes,
            memory_limit_bytes,
            page_size_bytes,
            mapped_region_bytes: 0,
            outlined_mapped_bytes: 0,
            current_write_page: None,
            allocator,
        }
    }

    /// Return the start of the region as a raw pointer. Note that it could be a dangling
    /// pointer so be careful dereferencing it.
    pub fn start_ptr(&self) -> CodePtr {
        CodePtr(0)
    }

    pub fn mapped_end_ptr(&self) -> CodePtr {
        self.start_ptr().add_bytes(self.mapped_region_bytes)
    }

    pub fn virtual_end_ptr(&self) -> CodePtr {
        self.start_ptr().add_bytes(self.region_size_bytes)
    }

    /// Offset at which the outlined arena starts, or `virtual_region_size()` when
    /// the region is not split.
    pub fn outlined_start_bytes(&self) -> usize {
        self.outlined_start_bytes
    }

    /// Size of the region in bytes that we have allocated physical memory for,
    /// counting both arenas.
    pub fn mapped_region_size(&self) -> usize {
        self.mapped_region_bytes + self.outlined_mapped_bytes
    }

    /// Physical memory backing the inlined (hot) arena.
    pub fn inlined_mapped_size(&self) -> usize {
        self.mapped_region_bytes
    }

    /// Physical memory backing the outlined (cold) arena.
    pub fn outlined_mapped_size(&self) -> usize {
        self.outlined_mapped_bytes
    }

    /// Size of the region in bytes where writes could be attempted.
    pub fn virtual_region_size(&self) -> usize {
        self.region_size_bytes
    }

    /// The granularity at which we can control memory permission.
    /// On Linux, this is the page size that mmap(2) talks about.
    pub fn system_page_size(&self) -> usize {
        self.page_size_bytes
    }

    /// Write a single byte. The first write to a page makes it readable.
    pub fn write_byte(&mut self, write_ptr: CodePtr, byte: u8) -> Result<(), WriteError> {
        let page_size = self.page_size_bytes;
        let raw: *mut u8 = write_ptr.raw_ptr(self) as *mut u8;
        let page_addr = (raw as usize / page_size) * page_size;

        if self.current_write_page == Some(page_addr) {
            // Writing within the last written to page, nothing to do
        } else {
            // Switching to a different and potentially new page.
            //
            // Pick the arena this address belongs to. The two arenas grow
            // independently, so `start` is the arena's own base and
            // `whole_region_end` its own end; the address space belonging to the
            // other arena is out of bounds from here.
            let region_start = self.region_start.as_ptr();
            let split = region_start.wrapping_add(self.outlined_start_bytes);
            let outlined = raw >= split;
            let (start, whole_region_end, arena_mapped_bytes) = if outlined {
                (split, region_start.wrapping_add(self.region_size_bytes), self.outlined_mapped_bytes)
            } else {
                (region_start, split, self.mapped_region_bytes)
            };
            let mapped_region_end = start.wrapping_add(arena_mapped_bytes);
            let other_arena_mapped_bytes = self.mapped_region_size() - arena_mapped_bytes;
            let alloc = &mut self.allocator;

            // Ignore zjit_alloc_size() if self.memory_limit_bytes is None for testing
            let mut required_region_bytes =
                other_arena_mapped_bytes + (page_addr + page_size - start as usize);
            if self.memory_limit_bytes.is_some() {
                required_region_bytes += zjit_alloc_bytes();
            }

            assert!((start..=whole_region_end).contains(&mapped_region_end));

            if (start..mapped_region_end).contains(&raw) {
                // Writing to a previously written to page.
                // Need to make page writable, but no need to fill.
                let page_size: u32 = page_size.try_into().unwrap();
                if !alloc.mark_writable(page_addr as *const _, page_size) {
                    return Err(FailedPageMapping);
                }

                self.current_write_page = Some(page_addr);
            } else if (start..whole_region_end).contains(&raw) &&
                    required_region_bytes < self.memory_limit_bytes.unwrap_or(self.region_size_bytes) {
                // Writing to a brand new page
                let mapped_region_end_addr = mapped_region_end as usize;
                let alloc_size = page_addr - mapped_region_end_addr + page_size;

                assert_eq!(0, alloc_size % page_size, "allocation size should be page aligned");
                assert_eq!(0, mapped_region_end_addr % page_size, "pointer should be page aligned");

                if alloc_size > page_size {
                    // This is unusual for the current setup, so keep track of it.
                    //crate::stats::incr_counter!(exec_mem_non_bump_alloc); // TODO
                }

                // Allocate new chunk
                let alloc_size_u32: u32 = alloc_size.try_into().unwrap();
                unsafe {
                    if !alloc.mark_writable(mapped_region_end.cast(), alloc_size_u32) {
                        return Err(FailedPageMapping);
                    }
                    if cfg!(target_arch = "x86_64") {
                        // Fill new memory with PUSH DS (0x1E) so that executing uninitialized memory
                        // will fault with #UD in 64-bit mode. On Linux it becomes SIGILL and use the
                        // usual Ruby crash reporter.
                        std::slice::from_raw_parts_mut(mapped_region_end, alloc_size).fill(0x1E);
                    } else if cfg!(target_arch = "aarch64") {
                        // In aarch64, all zeros encodes UDF, so it's already what we want.
                    } else {
                        unreachable!("unknown arch");
                    }
                }
                if outlined {
                    self.outlined_mapped_bytes += alloc_size;
                } else {
                    self.mapped_region_bytes += alloc_size;
                }

                self.current_write_page = Some(page_addr);
            } else {
                return Err(OutOfBounds);
            }
        }

        // We have permission to write if we get here
        unsafe { raw.write(byte) };

        Ok(())
    }

    /// Return true if write_byte() can allocate a new page
    pub fn can_allocate(&self) -> bool {
        let memory_usage_bytes = self.mapped_region_size() + zjit_alloc_bytes();
        let memory_limit_bytes = self.memory_limit_bytes.unwrap_or(self.region_size_bytes);
        memory_usage_bytes + self.page_size_bytes < memory_limit_bytes
    }

    /// Make all the code in the region writable. Call this before bulk writes (e.g. GC
    /// reference updates). See [Self] for usual usage flow.
    pub fn mark_all_writable(&mut self) {
        self.current_write_page = None;

        // Both arenas, and only the pages each of them has actually mapped: the
        // address space between them was never committed.
        for (start, bytes) in self.mapped_arenas() {
            let bytes: u32 = bytes.try_into().unwrap();
            if !self.allocator.mark_writable(start, bytes) {
                panic!("Cannot make JIT memory region writable");
            }
        }
    }

    /// Make all the code in the region executable. Call this at the end of a write session.
    /// See [Self] for usual usage flow.
    pub fn mark_all_executable(&mut self) {
        self.current_write_page = None;

        for (start, bytes) in self.mapped_arenas() {
            self.allocator.mark_executable(start, bytes.try_into().unwrap());
        }
    }

    /// The `(start, len)` of each arena's mapped pages, skipping arenas that have
    /// none. Protection changes have to cover both, and must not cover the
    /// uncommitted gap in between.
    fn mapped_arenas(&self) -> Vec<(*const u8, usize)> {
        let region_start = self.region_start.as_ptr();
        let mut arenas = Vec::with_capacity(2);
        if self.mapped_region_bytes > 0 {
            arenas.push((region_start as *const u8, self.mapped_region_bytes));
        }
        if self.outlined_mapped_bytes > 0 {
            let outlined_start = region_start.wrapping_add(self.outlined_start_bytes);
            arenas.push((outlined_start as *const u8, self.outlined_mapped_bytes));
        }
        arenas
    }

    /// Free a range of bytes. start_ptr must be memory page-aligned.
    pub fn free_bytes(&mut self, start_ptr: CodePtr, size: u32) {
        assert_eq!(start_ptr.raw_ptr(self) as usize % self.page_size_bytes, 0);

        // Bounds check the request. We should only free memory we manage.
        let mapped_region = self.start_ptr().raw_ptr(self)..self.mapped_end_ptr().raw_ptr(self);
        let virtual_region = self.start_ptr().raw_ptr(self)..self.virtual_end_ptr().raw_ptr(self);
        let last_byte_to_free = start_ptr.add_bytes(size.saturating_sub(1) as usize).raw_ptr(self);
        assert!(mapped_region.contains(&start_ptr.raw_ptr(self)));
        // On platforms where code page size != memory page size (e.g. Linux), we often need
        // to free code pages that contain unmapped memory pages. When it happens on the last
        // code page, it's more appropriate to check the last byte against the virtual region.
        assert!(virtual_region.contains(&last_byte_to_free));

        self.allocator.mark_unused(start_ptr.raw_ptr(self), size);
    }
}

/// Something that could provide a base pointer to compute a raw pointer from a [CodePtr].
pub trait CodePtrBase {
    fn base_ptr(&self) -> NonNull<u8>;
}

impl<A: Allocator> CodePtrBase for VirtualMemory<A> {
    fn base_ptr(&self) -> NonNull<u8> {
        self.region_start
    }
}

/// Requires linking with CRuby to work
pub mod sys {
    use crate::cruby::*;

    /// Zero size! This just groups together syscalls that require linking with CRuby.
    pub struct SystemAllocator;

    type VoidPtr = *mut std::os::raw::c_void;

    impl super::Allocator for SystemAllocator {
        fn mark_writable(&mut self, ptr: *const u8, size: u32) -> bool {
            crate::stats::trace_compile_phase("mark_writable", || {
                unsafe { rb_jit_mark_writable(ptr as VoidPtr, size) }
            })
        }

        fn mark_executable(&mut self, ptr: *const u8, size: u32) {
            crate::stats::trace_compile_phase("mark_executable", || {
                unsafe { rb_jit_mark_executable(ptr as VoidPtr, size) }
            })
        }

        fn mark_unused(&mut self, ptr: *const u8, size: u32) -> bool {
            crate::stats::trace_compile_phase("mark_unused", || {
                unsafe { rb_jit_mark_unused(ptr as VoidPtr, size) }
            })
        }
    }
}


#[cfg(test)]
pub mod tests {
    use super::*;

    // Track allocation requests and owns some fixed size backing memory for requests.
    // While testing we don't execute generated code.
    pub struct TestingAllocator {
        requests: Vec<AllocRequest>,
        memory: Vec<u8>,
    }

    #[derive(Debug)]
    enum AllocRequest {
        MarkWritable{ start_idx: usize, length: usize },
        MarkExecutable{ start_idx: usize, length: usize },
        MarkUnused,
    }
    use AllocRequest::*;

    impl TestingAllocator {
        pub fn new(mem_size: usize) -> Self {
            Self { requests: Vec::default(), memory: vec![0; mem_size] }
        }

        pub fn mem_start(&self) -> *const u8 {
            self.memory.as_ptr()
        }

        // Verify that write_byte() bounds checks. Return `ptr` as an index.
        fn bounds_check_request(&self, ptr: *const u8, size: u32) -> usize {
            let mem_start = self.memory.as_ptr() as usize;
            let index = ptr as usize - mem_start;

            assert!(index < self.memory.len());
            assert!(index + size as usize <= self.memory.len());

            index
        }
    }

    // Bounds check and then record the request
    impl super::Allocator for TestingAllocator {
        fn mark_writable(&mut self, ptr: *const u8, length: u32) -> bool {
            let index = self.bounds_check_request(ptr, length);
            self.requests.push(MarkWritable { start_idx: index, length: length as usize });

            true
        }

        fn mark_executable(&mut self, ptr: *const u8, length: u32) {
            let index = self.bounds_check_request(ptr, length);
            self.requests.push(MarkExecutable { start_idx: index, length: length as usize });

            // We don't try to execute generated code in cfg(test)
            // so no need to actually request executable memory.
        }

        fn mark_unused(&mut self, ptr: *const u8, length: u32) -> bool {
            self.bounds_check_request(ptr, length);
            self.requests.push(MarkUnused);

            true
        }
    }

    // Fictional architecture where each page is 4 bytes long
    const PAGE_SIZE: usize = 4;
    fn new_dummy_virt_mem() -> VirtualMemory<TestingAllocator> {
        new_dummy_virt_mem_with_outlined_start(PAGE_SIZE * 10)
    }

    fn new_dummy_virt_mem_with_outlined_start(outlined_start: usize) -> VirtualMemory<TestingAllocator> {
        unsafe {
            if crate::options::OPTIONS.is_none() {
                crate::options::OPTIONS = Some(crate::options::Options::default());
            }
        }

        let mem_size = PAGE_SIZE * 10;
        let alloc = TestingAllocator::new(mem_size);
        let mem_start: *const u8 = alloc.mem_start();

        VirtualMemory::new(
            alloc,
            PAGE_SIZE.try_into().unwrap(),
            NonNull::new(mem_start as *mut u8).unwrap(),
            mem_size,
            None,
            outlined_start,
        )
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn new_memory_is_initialized() {
        let mut virt = new_dummy_virt_mem();

        virt.write_byte(virt.start_ptr(), 1).unwrap();
        assert!(
            virt.allocator.memory[..PAGE_SIZE].iter().all(|&byte| byte != 0),
            "Entire page should be initialized",
        );

        // Skip a few page
        let three_pages = 3 * PAGE_SIZE;
        virt.write_byte(virt.start_ptr().add_bytes(three_pages), 1).unwrap();
        assert!(
            virt.allocator.memory[..three_pages].iter().all(|&byte| byte != 0),
            "Gaps between write requests should be filled",
        );
    }

    #[test]
    fn no_redundant_syscalls_when_writing_to_the_same_page() {
        let mut virt = new_dummy_virt_mem();

        virt.write_byte(virt.start_ptr(), 1).unwrap();
        virt.write_byte(virt.start_ptr(), 0).unwrap();

        assert!(
            matches!(
                virt.allocator.requests[..],
                [MarkWritable { start_idx: 0, length: PAGE_SIZE }],
            )
        );
    }

    #[test]
    fn bounds_checking() {
        use super::WriteError::*;
        let mut virt = new_dummy_virt_mem();

        let one_past_end = virt.start_ptr().add_bytes(virt.virtual_region_size());
        assert_eq!(Err(OutOfBounds), virt.write_byte(one_past_end, 0));

        let end_of_addr_space = CodePtr(u32::MAX);
        assert_eq!(Err(OutOfBounds), virt.write_byte(end_of_addr_space, 0));
    }

    #[test]
    fn only_written_to_regions_become_executable() {
        // ... so we catch attempts to read/write/execute never-written-to regions
        const THREE_PAGES: usize = PAGE_SIZE * 3;
        let mut virt = new_dummy_virt_mem();
        let page_two_start = virt.start_ptr().add_bytes(PAGE_SIZE * 2);
        virt.write_byte(page_two_start, 1).unwrap();
        virt.mark_all_executable();

        assert!(virt.virtual_region_size() > THREE_PAGES);
        assert!(
            matches!(
                virt.allocator.requests[..],
                [
                    MarkWritable { start_idx: 0, length: THREE_PAGES },
                    MarkExecutable { start_idx: 0, length: THREE_PAGES },
                ]
            ),
        );
    }

    #[test]
    fn split_arenas_map_pages_independently() {
        // The outlined arena starts halfway in. Writing to it must not commit the
        // address space in between, which is what a single bump watermark would do.
        const HALF: usize = PAGE_SIZE * 5;
        let mut virt = new_dummy_virt_mem_with_outlined_start(HALF);

        virt.write_byte(virt.start_ptr(), 1).unwrap();
        virt.write_byte(virt.start_ptr().add_bytes(HALF), 1).unwrap();

        assert_eq!(PAGE_SIZE, virt.inlined_mapped_size());
        assert_eq!(PAGE_SIZE, virt.outlined_mapped_size());
        assert_eq!(PAGE_SIZE * 2, virt.mapped_region_size());

        // Both arenas' pages get protection changes, and nothing in between does.
        virt.mark_all_executable();
        assert!(
            matches!(
                virt.allocator.requests[..],
                [
                    MarkWritable { start_idx: 0, length: PAGE_SIZE },
                    MarkWritable { start_idx: HALF, length: PAGE_SIZE },
                    MarkExecutable { start_idx: 0, length: PAGE_SIZE },
                    MarkExecutable { start_idx: HALF, length: PAGE_SIZE },
                ]
            ),
            "unexpected requests: {:?}", virt.allocator.requests,
        );
    }

    #[test]
    fn inlined_arena_cannot_grow_into_the_outlined_arena() {
        use super::WriteError::*;
        const HALF: usize = PAGE_SIZE * 5;
        let mut virt = new_dummy_virt_mem_with_outlined_start(HALF);

        // The last byte of the inlined arena is fine...
        virt.write_byte(virt.start_ptr().add_bytes(HALF - 1), 1).unwrap();
        assert_eq!(HALF, virt.inlined_mapped_size());
        assert_eq!(0, virt.outlined_mapped_size());

        // ...but the byte after it belongs to the outlined arena, so it starts that
        // arena's mapping rather than extending the inlined one.
        virt.write_byte(virt.start_ptr().add_bytes(HALF), 1).unwrap();
        assert_eq!(HALF, virt.inlined_mapped_size());
        assert_eq!(PAGE_SIZE, virt.outlined_mapped_size());

        // Past the end of the whole region is still out of bounds.
        let one_past_end = virt.start_ptr().add_bytes(virt.virtual_region_size());
        assert_eq!(Err(OutOfBounds), virt.write_byte(one_past_end, 0));
    }
}
