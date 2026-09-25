//! Model for creating generating textual assembler code.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Range;
use std::rc::Rc;
use std::cell::RefCell;
use std::mem;
use crate::state::rb_zjit_compiling_p;
use crate::virtualmem::*;

// Lots of manual vertical alignment in there that rustfmt doesn't handle well.
#[rustfmt::skip]
#[cfg(target_arch = "x86_64")]
pub mod x86_64;
#[cfg(target_arch = "aarch64")]
pub mod arm64;

/// Index to a label created by cb.new_label()
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct Label(pub usize);

/// The object that knows how to encode the branch instruction.
type BranchEncoder = Box<dyn Fn(&mut CodeBlock, i64, i64) -> Result<(), ()>>;

/// Reference to an ASM label
pub struct LabelRef {
    // Position in the code block where the label reference exists
    pos: usize,

    // Label which this refers to
    label: Label,

    /// The number of bytes that this label reference takes up in the memory.
    /// It's necessary to know this ahead of time so that when we come back to
    /// patch it it takes the same amount of space.
    num_bytes: usize,

    /// The object that knows how to encode the branch instruction.
    encode: BranchEncoder,
}

/// Block of memory into which instructions can be assembled
pub struct CodeBlock {
    // Memory for storing the encoded instructions
    mem_block: Rc<RefCell<VirtualMem>>,

    // Memory block size
    mem_size: usize,

    /// Size of a code page in bytes, or 0 when inline and outlined code are not
    /// interleaved. Like YJIT, each code page is split into two halves: inline (hot)
    /// code is written into the first half of each page and outlined (cold) code
    /// into the second half. When a half fills up, [`Self::next_page`] jumps to the
    /// same half of the next page and moves the other half to that page too, so
    /// cold code always stays within a page or so of the hot code jumping to it.
    /// Must be a multiple of the OS page size.
    page_size: usize,

    // Current writing position, in whichever half `outlined` names
    write_pos: usize,

    /// Write position of the half that is *not* current. [`Self::set_outlined`]
    /// swaps it with `write_pos`.
    other_write_pos: usize,

    /// Whether `write_pos` currently points into the outlined half.
    outlined: bool,

    /// Size reserved at the end of each half page for a jump to the next page.
    page_end_reserve: usize,

    /// When false, writes are not bounded by the end of the current half page.
    /// Used while patching existing code (label linking, invalidation), which
    /// only rewrites bytes that were already reserved by the original write.
    page_bounds_check: bool,

    /// Bytes written to half pages that each half has moved past, indexed by
    /// `outlined as usize`. Used for code size stats.
    past_page_bytes: [usize; 2],

    /// For each half page that was left with a jump to the next page, the
    /// position right after that jump, keyed by the position where the half
    /// page starts. Used to split a code range into the pieces that hold code.
    page_jump_ends: HashMap<usize, usize>,

    // Table of registered label addresses
    label_addrs: Vec<usize>,

    // Table of registered label names
    label_names: Vec<String>,

    // References to labels
    label_refs: Vec<LabelRef>,

    // A switch for keeping comments. They take up memory.
    keep_comments: bool,

    // Comments for assembly instructions, if that feature is enabled
    asm_comments: BTreeMap<usize, Vec<String>>,

    // Set if the CodeBlock is unable to output some instructions,
    // for example, when there is not enough space or when a jump
    // target is too far away.
    dropped_bytes: bool,
}

impl CodeBlock {
    /// Make a new CodeBlock
    pub fn new(mem_block: Rc<RefCell<VirtualMem>>, keep_comments: bool) -> Self {
        let mem_size = mem_block.borrow().virtual_region_size();
        Self {
            mem_block,
            mem_size,
            page_size: 0,
            write_pos: 0,
            other_write_pos: 0,
            outlined: false,
            page_end_reserve: 0,
            page_bounds_check: true,
            past_page_bytes: [0, 0],
            page_jump_ends: HashMap::new(),
            label_addrs: Vec::new(),
            label_names: Vec::new(),
            label_refs: Vec::new(),
            keep_comments,
            asm_comments: BTreeMap::new(),
            dropped_bytes: false,
        }
    }

    /// Works for common AArch64 systems that have 16 KiB pages and
    /// common x86_64 systems that use 4 KiB pages. Same as YJIT.
    const PREFERRED_CODE_PAGE_SIZE: usize = 16 * 1024;

    /// Make a new CodeBlock that interleaves inline and outlined code per code page like YJIT.
    pub fn new_interleaved(mem_block: Rc<RefCell<VirtualMem>>, keep_comments: bool) -> Self {
        // Pick the code page size
        let system_page_size = mem_block.borrow().system_page_size();
        let page_size = if 0 == Self::PREFERRED_CODE_PAGE_SIZE % system_page_size {
            Self::PREFERRED_CODE_PAGE_SIZE
        } else {
            system_page_size
        };

        let mut cb = Self::new(mem_block, keep_comments);
        assert_eq!(0, cb.mem_size % page_size, "partially in-bounds code pages should be impossible");
        cb.page_size = page_size;
        cb.write_pos = 0;
        cb.other_write_pos = page_size / 2;
        cb.page_end_reserve = cb.jmp_ptr_bytes();
        cb
    }

    /// True when inline and outlined code are interleaved per code page.
    pub fn is_interleaved(&self) -> bool {
        self.page_size != 0
    }

    /// Size of a code page, or 0 if not interleaved.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Whether writes currently go to the outlined half.
    pub fn is_outlined(&self) -> bool {
        self.outlined
    }

    /// Direct the next writes at the outlined (cold) half of each code page, or back at the
    /// inline half. Returns the previous setting. No-op when not interleaved.
    pub fn set_outlined(&mut self, outlined: bool) -> bool {
        let was_outlined = self.outlined;
        if !self.is_interleaved() || outlined == was_outlined {
            return was_outlined;
        }
        mem::swap(&mut self.write_pos, &mut self.other_write_pos);
        self.outlined = outlined;
        was_outlined
    }

    /// The write position of the given half, whether or not it is the current one.
    fn write_pos_for(&self, outlined: bool) -> usize {
        if outlined == self.outlined { self.write_pos } else { self.other_write_pos }
    }

    /// A pointer to the inline half's write position.
    pub fn inlined_write_ptr(&self) -> CodePtr {
        self.get_ptr(self.write_pos_for(false))
    }

    /// A pointer to the outlined half's write position.
    pub fn outlined_write_ptr(&self) -> CodePtr {
        self.get_ptr(self.write_pos_for(true))
    }

    /// Offset within a code page where the given half starts
    fn half_start(&self, outlined: bool) -> usize {
        if outlined { self.page_size / 2 } else { 0 }
    }

    /// Offset within each page where the current half should stop writing (exclusive)
    pub fn page_end(&self) -> usize {
        let page_end = if self.outlined { self.page_size } else { self.page_size / 2 };
        page_end - self.page_end_reserve // reserve space to jump to the next page
    }

    /// Position where the half page that `pos` writes to starts. The end of the
    /// outlined half of a page is the start of the next page, so it's not just
    /// `pos % page_size`.
    fn half_page_start(&self, pos: usize, outlined: bool) -> usize {
        let page_idx = if outlined { pos.saturating_sub(self.page_size / 2) / self.page_size } else { pos / self.page_size };
        page_idx * self.page_size + self.half_start(outlined)
    }

    /// Bytes written to the current page of the given half at `pos`.
    fn current_page_bytes(&self, pos: usize, outlined: bool) -> usize {
        pos - self.half_page_start(pos, outlined)
    }

    /// Bytes of machine code written into the inline (hot) half of code pages.
    pub fn inlined_code_size(&self) -> usize {
        if !self.is_interleaved() {
            return self.write_pos;
        }
        self.past_page_bytes[0] + self.current_page_bytes(self.write_pos_for(false), false)
    }

    /// Bytes of machine code written into the outlined (cold) half of code pages.
    pub fn outlined_code_size(&self) -> usize {
        if !self.is_interleaved() {
            return 0;
        }
        self.past_page_bytes[1] + self.current_page_bytes(self.write_pos_for(true), true)
    }

    /// Check if this code block has sufficient remaining capacity in the current half page
    pub fn has_capacity(&self, num_bytes: usize) -> bool {
        if !self.is_interleaved() || !self.page_bounds_check {
            return self.write_pos + num_bytes <= self.mem_size;
        }
        let page_offset = self.write_pos - self.half_page_start(self.write_pos, self.outlined) + self.half_start(self.outlined);
        let capacity = self.page_end().saturating_sub(page_offset);
        num_bytes <= capacity
    }

    /// Call a given function without the page bounds check. For patching bytes that
    /// have been already reserved by a bounds-checked write.
    pub fn without_page_bounds_check<R>(&mut self, block: impl FnOnce(&mut Self) -> R) -> R {
        let old_page_bounds_check = self.page_bounds_check;
        self.page_bounds_check = false;
        let ret = block(self);
        self.page_bounds_check = old_page_bounds_check;
        ret
    }

    /// Move the current half to the next code page: rewind to `base_ptr`, write a jump
    /// from there to the start of the same half on the next page, and continue writing
    /// there. The other half is moved to the same page too, unless it is already past it.
    /// This mirrors YJIT's `CodeBlock::next_page`. Returns false if there are no more
    /// pages or the jump could not be written.
    #[must_use]
    pub fn next_page(&mut self, base_ptr: CodePtr, jmp_ptr: impl Fn(&mut CodeBlock, CodePtr)) -> bool {
        if !self.is_interleaved() {
            return false;
        }
        let old_write_pos = self.write_pos;
        self.set_write_ptr(base_ptr);

        // If we're already at the start of a fresh half page, moving to another fresh one
        // doesn't help. Whatever failed doesn't fit in half a page, or memory can't be mapped.
        if self.write_pos == self.half_page_start(self.write_pos, self.outlined) {
            self.write_pos = old_write_pos;
            return false;
        }

        // Move self to the next page
        let next_page_idx = self.half_page_start(self.write_pos, self.outlined) / self.page_size + 1;
        if !self.set_page(next_page_idx, &jmp_ptr) {
            self.write_pos = old_write_pos; // rollback if there are no more pages
            return false;
        }

        // Move the other half to the same page if it's not ahead of it. Unlike YJIT's
        // other_cb().set_page(), we don't write a jump in the other half: ZJIT never
        // leaves the other half in the middle of a code sequence that falls through.
        let other_outlined = !self.outlined;
        let other_dst_pos = self.page_size * next_page_idx + self.half_start(other_outlined);
        if self.other_write_pos < other_dst_pos {
            self.past_page_bytes[other_outlined as usize] += self.current_page_bytes(self.other_write_pos, other_outlined);
            self.other_write_pos = other_dst_pos;
        }

        !self.dropped_bytes
    }

    /// Move the current half to page_idx only if it's not going backwards.
    fn set_page(&mut self, page_idx: usize, jmp_ptr: &impl Fn(&mut CodeBlock, CodePtr)) -> bool {
        let dst_pos = self.page_size * page_idx + self.half_start(self.outlined);
        if self.write_pos < dst_pos {
            // Fail if next page is out of bounds
            if dst_pos >= self.mem_size {
                return false;
            }

            // Reset dropped_bytes
            self.dropped_bytes = false;

            // Generate jmp_ptr from src_pos to dst_pos
            let dst_ptr = self.get_ptr(dst_pos);
            let half_page_start = self.half_page_start(self.write_pos, self.outlined);
            let old_page_end_reserve = self.page_end_reserve;
            self.page_end_reserve = 0;
            assert!(self.has_capacity(self.jmp_ptr_bytes()));
            self.add_comment("jump to next page");
            jmp_ptr(self, dst_ptr);
            self.page_end_reserve = old_page_end_reserve;
            if self.dropped_bytes {
                return false;
            }
            self.page_jump_ends.insert(half_page_start, self.write_pos);

            // Update past_page_bytes for code size stats
            self.past_page_bytes[self.outlined as usize] += self.current_page_bytes(self.write_pos, self.outlined);

            // Start the next code from dst_pos
            self.write_pos = dst_pos;
        }
        !self.dropped_bytes
    }

    /// Split a code range written by a single half into the pieces that are within a
    /// half page, skipping the other half of pages and the unused tail of half pages
    /// left with a jump to the next page. Returns `[(start, end)]` of non-empty pieces.
    pub fn code_ranges(&self, start: CodePtr, end: CodePtr) -> Vec<(CodePtr, CodePtr)> {
        let base = self.get_ptr(0).as_offset();
        let start_pos = (start.as_offset() - base) as usize;
        let end_pos = (end.as_offset() - base) as usize;
        if start_pos >= end_pos {
            return vec![];
        }
        if !self.is_interleaved() {
            return vec![(start, end)];
        }

        let mut ranges = vec![];
        let mut pos = start_pos;
        while pos < end_pos {
            let page_base = pos / self.page_size * self.page_size;
            let half_start = if pos % self.page_size < self.page_size / 2 { 0 } else { self.page_size / 2 };
            let half_end = page_base + half_start + self.page_size / 2;
            if end_pos <= half_end {
                ranges.push((self.get_ptr(pos), end));
                break;
            }
            let piece_end = self.page_jump_ends.get(&(page_base + half_start)).copied().unwrap_or(half_end);
            if pos < piece_end {
                ranges.push((self.get_ptr(pos), self.get_ptr(piece_end)));
            }
            pos = page_base + self.page_size + half_start;
        }
        ranges
    }

    /// Number of bytes of code in the ranges returned by [`Self::code_ranges`].
    pub fn code_size_between(&self, start: CodePtr, end: CodePtr) -> usize {
        self.code_ranges(start, end).iter().map(|(s, e)| (e.as_offset() - s.as_offset()) as usize).sum()
    }

    /// Size of the region in bytes that we have allocated physical memory for.
    pub fn mapped_region_size(&self) -> usize {
        self.mem_block.borrow().mapped_region_size()
    }

    /// Size of the region in bytes where writes could be attempted.
    pub fn virtual_region_size(&self) -> usize {
        self.mem_size
    }

    /// Add an assembly comment if the feature is on.
    pub fn add_comment(&mut self, comment: &str) {
        if !self.keep_comments {
            return;
        }

        let cur_ptr = self.get_write_ptr().raw_addr(self);

        // If there's no current list of comments for this line number, add one.
        let this_line_comments = self.asm_comments.entry(cur_ptr).or_default();

        // Unless this comment is the same as the last one at this same line, add it.
        if this_line_comments.last().map(String::as_str) != Some(comment) {
            this_line_comments.push(comment.to_string());
        }
    }

    pub fn comments_at(&self, pos: usize) -> Option<&Vec<String>> {
        self.asm_comments.get(&pos)
    }

    pub fn get_write_pos(&self) -> usize {
        self.write_pos
    }

    pub fn write_mem(&self, write_ptr: CodePtr, byte: u8) -> Result<(), WriteError> {
        self.mem_block.borrow_mut().write_byte(write_ptr, byte)
    }

    /// Get a (possibly dangling) direct pointer to the current write position
    pub fn get_write_ptr(&self) -> CodePtr {
        self.get_ptr(self.write_pos)
    }

    /// Set the current write position from a pointer
    pub fn set_write_ptr(&mut self, code_ptr: CodePtr) {
        let pos = code_ptr.as_offset() - self.mem_block.borrow().start_ptr().as_offset();
        self.write_pos = pos.try_into().unwrap();
    }

    /// Invoke a callback with write_ptr temporarily adjusted to a given address
    pub fn with_write_ptr(&mut self, code_ptr: CodePtr, callback: impl Fn(&mut CodeBlock)) -> Range<CodePtr> {
        // Temporarily update the write_pos. Ignore the dropped_bytes flag at the old address.
        let old_write_pos = self.write_pos;
        let old_dropped_bytes = self.dropped_bytes;
        self.set_write_ptr(code_ptr);
        self.dropped_bytes = false;

        // Invoke the callback. Patching rewrites bytes that are already written,
        // so it's not bounded by the current half page.
        self.without_page_bounds_check(|cb| callback(cb));

        // Build a code range modified by the callback
        let ret = code_ptr..self.get_write_ptr();

        // Restore the original write_pos and dropped_bytes flag.
        self.dropped_bytes = old_dropped_bytes;
        self.write_pos = old_write_pos;
        ret
    }

    /// Get a (possibly dangling) direct pointer into the executable memory block
    pub fn get_ptr(&self, offset: usize) -> CodePtr {
        self.mem_block.borrow().start_ptr().add_bytes(offset)
    }

    /// Write a single byte at the current position.
    pub fn write_byte(&mut self, byte: u8) {
        let write_ptr = self.get_write_ptr();
        if self.has_capacity(1) && self.mem_block.borrow_mut().write_byte(write_ptr, byte).is_ok() {
            self.write_pos += 1;
        } else {
            self.dropped_bytes = true;
        }
    }

    /// Write multiple bytes starting from the current position.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.write_byte(*byte);
        }
    }

    /// Write an integer over the given number of bits at the current position.
    pub fn write_int(&mut self, val: u64, num_bits: u32) {
        assert!(num_bits > 0);
        assert!(num_bits % 8 == 0);

        // Switch on the number of bits
        match num_bits {
            8 => self.write_byte(val as u8),
            16 => self.write_bytes(&[(val & 0xff) as u8, ((val >> 8) & 0xff) as u8]),
            32 => self.write_bytes(&[
                (val & 0xff) as u8,
                ((val >> 8) & 0xff) as u8,
                ((val >> 16) & 0xff) as u8,
                ((val >> 24) & 0xff) as u8,
            ]),
            _ => {
                let mut cur = val;

                // Write out the bytes
                for _byte in 0..(num_bits / 8) {
                    self.write_byte((cur & 0xff) as u8);
                    cur >>= 8;
                }
            }
        }
    }

    /// Check if bytes have been dropped (unwritten because of insufficient space)
    pub fn has_dropped_bytes(&self) -> bool {
        self.dropped_bytes
    }

    /// Set dropped_bytes to false if the current zjit_alloc_bytes() + code_region_size
    /// + page_size is below --zjit-mem-size.
    pub fn update_dropped_bytes(&mut self) {
        if self.mem_block.borrow().can_allocate() {
            self.dropped_bytes = false;

            // Memory is available again, so let the interpreter resume
            // triggering compilation.
            unsafe { rb_zjit_compiling_p = true; }
        }
    }

    /// Allocate a new label with a given name
    pub fn new_label(&mut self, name: String) -> Label {
        assert!(!name.contains(' '), "use underscores in label names, not spaces");

        // This label doesn't have an address yet
        self.label_addrs.push(0);
        self.label_names.push(name);

        Label(self.label_addrs.len() - 1)
    }

    /// Write a label at the current address
    pub fn write_label(&mut self, label: Label) {
        self.label_addrs[label.0] = self.write_pos;
    }

    // Add a label reference at the current write position
    pub fn label_ref(&mut self, label: Label, num_bytes: usize, encode: impl Fn(&mut CodeBlock, i64, i64) -> Result<(), ()> + 'static) {
        assert!(label.0 < self.label_addrs.len());

        // Keep track of the reference
        self.label_refs.push(LabelRef { pos: self.write_pos, label, num_bytes, encode: Box::new(encode) });

        // Move past however many bytes the instruction takes up.
        if self.has_capacity(num_bytes) {
            // Reserve the bytes by writing them rather than by moving the cursor
            // over them. Pages are mapped on first write, so a cursor bump alone
            // leaves a page that cannot be mapped.
            const RESERVED: [u8; 16] = [0; 16];
            assert!(num_bytes <= RESERVED.len(), "label reference wants {num_bytes} bytes");
            self.write_bytes(&RESERVED[..num_bytes]);
        } else {
            self.dropped_bytes = true; // retry emitting the Insn after next_page
        }
    }

    /// Number of label references recorded so far. Used to roll back an Insn to retry it on the next page.
    pub fn label_refs_len(&self) -> usize {
        self.label_refs.len()
    }

    /// Drop label references recorded after `len`. Used to roll back an Insn to retry it on the next page.
    pub fn truncate_label_refs(&mut self, len: usize) {
        self.label_refs.truncate(len);
    }

    // Link internal label references
    pub fn link_labels(&mut self) -> Result<(), ()> {
        // Label references rewrite bytes reserved by label_ref(), which may be in either half.
        self.without_page_bounds_check(|cb| cb.link_labels_inner())
    }

    fn link_labels_inner(&mut self) -> Result<(), ()> {
        let orig_pos = self.write_pos;
        let mut link_result = Ok(());

        // For each label reference
        for label_ref in mem::take(&mut self.label_refs) {
            let ref_pos = label_ref.pos;
            let label_idx = label_ref.label.0;
            assert!(ref_pos < self.mem_size);

            let label_addr = self.label_addrs[label_idx];
            assert!(label_addr < self.mem_size);

            self.write_pos = ref_pos;
            let encode_result = (label_ref.encode.as_ref())(self, (ref_pos + label_ref.num_bytes) as i64, label_addr as i64);
            link_result = link_result.and(encode_result);

            // Verify number of bytes written when the callback returns Ok
            if encode_result.is_ok() {
                assert_eq!(self.write_pos, ref_pos + label_ref.num_bytes, "label_ref \
                    callback didn't write number of bytes it claimed to write upfront");
            }
        }

        self.write_pos = orig_pos;

        // Clear the label positions and references
        self.label_addrs.clear();
        self.label_names.clear();
        assert!(self.label_refs.is_empty());

        link_result
    }

    /// Convert a Label to CodePtr
    pub fn resolve_label(&self, label: Label) -> CodePtr {
        self.get_ptr(self.label_addrs[label.0])
    }

    pub fn clear_labels(&mut self) {
        self.label_addrs.clear();
        self.label_names.clear();
        self.label_refs.clear();
    }

    /// Make all the code in the region executable. Call this at the end of a write session.
    pub fn mark_all_writable(&mut self) {
        self.mem_block.borrow_mut().mark_all_writable();
    }

    pub fn mark_all_executable(&mut self) {
        self.mem_block.borrow_mut().mark_all_executable();
    }

    /// Call a func with the disasm of generated code for testing
    #[allow(unused_variables)]
    #[cfg(all(test, feature = "disasm"))]
    pub fn disasm(&self) -> String {
        let start_addr = self.get_ptr(0).raw_addr(self);
        let end_addr = self.get_write_ptr().raw_addr(self);
        crate::disasm::disasm_addr_range(self, start_addr, end_addr)
    }

    /// Read a written byte for testing
    #[cfg(test)]
    pub fn hexdump_byte_at(&self, pos: usize) -> [u8; 1] {
        let mem_block = &*self.mem_block.borrow();
        [unsafe { mem_block.start_ptr().raw_ptr(mem_block).add(pos).read() }]
    }

    /// Return the hex dump of generated code for testing
    #[cfg(test)]
    pub fn hexdump(&self) -> String {
        format!("{:x}", self)
    }
}

/// Run assert_snapshot! only if cfg!(feature = "disasm").
/// $actual can be not only `cb.disasm()` but also `disasms!(cb1, cb2, ...)`.
#[cfg(test)]
#[macro_export]
macro_rules! assert_disasm_snapshot {
    ($actual: expr, @$($tt: tt)*) => {{
        #[cfg(feature = "disasm")]
        assert_snapshot!($actual, @$($tt)*)
    }};
}

/// Combine multiple cb.disasm() results to match all of them at once, which allows
/// us to avoid running the set of zjit-test -> zjit-test-update multiple times.
#[cfg(all(test, feature = "disasm"))]
#[macro_export]
macro_rules! disasms {
    ($( $cb:expr ),+ $(,)?) => {{
        crate::disasms_with!("", $( $cb ),+)
    }};
}

/// Basically `disasms!` but allows a non-"" delimiter, such as "\n"
#[cfg(all(test, feature = "disasm"))]
#[macro_export]
macro_rules! disasms_with {
    ($join:expr, $( $cb:expr ),+ $(,)?) => {{
        vec![$( $cb.disasm() ),+].join($join)
    }};
}

/// Combine multiple cb.hexdump() results to match all of them at once, which allows
/// us to avoid running the set of zjit-test -> zjit-test-update multiple times.
#[cfg(test)]
#[macro_export]
macro_rules! hexdumps {
    ($( $cb:expr ),+ $(,)?) => {{
        vec![$( $cb.hexdump() ),+].join("\n")
    }};
}

/// Produce hex string output from the bytes in a code block
impl fmt::LowerHex for CodeBlock {
    fn fmt(&self, fmtr: &mut fmt::Formatter) -> fmt::Result {
        for pos in 0..self.write_pos {
            let mem_block = &*self.mem_block.borrow();
            let byte = unsafe { mem_block.start_ptr().raw_ptr(mem_block).add(pos).read() };
            fmtr.write_fmt(format_args!("{byte:02x}"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
impl CodeBlock {
    /// Stubbed CodeBlock for testing. Can't execute generated code.
    pub fn new_dummy() -> Self {
        const DEFAULT_MEM_SIZE: usize = 1024 * 1024;
        CodeBlock::new_dummy_sized(DEFAULT_MEM_SIZE)
    }

    pub fn new_dummy_sized(mem_size: usize) -> Self {
        use crate::virtualmem::*;
        let virt_mem = VirtualMem::alloc(mem_size, None);
        Self::new(Rc::new(RefCell::new(virt_mem)), false)
    }

    /// Stubbed CodeBlock that interleaves inline and outlined code per code page.
    pub fn new_dummy_interleaved() -> Self {
        use crate::virtualmem::*;
        const DEFAULT_MEM_SIZE: usize = 1024 * 1024;
        let virt_mem = VirtualMem::alloc(DEFAULT_MEM_SIZE, None);
        Self::new_interleaved(Rc::new(RefCell::new(virt_mem)), false)
    }
}

impl crate::virtualmem::CodePtrBase for CodeBlock {
    fn base_ptr(&self) -> std::ptr::NonNull<u8> {
        self.mem_block.borrow().base_ptr()
    }
}

/// Compute the number of bits needed to encode a signed value
pub fn imm_num_bits(imm: i64) -> u8
{
    // Compute the smallest size this immediate fits in
    if imm >= i8::MIN.into() && imm <= i8::MAX.into() {
        return 8;
    }
    if imm >= i16::MIN.into() && imm <= i16::MAX.into() {
        return 16;
    }
    if imm >= i32::MIN.into() && imm <= i32::MAX.into() {
        return 32;
    }

    64
}

/// Compute the number of bits needed to encode an unsigned value
pub fn uimm_num_bits(uimm: u64) -> u8
{
    // Compute the smallest size this immediate fits in
    if uimm <= u8::MAX.into() {
        return 8;
    }
    else if uimm <= u16::MAX.into() {
        return 16;
    }
    else if uimm <= u32::MAX.into() {
        return 32;
    }

    64
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_imm_num_bits()
    {
        assert_eq!(imm_num_bits(i8::MIN.into()), 8);
        assert_eq!(imm_num_bits(i8::MAX.into()), 8);

        assert_eq!(imm_num_bits(i16::MIN.into()), 16);
        assert_eq!(imm_num_bits(i16::MAX.into()), 16);

        assert_eq!(imm_num_bits(i32::MIN.into()), 32);
        assert_eq!(imm_num_bits(i32::MAX.into()), 32);

        assert_eq!(imm_num_bits(i64::MIN), 64);
        assert_eq!(imm_num_bits(i64::MAX), 64);
    }

    /// Write a fake jump for next_page(): jmp_ptr_bytes() of 0xEE
    fn fake_jmp(cb: &mut CodeBlock, _dst: CodePtr) {
        for _ in 0..cb.jmp_ptr_bytes() {
            cb.write_byte(0xEE);
        }
    }

    #[test]
    fn test_set_outlined_keeps_a_write_position_per_half() {
        let mut cb = CodeBlock::new_dummy_interleaved();
        assert!(cb.is_interleaved());
        let half = cb.page_size() / 2;
        assert_eq!(half as i64, cb.outlined_write_ptr().as_offset());

        cb.write_byte(1);
        assert_eq!(1, cb.inlined_code_size());
        assert_eq!(0, cb.outlined_code_size());

        let was_outlined = cb.set_outlined(true);
        assert!(!was_outlined);
        assert!(cb.is_outlined());
        cb.write_bytes(&[2, 3]);
        assert_eq!(1, cb.inlined_code_size());
        assert_eq!(2, cb.outlined_code_size());
        assert_eq!(half as i64 + 2, cb.outlined_write_ptr().as_offset());

        cb.set_outlined(was_outlined);
        assert_eq!(1, cb.get_write_pos());
        cb.write_byte(4);
        assert_eq!(2, cb.inlined_code_size());
        assert_eq!(2, cb.outlined_code_size());
    }

    #[test]
    fn test_non_interleaved_code_block_ignores_set_outlined() {
        let mut cb = CodeBlock::new_dummy();
        assert!(!cb.is_interleaved());
        assert!(!cb.set_outlined(true));
        assert!(!cb.is_outlined());
        cb.write_byte(1);
        assert_eq!(1, cb.inlined_code_size());
        assert_eq!(0, cb.outlined_code_size());
        assert!(!cb.next_page(cb.get_write_ptr(), fake_jmp));
    }

    #[test]
    fn test_writes_are_bounded_by_the_half_page() {
        let mut cb = CodeBlock::new_dummy_interleaved();
        let page_end = cb.page_end();
        assert_eq!(cb.page_size() / 2 - cb.jmp_ptr_bytes(), page_end);
        for _ in 0..page_end {
            cb.write_byte(0x90);
        }
        assert!(!cb.has_dropped_bytes());
        assert!(!cb.has_capacity(1));

        // The next byte doesn't fit, so it's dropped rather than written into the reserve
        cb.write_byte(0x90);
        assert!(cb.has_dropped_bytes());
        assert_eq!(page_end, cb.get_write_pos());
    }

    #[test]
    fn test_next_page_moves_both_halves() {
        let mut cb = CodeBlock::new_dummy_interleaved();
        let page_size = cb.page_size();
        let half = page_size / 2;

        // Fill the inline half and fail to write an instruction that straddles its end
        for _ in 0..(cb.page_end() - 2) {
            cb.write_byte(0x90);
        }
        let src_ptr = cb.get_write_ptr();
        cb.write_bytes(&[1, 2, 3, 4]);
        assert!(cb.has_dropped_bytes());

        // Jump to the next page and retry
        assert!(cb.next_page(src_ptr, fake_jmp));
        assert!(!cb.has_dropped_bytes());
        assert_eq!(page_size, cb.get_write_pos());
        // The other half moves to the same page too
        assert_eq!((page_size + half) as i64, cb.outlined_write_ptr().as_offset());
        cb.write_bytes(&[1, 2, 3, 4]);
        assert!(!cb.has_dropped_bytes());

        // The jump was written where the instruction was going to start
        let jmp_end = cb.page_end() - 2 + cb.jmp_ptr_bytes();
        assert_eq!(&[0xEE], &cb.hexdump_byte_at(cb.page_end() - 2)[..]);
        assert!(jmp_end <= half, "the jump should fit in the reserved space");
        assert_eq!(jmp_end + 4, cb.inlined_code_size());

        // The range spanning the two pages skips the outlined half in between
        let ranges = cb.code_ranges(cb.get_ptr(0), cb.get_write_ptr());
        assert_eq!(vec![
            (cb.get_ptr(0), cb.get_ptr(jmp_end)),
            (cb.get_ptr(page_size), cb.get_ptr(page_size + 4)),
        ], ranges);
        assert_eq!(cb.inlined_code_size(), cb.code_size_between(cb.get_ptr(0), cb.get_write_ptr()));

        // Moving the outlined half from page 1 doesn't move the inline half back
        cb.set_outlined(true);
        for _ in 0..(cb.page_end() - half) {
            cb.write_byte(0x90);
        }
        let src_ptr = cb.get_write_ptr();
        cb.write_byte(0x90);
        assert!(cb.has_dropped_bytes());
        assert!(cb.next_page(src_ptr, fake_jmp));
        assert_eq!((2 * page_size + half) as i64, cb.get_write_ptr().as_offset());
        // The jump filled the outlined half of page 1 up to the end of the page
        assert_eq!(half, cb.outlined_code_size());
        cb.set_outlined(false);
        assert_eq!(2 * page_size, cb.get_write_pos());
    }

    #[test]
    fn test_next_page_at_the_start_of_a_page_fails() {
        // An instruction that doesn't fit in a fresh half page would never fit
        let mut cb = CodeBlock::new_dummy_interleaved();
        assert!(!cb.next_page(cb.get_write_ptr(), fake_jmp));
        assert_eq!(0, cb.get_write_pos());
    }

    #[test]
    fn test_next_page_fails_on_the_last_page() {
        let mut cb = CodeBlock::new_dummy_interleaved();
        let page_size = cb.page_size();
        let last_page_pos = cb.virtual_region_size() - page_size;
        cb.set_write_ptr(cb.get_ptr(last_page_pos + 8));
        let src_ptr = cb.get_write_ptr();
        assert!(!cb.next_page(src_ptr, fake_jmp));
        assert_eq!(last_page_pos + 8, cb.get_write_pos());
    }

    #[test]
    fn test_uimm_num_bits() {
        assert_eq!(uimm_num_bits(u8::MIN.into()), 8);
        assert_eq!(uimm_num_bits(u8::MAX.into()), 8);

        assert_eq!(uimm_num_bits(((u8::MAX as u16) + 1).into()), 16);
        assert_eq!(uimm_num_bits(u16::MAX.into()), 16);

        assert_eq!(uimm_num_bits(((u16::MAX as u32) + 1).into()), 32);
        assert_eq!(uimm_num_bits(u32::MAX.into()), 32);

        assert_eq!(uimm_num_bits((u32::MAX as u64) + 1), 64);
        assert_eq!(uimm_num_bits(u64::MAX), 64);
    }

    #[test]
    fn test_label_ref_at_an_unmappable_page_sets_dropped_bytes() {
        // Two pages of address space, but a memory limit that only lets
        // the first page be mapped.
        let page_size = unsafe { crate::cruby::rb_jit_get_page_size() } as usize;
        let limit = crate::stats::zjit_alloc_bytes() + page_size + page_size / 2;
        let virt_mem = VirtualMem::alloc(2 * page_size, Some(limit));
        let mut cb = CodeBlock::new(Rc::new(RefCell::new(virt_mem)), false);

        // Fill the mappable page up to 2 bytes below its end, so that
        // a 5-byte jump reservation straddles into the unmappable page.
        for _ in 0..(page_size - 2) {
            cb.write_byte(0x90);
        }
        assert!(!cb.has_dropped_bytes(), "the first page must map");

        // label_ref() must reserve its bytes by writing them, so that the
        // unmappable page surfaces as dropped_bytes here, where the caller
        // still reads it and turns it into CompileError::OutOfMemory.
        let label = cb.new_label("over_the_page_boundary".to_string());
        cb.label_ref(label, 5, |cb, _, _| {
            cb.write_bytes(&[0; 5]);
            Ok(())
        });
        cb.write_label(label);

        // Mirror what compile_with_regs does after emit. It calls link_labels()
        // only if emit did not go OOM. This reproduces an assertion failure in
        // link_labels() when dropped_bytes isn't updated properly.
        if !cb.has_dropped_bytes() {
            cb.link_labels().unwrap();
        }
        assert!(cb.has_dropped_bytes(), "the reservation must discover the unmappable page");
    }
}
