//! Model for creating generating textual assembler code.

use std::collections::BTreeMap;
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

    /// Offset where the outlined (cold) half of the region begins, or `mem_size`
    /// when the region is not split. See [`crate::virtualmem::VirtualMemory`].
    outlined_start: usize,

    // Current writing position, in whichever half `outlined` names
    write_pos: usize,

    /// Write position of the half that is *not* current. [`Self::set_outlined`]
    /// swaps it with `write_pos`.
    other_write_pos: usize,

    /// Whether `write_pos` currently points into the outlined half.
    outlined: bool,

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
        let (mem_size, outlined_start) = {
            let mem_block = mem_block.borrow();
            (mem_block.virtual_region_size(), mem_block.outlined_start_bytes())
        };
        Self {
            mem_block,
            mem_size,
            outlined_start,
            write_pos: 0,
            other_write_pos: outlined_start,
            outlined: false,
            label_addrs: Vec::new(),
            label_names: Vec::new(),
            label_refs: Vec::new(),
            keep_comments,
            asm_comments: BTreeMap::new(),
            dropped_bytes: false,
        }
    }

    /// Size of the region in bytes that we have allocated physical memory for.
    pub fn mapped_region_size(&self) -> usize {
        self.mem_block.borrow().mapped_region_size()
    }

    /// Physical memory backing the inlined (hot) half of the region.
    pub fn inlined_mapped_size(&self) -> usize {
        self.mem_block.borrow().inlined_mapped_size()
    }

    /// Physical memory backing the outlined (cold) half of the region.
    pub fn outlined_mapped_size(&self) -> usize {
        self.mem_block.borrow().outlined_mapped_size()
    }

    /// Bytes of machine code written into the inlined (hot) half.
    pub fn inlined_code_size(&self) -> usize {
        self.write_pos_for(false)
    }

    /// Bytes of machine code written into the outlined (cold) half.
    pub fn outlined_code_size(&self) -> usize {
        self.write_pos_for(true) - self.outlined_start
    }

    /// Size of the region in bytes where writes could be attempted.
    pub fn virtual_region_size(&self) -> usize {
        self.mem_size
    }

    /// True when the region has an outlined half to emit cold code into.
    pub fn has_outlined_region(&self) -> bool {
        self.outlined_start < self.mem_size
    }

    /// Whether writes currently go to the outlined half.
    pub fn is_outlined(&self) -> bool {
        self.outlined
    }

    /// Direct the next writes at the outlined (cold) half of the region, or back at
    /// the inlined half. Each half keeps its own write position, so switching back
    /// and forth appends to whichever one is named.
    ///
    /// Returns the previous setting, so a caller that borrowed the outlined half can
    /// put it back. On a region with no outlined half this is a no-op that always
    /// reports `false`, which is what keeps the unit tests on the old single-arena
    /// layout.
    pub fn set_outlined(&mut self, outlined: bool) -> bool {
        let was_outlined = self.outlined;
        if !self.has_outlined_region() || outlined == was_outlined {
            return was_outlined;
        }
        std::mem::swap(&mut self.write_pos, &mut self.other_write_pos);
        self.outlined = outlined;
        was_outlined
    }

    /// The write position of the given half, whether or not it is the current one.
    fn write_pos_for(&self, outlined: bool) -> usize {
        if outlined == self.outlined { self.write_pos } else { self.other_write_pos }
    }

    /// A pointer to the outlined half's write position, wherever emission is
    /// pointed right now. Used to report the range a compile appended there.
    pub fn outlined_write_ptr(&self) -> CodePtr {
        self.get_ptr(self.write_pos_for(true))
    }

    /// One past the last offset the current half accepts writes at. Hot code stops
    /// at the start of the outlined half rather than running into it.
    fn write_limit(&self) -> usize {
        if self.outlined { self.mem_size } else { self.outlined_start }
    }

    /// Whether an offset falls in the outlined half. Used to aim the write limit at
    /// the half a patch site is in, which is not always the half emission is on.
    fn is_outlined_pos(&self, pos: usize) -> bool {
        pos >= self.outlined_start
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
        // The callback overwrites existing code in place, so it must not be one that
        // switches halves: a `set_outlined` in the middle would carry `code_ptr` off
        // into the other half's saved position and hand back an empty written range,
        // which remove_gc_offsets reads.
        // Everything that patches (invalidation jumps, call-site regeneration) emits
        // a single jump or call and has no side exits, so there is nothing to switch.
        debug_assert!(!self.outlined, "with_write_ptr patches the half it is already on");

        // Temporarily update the write_pos. Ignore the dropped_bytes flag at the old address.
        let old_write_pos = self.write_pos;
        let old_dropped_bytes = self.dropped_bytes;
        self.set_write_ptr(code_ptr);
        self.dropped_bytes = false;

        // Invoke the callback
        callback(self);

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
        // Stop at the end of the half being written rather than running into the
        // other one. [`VirtualMem`] bounds-checks the whole region, which is no
        // longer the same thing: a write one byte past the inlined half is a
        // perfectly good address in the outlined half, and would overwrite cold
        // code that the outlined write position still thinks it owns. Dropping the
        // byte instead reports the half as full the way running out of region used
        // to, and `arch_emit` turns that into CompileError::OutOfMemory.
        if self.write_pos >= self.write_limit() {
            self.dropped_bytes = true;
            return;
        }

        let write_ptr = self.get_write_ptr();
        // TODO: check has_capacity()
        if self.mem_block.borrow_mut().write_byte(write_ptr, byte).is_ok() {
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

        // Move past however many bytes the instruction takes up
        if self.write_pos + num_bytes < self.write_limit() {
            // Reserve the bytes by writing them rather than by moving the cursor
            // over them. Pages are mapped on first write, so a cursor bump alone
            // leaves a page that cannot be mapped -- the memory limit is
            // reached, say -- to be discovered by link_labels(), which comes
            // back to fill these in long after `dropped_bytes` has stopped being
            // read and can only report a short write by tripping its own
            // assertion. Writing here keeps the failure on the path that already
            // handles it: `arch_emit` turns `dropped_bytes` into
            // CompileError::OutOfMemory and the function is retried or dropped.
            const RESERVED: [u8; 16] = [0; 16];
            assert!(num_bytes <= RESERVED.len(), "label reference wants {num_bytes} bytes");
            self.write_bytes(&RESERVED[..num_bytes]);
        } else {
            self.dropped_bytes = true; // retry emitting the Insn after next_page
        }
    }

    // Link internal label references
    pub fn link_labels(&mut self) -> Result<(), ()> {
        let orig_pos = self.write_pos;
        let orig_outlined = self.outlined;
        let mut link_result = Ok(());

        // For each label reference
        for label_ref in mem::take(&mut self.label_refs) {
            let ref_pos = label_ref.pos;
            let label_idx = label_ref.label.0;
            assert!(ref_pos < self.mem_size);

            let label_addr = self.label_addrs[label_idx];
            assert!(label_addr < self.mem_size);

            self.write_pos = ref_pos;
            // A reference from cold code is filled in after emission has been put
            // back on the hot half, so name the half the reference is really in:
            // otherwise its writes look like hot code running past its end.
            self.outlined = self.is_outlined_pos(ref_pos);
            let encode_result = (label_ref.encode.as_ref())(self, (ref_pos + label_ref.num_bytes) as i64, label_addr as i64);
            link_result = link_result.and(encode_result);

            // Verify number of bytes written when the callback returns Ok
            if encode_result.is_ok() {
                assert_eq!(self.write_pos, ref_pos + label_ref.num_bytes, "label_ref \
                    callback didn't write number of bytes it claimed to write upfront");
            }
        }

        self.write_pos = orig_pos;
        self.outlined = orig_outlined;

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

    /// Stubbed CodeBlock whose region is split into an inlined and an outlined
    /// half, like the one ZJIT runs with. [`Self::new_dummy`] deliberately is not,
    /// so that the backend's disassembly snapshots keep seeing one contiguous
    /// stream.
    pub fn new_dummy_split() -> Self {
        use crate::virtualmem::*;
        const DEFAULT_MEM_SIZE: usize = 1024 * 1024;
        let virt_mem = VirtualMem::alloc_split(DEFAULT_MEM_SIZE, None);
        Self::new(Rc::new(RefCell::new(virt_mem)), false)
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

    #[test]
    fn test_set_outlined_keeps_a_write_position_per_half() {
        let mut cb = CodeBlock::new_dummy_split();
        assert!(cb.has_outlined_region());
        let outlined_start = cb.outlined_write_ptr().as_offset();
        assert!(outlined_start > 0);

        cb.write_byte(1);
        assert_eq!(1, cb.inlined_code_size());
        assert_eq!(0, cb.outlined_code_size());

        // Switching halves parks the inlined write position rather than losing it.
        let was_outlined = cb.set_outlined(true);
        assert!(!was_outlined);
        assert!(cb.is_outlined());
        cb.write_bytes(&[2, 3]);
        assert_eq!(1, cb.inlined_code_size());
        assert_eq!(2, cb.outlined_code_size());
        assert_eq!(outlined_start + 2, cb.outlined_write_ptr().as_offset());

        // And switching back appends to the inlined half where it left off.
        cb.set_outlined(was_outlined);
        assert!(!cb.is_outlined());
        assert_eq!(1, cb.get_write_pos());
        cb.write_byte(4);
        assert_eq!(2, cb.inlined_code_size());
        assert_eq!(2, cb.outlined_code_size());
    }

    #[test]
    fn test_inlined_half_stops_at_the_boundary_instead_of_overwriting_cold_code() {
        let mut cb = CodeBlock::new_dummy_split();
        let half = cb.virtual_region_size() / 2;

        // Put a byte in the outlined half...
        cb.set_outlined(true);
        cb.write_byte(0x11);
        cb.set_outlined(false);

        // ...then run the inlined half up to the boundary and one byte past it.
        cb.set_write_ptr(cb.get_ptr(half - 1));
        cb.write_byte(0x22);
        assert!(!cb.has_dropped_bytes(), "the last byte of the inlined half is writable");
        assert_eq!(half, cb.get_write_pos());

        cb.write_byte(0x33);
        assert!(cb.has_dropped_bytes(), "a write past the inlined half should be dropped");
        assert_eq!(half, cb.get_write_pos(), "the dropped byte should not advance the position");
        assert_eq!(1, cb.outlined_code_size(), "cold code should not have been overwritten");
    }

    #[test]
    fn test_unsplit_code_block_ignores_set_outlined() {
        // The backends' disassembly snapshots run on an unsplit region, where
        // asking for the outlined half has to be a no-op.
        let mut cb = CodeBlock::new_dummy();
        assert!(!cb.has_outlined_region());
        assert!(!cb.set_outlined(true));
        assert!(!cb.is_outlined());
        cb.write_byte(1);
        assert_eq!(1, cb.inlined_code_size());
        assert_eq!(0, cb.outlined_code_size());
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
