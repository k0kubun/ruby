use std::alloc::{handle_alloc_error, Layout};
use std::ptr;

use crate::cast::{IntoU64, IntoUsize};
use crate::cruby::{IseqPtr, VALUE, Qnil, rb_gc_mark_movable, rb_gc_location};
use crate::cruby::{ZJIT_STACK_MAP_OP_END, ZJIT_STACK_MAP_OP_VREG_MIN, ZJIT_STACK_MAP_OP_VREG_MAX, ZJIT_STACK_MAP_OP_VREG, ZJIT_STACK_MAP_OP_NIL, ZJIT_STACK_MAP_OP_IMM, ZJIT_STACK_MAP_OP_SKIP_VM_ENV_DATA, ZJIT_STACK_MAP_OP_PREV_FRAME, ZJIT_STACK_MAP_OP_BASE_PTR, ZJIT_STACK_MAP_OP_SKIP, ZJIT_STACK_MAP_OP_SKIP_MIN};
use crate::cruby::zjit_jit_frame;
use crate::codegen::iseq_may_write_block_code;
use crate::options::get_option;
use crate::state::ZJITState;
use crate::stats::{incr_counter_by, Counter};
use crate::virtualmem::VirtualMem;

/// JITFrame struct is defined in zjit.h (the single source of truth) and
/// imported into Rust via bindgen. See zjit.h for field documentation.
pub type JITFrame = zjit_jit_frame;

/// A bump allocator for JITFrame backed by address space below INT32_MAX.
pub struct JITFrameAllocator {
    /// Address space below `INT32_MAX`. Physical pages are mapped in on demand
    /// and, since the region is never marked executable, stay read/write for the
    /// life of the process.
    virt_mem: VirtualMem,

    /// Offset of the next free byte in `virt_mem`, or the region size once the
    /// arena is exhausted.
    cursor: usize,
}

impl JITFrameAllocator {
    /// Reserve the arena's address space, or return None when the platform cannot
    /// provide low memory.
    pub fn new() -> Option<Self> {
        // A quarter of --zjit-mem-size is reserved, not mapped: physical pages are
        // mapped in one at a time as the bump cursor crosses into them. We pick this
        // ratio emperically by running against ruby-bench.
        Some(JITFrameAllocator {
            virt_mem: VirtualMem::alloc_low(get_option!(mem_bytes) / 4)?,
            cursor: 0,
        })
    }

    /// The number of bytes the allocator has mapped physical memory for.
    pub fn mapped_bytes(&self) -> usize {
        self.virt_mem.mapped_region_size()
    }

    /// Bump-allocate `layout` from the arena, or return null when it's full.
    fn try_alloc(&mut self, layout: Layout) -> *mut u8 {
        // The region start is page-aligned, so aligning the offset aligns the pointer.
        debug_assert!(layout.align() <= self.virt_mem.system_page_size());
        let start = self.cursor.next_multiple_of(layout.align());
        let Some(end) = start.checked_add(layout.size()) else {
            return ptr::null_mut();
        };
        if end > self.virt_mem.virtual_region_size() {
            return ptr::null_mut();
        }
        // Touch the last byte of the allocation on first use: write_byte maps every
        // page between the end of the mapped region and the byte it writes.
        if end > self.virt_mem.mapped_region_size() {
            let last_byte = self.virt_mem.start_ptr().add_bytes(end - 1);
            if self.virt_mem.write_byte(last_byte, 0).is_err() {
                // write_byte's bound check refuses the final page of the region, so
                // treat any failure as permanent exhaustion instead of retrying on
                // every allocation.
                self.cursor = self.virt_mem.virtual_region_size();
                return ptr::null_mut();
            }
        }
        self.cursor = end;
        self.virt_mem.start_ptr().add_bytes(start).raw_ptr(&self.virt_mem) as *mut u8
    }
}

/// Bump-allocate `layout` from the JITFrame arena, falling back to the heap.
fn alloc_jit_frame_memory(layout: Layout) -> *mut u8 {
    // Prefer the low-address arena so that call sites can store JITFrame pointers
    // as a 32-bit immediate. Falling back to the heap only costs code size.
    let mut raw_ptr = match ZJITState::get_jit_frame_allocator() {
        Some(arena) => arena.try_alloc(layout),
        None => ptr::null_mut(),
    };
    if raw_ptr.is_null() {
        raw_ptr = unsafe { std::alloc::alloc(layout) };
        if raw_ptr.is_null() {
            handle_alloc_error(layout);
        }
        incr_counter_by(Counter::jit_frame_heap_bytes, layout.size().as_u64());
    }
    raw_ptr
}

/// Stack maps with no entries, indexed by materialize_block_code.
static EMPTY_STACK_MAPS: [[u8; 2]; 2] = [[0, ZJIT_STACK_MAP_OP_END as u8], [1, ZJIT_STACK_MAP_OP_END as u8]];

/// Writes a JITFrame stack map. Byte 0 holds flags, whose bit 0 is `materialize_block_code`.
/// The rest is a sequence of one-byte opcodes, some followed by unsigned LEB128 operands, and
/// [`ZJIT_STACK_MAP_OP_END`] ends it. See zjit.h for the opcodes.
///
/// zjit_materialize_frames() decodes the opcodes in order while moving a write cursor down the VM
/// stack. Most opcodes are a native stack slot index small enough to be the opcode itself, so the
/// decoder writes most slots with one byte load and one compare, without decoding any operand.
pub struct StackMapWriter {
    bytes: Vec<u8>,
    /// The length of `bytes` after the last opcode that writes a slot. The opcodes after it
    /// only move the cursor, so [Self::finish] drops them.
    written_len: usize,
    /// Where the last opcode starts, if it's a skip that fits in the opcode. Consecutive skips
    /// are merged into it.
    short_skip_pos: Option<usize>,
}

#[cfg(test)]
#[test]
fn test_stack_map_example() {
    let mut writer = StackMapWriter::new(true);
    writer.vreg(5);      // slot index as the opcode
    writer.vreg(300);    // VREG, then 300 in LEB128
    writer.skip(16);     // number of slots in the opcode
    writer.skip(1);      // merged into the previous skip
    writer.imm(Qnil);
    writer.prev_frame(); // dropped, since no slot is written after it
    writer.skip(3);
    assert_eq!(writer.finish(), [
        1,                                 // materialize_block_code
        5,
        ZJIT_STACK_MAP_OP_VREG as u8, 0b1_0101100, 0b0_0000010,
        (ZJIT_STACK_MAP_OP_SKIP_MIN - 1 + 17) as u8,
        ZJIT_STACK_MAP_OP_NIL as u8,
        ZJIT_STACK_MAP_OP_END as u8,
    ]);

    // A stack map that writes no slot has no opcode other than END
    let mut writer = StackMapWriter::new(false);
    writer.base_ptr(3, 2);
    writer.skip(2);
    writer.skip_vm_env_data();
    assert_eq!(writer.finish(), [0, ZJIT_STACK_MAP_OP_END as u8]);
}

impl StackMapWriter {
    pub fn new(materialize_block_code: bool) -> Self {
        Self { bytes: vec![u8::from(materialize_block_code)], written_len: 1, short_skip_pos: None }
    }

    fn write_op(&mut self, op: u32) {
        self.short_skip_pos = None;
        self.bytes.push(op.try_into().unwrap());
    }

    /// Mark the end of an opcode that writes a slot
    fn wrote_slot(&mut self) {
        self.written_len = self.bytes.len();
    }

    fn write_uleb128(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                self.bytes.push(byte);
                break;
            }
            self.bytes.push(byte | 0x80);
        }
    }

    /// Write `jit_return[-slot_index]`. A slot index in
    /// ZJIT_STACK_MAP_OP_VREG_MIN..=ZJIT_STACK_MAP_OP_VREG_MAX is the opcode itself.
    /// Others use ZJIT_STACK_MAP_OP_VREG, whose operand is the slot index.
    pub fn vreg(&mut self, slot_index: usize) {
        if (ZJIT_STACK_MAP_OP_VREG_MIN.to_usize()..=ZJIT_STACK_MAP_OP_VREG_MAX.to_usize()).contains(&slot_index) {
            self.write_op(slot_index as u32);
        } else {
            self.write_op(ZJIT_STACK_MAP_OP_VREG);
            self.write_uleb128(slot_index.as_u64());
        }
        self.wrote_slot();
    }

    /// ZJIT_STACK_MAP_OP_NIL writes Qnil. Other immediates use
    /// ZJIT_STACK_MAP_OP_IMM, whose operand is the VALUE to write.
    pub fn imm(&mut self, value: VALUE) {
        assert!(value.special_const_p(), "StackMap should only materialize immediate VALUEs, but got: {value:?}");
        if value == Qnil {
            self.write_op(ZJIT_STACK_MAP_OP_NIL);
        } else {
            self.write_op(ZJIT_STACK_MAP_OP_IMM);
            self.write_uleb128(value.as_u64());
        }
        self.wrote_slot();
    }

    /// Move the cursor down `size` slots without writing. Up to
    /// `0xff - ZJIT_STACK_MAP_OP_SKIP_MIN + 1` slots fit in the opcode. Others use
    /// ZJIT_STACK_MAP_OP_SKIP, whose operand is the number of slots.
    pub fn skip(&mut self, mut size: usize) {
        if size == 0 {
            return;
        }
        // Merge consecutive skips
        if let Some(pos) = self.short_skip_pos {
            size += usize::from(self.bytes[pos]) - (ZJIT_STACK_MAP_OP_SKIP_MIN.to_usize() - 1);
            self.bytes.truncate(pos);
        }
        let short_op = ZJIT_STACK_MAP_OP_SKIP_MIN.to_usize() - 1 + size;
        if short_op <= u8::MAX.into() {
            self.write_op(short_op as u32);
            self.short_skip_pos = Some(self.bytes.len() - 1);
        } else {
            self.write_op(ZJIT_STACK_MAP_OP_SKIP);
            self.write_uleb128(size.as_u64());
        }
    }

    /// ZJIT_STACK_MAP_OP_BASE_PTR. Operands: slot index, stack size. Anchors the
    /// cursor at `((VALUE **)cfp->jit_return)[-slot_index] + stack_size`, i.e. on
    /// the SP register the JIT saved on its native stack instead of on `cfp->sp`.
    /// `cfp->sp` is not a reliable starting point for a frame that is in the
    /// middle of a non-leaf C call, as e.g. raising ArgumentError can push through
    /// and move `cfp->sp`, then use the stack map. gen_prepare_non_leaf_call()
    /// emits this opcode, always as the first one, so the opcodes after decode to
    /// the right place.
    pub fn base_ptr(&mut self, slot_index: u32, stack_size: u32) {
        debug_assert_eq!(self.bytes.len(), 1, "base_ptr must be the first StackMap entry so later entries decode from it");
        self.write_op(ZJIT_STACK_MAP_OP_BASE_PTR);
        self.write_uleb128(slot_index.into());
        self.write_uleb128(stack_size.into());
    }

    /// ZJIT_STACK_MAP_OP_SKIP_VM_ENV_DATA. Moves the cursor past the frame's env
    /// data (me/cref, specval, flags) and starts the frame's locals area. Writes in
    /// the locals area are dropped when the frame's env has escaped.
    pub fn skip_vm_env_data(&mut self) {
        self.write_op(ZJIT_STACK_MAP_OP_SKIP_VM_ENV_DATA);
    }

    /// ZJIT_STACK_MAP_OP_PREV_FRAME. Continues decoding in the caller's control
    /// frame, leaving the locals area. It doesn't move the cursor, so the inlined
    /// callee's receiver slot, if any, is skipped with a preceding SKIP.
    pub fn prev_frame(&mut self) {
        self.write_op(ZJIT_STACK_MAP_OP_PREV_FRAME);
    }

    /// Terminate the stack map with ZJIT_STACK_MAP_OP_END and return its bytes. The opcodes
    /// after the last one that writes a slot only move the cursor, so they are dropped.
    pub fn finish(mut self) -> Vec<u8> {
        self.bytes.truncate(self.written_len);
        self.write_op(ZJIT_STACK_MAP_OP_END);
        self.bytes
    }
}

impl JITFrame {
    /// Allocate a JITFrame with an empty stack map, register it with ZJITState,
    /// and return a raw pointer that remains valid for the lifetime of the process.
    fn alloc(pc: *const VALUE, iseq: IseqPtr, materialize_block_code: bool) -> *const Self {
        let raw_ptr: *mut Self = alloc_jit_frame_memory(Layout::new::<JITFrame>()).cast();
        unsafe {
            ptr::write(raw_ptr, JITFrame {
                pc,
                iseq,
                stack_map: EMPTY_STACK_MAPS[usize::from(materialize_block_code)].as_ptr(),
            });
        }
        ZJITState::get_jit_frames().push(raw_ptr);
        raw_ptr.cast_const()
    }

    pub fn materialize_block_code(&self) -> bool {
        // Bit 0 of byte 0 is this boolean.
        let zeroth_byte = unsafe { *self.stack_map };
        zeroth_byte & 1 != 0
    }

    pub fn stack_map_writer(&self) -> StackMapWriter {
        StackMapWriter::new(self.materialize_block_code())
    }

    /// Install a stack map built from [Self::stack_map_writer].
    pub fn set_stack_map(&mut self, writer: StackMapWriter) {
        debug_assert_eq!(writer.bytes.first().map(|byte| byte & 1 != 0), Some(self.materialize_block_code()));
        // Avoid allocating memory more than once for the same stack map
        if EMPTY_STACK_MAPS.iter().all(|empty| !ptr::eq(self.stack_map, empty.as_ptr())) {
            return;
        }
        let bytes = writer.finish();
        let stack_map = alloc_jit_frame_memory(Layout::array::<u8>(bytes.len()).unwrap());
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), stack_map, bytes.len()); }
        incr_counter_by(Counter::jit_frame_stack_map_bytes, bytes.len().as_u64());
        self.stack_map = stack_map;
    }

    /// Create a JITFrame for an ISEQ frame.
    pub fn new_iseq(pc: *const VALUE, iseq: IseqPtr) -> *const Self {
        let materialize_block_code = !iseq_may_write_block_code(iseq);
        Self::alloc(pc, iseq, materialize_block_code)
    }

    /// Mark the iseq pointer for GC. Called from rb_zjit_root_mark.
    pub fn mark(&self) {
        if !self.iseq.is_null() {
            unsafe { rb_gc_mark_movable(VALUE::from(self.iseq)); }
        }
    }

    /// Update the iseq pointer after GC compaction.
    pub fn update_references(&mut self) {
        if !self.iseq.is_null() {
            let new_iseq = unsafe { rb_gc_location(VALUE::from(self.iseq)) }.as_iseq();
            if self.iseq != new_iseq {
                self.iseq = new_iseq;
            }
        }
    }
}

/// Update the iseq pointer in an on-stack JITFrame during GC compaction.
/// Called from rb_execution_context_update in vm.c.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_jit_frame_update_references(jit_frame: *mut JITFrame) {
    unsafe { &mut *jit_frame }.update_references();
}

#[cfg(test)]
mod tests {
    use crate::cruby::{eval, inspect};
    use insta::assert_snapshot;

    #[test]
    fn test_jit_frame_entry_first() {
        eval(r#"
            def test
              itself
              callee
            end

            def callee
              caller
            end

            test
        "#);
        assert_snapshot!(inspect("test.first"), @r#""<compiled>:4:in 'Object#test'""#);
    }

    #[test]
    fn test_materialize_one_frame() {
        assert_snapshot!(inspect("
            def jit_entry
              raise rescue 1
            end
            jit_entry
            jit_entry
        "), @"1");
    }

    #[test]
    fn test_materialize_two_frames() { // materialize caller frames on raise
        // At the point of `rescue`, there are two inline frames on stack and both need to be
        // materialized before passing control to interpreter.
        assert_snapshot!(inspect("
            def jit_entry = raise_and_rescue
            def raise_and_rescue
              raise rescue 1
            end
            jit_entry
            jit_entry
        "), @"1");
    }

    // Direct JIT-to-JIT entry passes callee locals as native arguments. If the
    // callee ISEQ has already escaped EP, later getlocal reads use EP memory,
    // so JIT entry must materialize those locals into the callee frame.
    #[test]
    fn test_jit_entry_materializes_ep_escaped_locals() {
        assert_snapshot!(inspect("
            def poison(*) = nil

            def victim(a, b, c)
              lambda { a }
              a
            end

            def jit_entry
              poison([], [], [], [])
              victim(:expected, 1, 2)
            end

            jit_entry
            Array.new(100) { jit_entry }.uniq
        "), @"[:expected]");
    }

    // Materialize frames on side exit: a type guard triggers a side exit with
    // multiple JIT frames on the stack. All frames must be materialized before
    // the interpreter resumes.
    #[test]
    fn test_side_exit_materialize_frames() {
        assert_snapshot!(inspect("
            def side_exit(n) = 1 + n
            def jit_frame(n) = 1 + side_exit(n)
            def entry(n) = jit_frame(n)
            entry(2)
            [entry(2), entry(2.0)]
        "), @"[4, 4.0]");
    }

    // BOP invalidation must not overwrite the top-most frame's PC with
    // jit_frame's PC. After invalidation the interpreter resumes at a new
    // PC, so a stale jit_frame PC would cause wrong execution.
    #[test]
    fn test_bop_invalidation() {
        assert_snapshot!(inspect(r#"
            def test
              eval("class Integer; def +(_) = 100; end")
              1 + 2
            end
            test
            test
        "#), @"100");
    }

    // Side exit at the very start of a method, before gen_save_pc_for_gc has
    // updated the entry JITFrame.
    #[test]
    fn test_side_exit_before_jit_frame_update() {
        assert_snapshot!(inspect("
            def entry(n) = n + 1
            entry(1)
            [entry(1), entry(1.0)]
        "), @"[2, 2.0]");
    }

    #[test]
    fn test_caller_iseq() {
        assert_snapshot!(inspect(r#"
            def callee = call_caller
            def test = callee

            def callee2 = call_caller
            def test2 = callee2

            def call_caller = caller

            test
            test2
            test.first
        "#), @r#""<compiled>:2:in 'Object#callee'""#);
    }

    // ISEQ must be readable during exception handling so the interpreter
    // can look up rescue/ensure tables.
    #[test]
    fn test_iseq_on_raise() {
        assert_snapshot!(inspect(r#"
            def jit_entry(v) = make_range_then_exit(v)
            def make_range_then_exit(v)
              range = (v..1)
              super rescue range
            end
            jit_entry(0)
            jit_entry(0)
            jit_entry(0/1r)
        "#), @"(0/1)..1");
    }

    // Multiple exception raises during keyword argument evaluation: each
    // raise needs correct ISEQ for catch table lookup.
    #[test]
    fn test_iseq_on_raise_on_ensure() {
        assert_snapshot!(inspect(r#"
            def raise_a = raise "a"
            def raise_b = raise "b"
            def raise_c = raise "c"

            def foo(a: raise_a, b: raise_b, c: raise_c)
              [a, b, c]
            end

            def test_a
              foo(b: 2, c: 3)
            rescue RuntimeError => e
              e.message
            end

            def test_b
              foo(a: 1, c: 3)
            rescue RuntimeError => e
              e.message
            end

            def test_c
              foo(a: 1, b: 2)
            rescue RuntimeError => e
              e.message
            end

            def test
              [test_a, test_b, test_c]
            end

            test
            test
        "#), @r#"["a", "b", "c"]"#);
    }

    // Send fallback (e.g. method_missing) calls into the interpreter, which
    // reads cfp->iseq via GET_ISEQ(). gen_prepare_non_leaf_call writes the
    // iseq to JITFrame, but GET_ISEQ reads cfp->iseq directly. This test
    // ensures the interpreter can resolve the caller iseq for backtraces.
    #[test]
    fn test_send_fallback_caller_location() {
        assert_snapshot!(inspect(r#"
            def callee = caller_locations(1, 1)[0].label
            def test = callee
            test
            test
        "#), @r#""Object#test""#);
    }

    // A fallback send hands the receiver and arguments to a VM helper that pushes the
    // callee's frame over those same slots and drops this frame's cfp->sp below them
    // (vm_call_cfunc_with_frame_(), vm_call_iseq_setup_normal()). Here define_method
    // turns the literal block into a Proc, which escapes the calling block's env and
    // materializes it while the define_method frame is still live, so this frame's
    // stack map has to skip its operand stack rather than write it back over that
    // frame. See gen_prepare_fallback_call(); on a build with RUBY_DEBUG this dies in
    // zjit_check_owned_slot() without the skip.
    #[test]
    fn test_fallback_send_does_not_write_over_callee_frame() {
        assert_snapshot!(inspect(r#"
            def go(i)
              Class.new { define_method(:m) { i } }
            end
            4.times { |i| go(i) }
            go(99).new.m
        "#), @"99");
    }

    // A send fallback may throw (e.g. via method_missing raising). The
    // interpreter must be able to find the correct rescue handler in the
    // caller's ISEQ catch table. This exercises throw through send fallback.
    #[test]
    fn test_send_fallback_throw() {
        assert_snapshot!(inspect(r#"
            class Foo
              def method_missing(name, *) = raise("no #{name}")
            end
            def test
              Foo.new.bar
            rescue RuntimeError => e
              e.message
            end
            test
            test
        "#), @r#""no bar""#);
    }

    // This test makes a JIT control frame move cfp->sp from as set by gen_prepare_non_leaf_call()
    // and then ask for materialization. A C function that calls back into Ruby (rb_const_missing
    // here) pushes recv+args using jit_entry's cfp->sp via vm_call0_body(). A wrong arity makes
    // vm_callee_setup_arg() raise before vm_call_iseq_setup_normal() restores cfp->sp, so cfp->sp
    // is left 1+argc slots high while this frame is materialized for the rescue. At the time of
    // materialization, the top most control frame is the JIT frame.
    #[test]
    fn test_stack_map_anchor_after_callee_arity_error() {
        assert_snapshot!(inspect(r#"
            class Holder
              def self.const_missing(a, b) = nil # wrong arity: called with 1 arg
            end
            def jit_entry
              [1, (begin # the 1 is live across the const_missing call
                     Holder::NOPE
                   rescue ArgumentError
                     2
                   end)]
            end
            jit_entry
            jit_entry
        "#), @"[1, 2]");
    }

    // Same displaced cfp->sp, but reaching across an inlined frame: `defined?`
    // calls respond_to_missing? with 2 args, which raises on arity.
    #[test]
    fn test_stack_map_anchor_with_inlined_frame() {
        assert_snapshot!(inspect(r#"
            class BadResponder
              def respond_to_missing?(name) = true # wrong arity: called with 2
            end
            class Test
              def initialize = @o = BadResponder.new
              def inner = defined?(@o.nope) # must have 0 locals
              def outer = [1, inner] # the 1 is live across the inlined call
            end
            test = Test.new
            test.outer
            test.outer
        "#), @"[1, nil]");
    }

    // vm_make_env_each() spills through the same decoder, so an escaping env in
    // an inlined frame decodes a map that spans three control frames and crosses
    // two StackMapEntry::PrevFrame boundaries. Unlike the raise-based tests in
    // codegen_tests.rs, nothing unwinds here: `keep` has to survive in the
    // caller frame that the map also describes.
    #[test]
    fn test_stack_map_spill_across_inlined_frames() {
        assert_snapshot!(inspect("
            def leaf(escape)
              proc { } if escape # escapes only this frame's env
              escape ? 2 : 1
            end

            def middle(n)
              keep = n + 40
              keep + leaf(n > 0)
            end

            def outer(n) = middle(n)

            outer(0)
            outer(0)
            [outer(1), outer(0)]
        "), @"[43, 41]");
    }

    // Integer#times and its block are inlined into test, and Fiber.new escapes the env
    // of the block and then of test, which has no stack map of its own while the
    // inlined frames run. Its locals come from the stack map of the inlined block.
    #[test]
    fn test_escape_env_of_frame_with_inlined_callee() {
        assert_snapshot!(inspect("
            def test(escape)
              max = 3
              r1 = max.times { Fiber.new {} if escape }
              r2 = max.times { }
              [max, r1, r2]
            end

            test(false)
            test(false)
            test(true)
        "), @"[3, 3, 3]");
    }

    // A non-local return from a block called by a C method longjmps over the block,
    // the C method and find_key. The caller of find_key resumes in the interpreter, so
    // its locals have to be written from its stack map.
    #[test]
    fn test_nonlocal_return_through_cfunc_keeps_caller_locals() {
        assert_snapshot!(inspect("
            def find_key(h)
              unused = h.size + 1
              h.each { |k, v| return k if v == 2 }
              unused
            end

            def test(h)
              a = h.size
              b = [a]
              r = find_key(h)
              [a, b, r]
            end

            h = {x: 1, y: 2}
            test(h)
            test(h)
            test(h)
        "), @"[2, [2], :y]");
    }

    // find_key escapes the env of test to a Proc, then returns non-locally through a
    // C method. The locals of test, which the Proc reads through the heap env, survive.
    #[test]
    fn test_nonlocal_return_after_escaping_caller_env() {
        assert_snapshot!(inspect("
            def find_key(h, &blk)
              $blk = blk
              h.each { |k, v| return k if v == 2 }
              nil
            end

            def test(h)
              a = h.size
              b = [a]
              r = find_key(h) { a += 1 }
              [a, b, r, $blk.call, a]
            end

            h = {x: 1, y: 2}
            test(h)
            test(h)
            [test(h), test(h)]
        "), @"[[2, [2], :y, 3, 3], [2, [2], :y, 3, 3]]");
    }

    // An ensure clause in a frame that a non-local return unwinds runs before the
    // frames below it are popped, and reads find_key's locals through its env.
    #[test]
    fn test_nonlocal_return_runs_ensure_with_outer_locals() {
        assert_snapshot!(inspect("
            def find_key(h, log)
              seen = 0
              h.each do |k, v|
                seen += 1
                begin
                  return k if v == 2
                ensure
                  log << seen
                end
              end
              nil
            end

            log = []
            3.times { find_key({x: 1, y: 2}, log) }
            log
        "), @"[1, 2, 1, 2, 1, 2]");
    }

    // A break from a block called by a C method resumes the frame that called it.
    #[test]
    fn test_break_through_cfunc_keeps_caller_locals() {
        assert_snapshot!(inspect("
            def test(h)
              a = h.size
              r = h.each { |k, v| break [k, a] if v == 2 }
              [a, r]
            end

            h = {x: 1, y: 2}
            test(h)
            test(h)
            test(h)
        "), @"[2, [:y, 2]]");
    }

    // An exception raised in a block called by a C method is rescued by an older frame,
    // which has to see its own locals.
    #[test]
    fn test_raise_through_cfunc_keeps_rescuer_locals() {
        assert_snapshot!(inspect(r##"
            def inner(h)
              c = h.size
              h.each { |k, v| raise "#{k}#{c}" if v == 2 }
            end

            def outer(h)
              a = h.size * 3
              begin
                inner(h)
              rescue => e
                [a, e.message]
              end
            end

            h = {x: 1, y: 2}
            outer(h)
            outer(h)
            outer(h)
        "##), @r#"[6, "y2"]"#);
    }

    // A :return hook sees the locals of a frame that a non-local return unwinds.
    #[test]
    fn test_nonlocal_return_return_hook_sees_locals() {
        assert_snapshot!(inspect("
            def find_key(h)
              local = h.size * 21
              h.each { |k, v| return k if v == 2 }
            end

            3.times { find_key({x: 1, y: 2}) }
            vals = []
            tp = TracePoint.new(:return) { |t| vals << t.binding.local_variable_get(:local) if t.method_id == :find_key }
            tp.enable { find_key({x: 1, y: 2}) }
            vals
        "), @"[42]");
    }

    // Proc.new inside a block passed via invokeblock captures the caller's
    // block_code. When the JIT compiles the caller, block_code must be
    // correctly available for the proc to work.
    #[test]
    fn test_proc_from_invokeblock() {
        assert_snapshot!(inspect("
            def capture_block(&blk) = blk
            def test = capture_block { 42 }
            test
            test.call
        "), @"42");
    }

    // binding() called from a JIT-compiled callee must see the correct
    // source location (iseq + pc) of the caller frame.
    #[test]
    fn test_binding_source_location() {
        assert_snapshot!(inspect(r#"
            def callee = binding
            def test = callee
            test
            b = test
            b.source_location[1] > 0
        "#), @"true");
    }

    // $~ (Regexp special variable) is stored via svar which walks the EP
    // chain to find the LEP. rb_vm_svar_lep uses rb_zjit_cfp_has_iseq to
    // skip C frames, so it must work correctly with JITFrame.
    #[test]
    fn test_svar_regexp_match() {
        assert_snapshot!(inspect(r#"
            def test(s)
              s =~ /hello/
              $~
            end
            test("hello world")
            test("hello world").to_s
        "#), @r#""hello""#);
    }

    // C function calls with rb_block_call (like Array#each, Enumerable#map)
    // write an ifunc to cfp->block_code after the JIT pushes the C frame.
    // GC must mark and relocate this ifunc. This test exercises the code
    // path fixed by "Fix ZJIT segfault: write block_code for C frames and
    // fix GC marking".
    #[test]
    fn test_cfunc_block_code_gc() {
        assert_snapshot!(inspect("
            def test
              # Use a cfunc that calls back into Ruby with a block (rb_block_call)
              [1, 2, 3].map { |x| x.to_s }
            end
            test
            test
        "), @r#"["1", "2", "3"]"#);
    }

    // Multiple levels of cfunc-with-block: a JIT-compiled method calls a
    // cfunc that yields, and the block itself calls another cfunc that
    // yields. Each C frame's block_code must be properly initialized.
    #[test]
    fn test_nested_cfunc_with_block() {
        assert_snapshot!(inspect("
            def test
              [1, 2].flat_map { |x| [x, x + 10].map { |y| y * 2 } }
            end
            test
            test
        "), @"[2, 22, 4, 24]");
    }
}
