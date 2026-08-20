use std::alloc::{alloc, handle_alloc_error, Layout};
use std::cell::Cell;
use std::mem::{align_of, size_of};
use std::ptr;

use crate::cruby::{__IncompleteArrayField, IseqPtr, VALUE, rb_gc_location, rb_jit_reserve_low_addr_space};
use crate::cruby::zjit_jit_frame;
use crate::codegen::iseq_may_write_block_code;
use crate::state::ZJITState;

/// JITFrame struct is defined in zjit.h (the single source of truth) and
/// imported into Rust via bindgen. See zjit.h for field documentation.
pub type JITFrame = zjit_jit_frame;

/// How much address space to grab per arena chunk. Lobsters, ZJIT's largest
/// benchmark, allocates under 4MiB of JITFrames in total, so one chunk normally
/// serves a whole process.
const ARENA_CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// A bump allocator for [`JITFrame`]s backed by memory below `INT32_MAX`.
///
/// Every JIT-to-JIT call site stores the address of a JITFrame into the native
/// stack slot that `cfp->jit_return` points at. That address is a compile-time
/// constant, so its width decides the width of the store: an address that
/// survives sign extension from 32 bits is folded into the store as an immediate
/// (`mov qword ptr [rbp - 8], imm32`, 8 bytes on x86-64), while a full 64-bit
/// address needs a `movabs` into a scratch register first (14 bytes). The stores
/// happen at essentially every call site in JIT code, so the difference is worth
/// a dedicated allocator.
///
/// JITFrames are never freed -- a frame stays reachable for as long as the code
/// that references it, which is for the life of the process -- so a bump
/// allocator with no free list is all this needs. When the low address space is
/// unavailable or exhausted, allocation falls back to the ordinary Rust heap and
/// codegen simply emits the wide form for those frames.
struct LowArena {
    /// Next free byte in the current chunk, or null before the first chunk.
    cursor: Cell<*mut u8>,
    /// One past the last usable byte of the current chunk.
    end: Cell<*mut u8>,
    /// Set once the platform has told us it cannot provide low memory, so that
    /// we stop asking on every allocation.
    exhausted: Cell<bool>,
    /// Total bytes mapped, for --zjit-stats memory accounting.
    mapped_bytes: Cell<usize>,
}

impl LowArena {
    const fn new() -> Self {
        LowArena {
            cursor: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            exhausted: Cell::new(false),
            mapped_bytes: Cell::new(0),
        }
    }

    /// Bump-allocate `layout` from low memory, or return null when unavailable.
    fn try_alloc(&self, layout: Layout) -> *mut u8 {
        debug_assert!(layout.align() <= 16, "arena chunks are page aligned");
        if self.exhausted.get() {
            return ptr::null_mut();
        }
        loop {
            let cursor = self.cursor.get();
            if !cursor.is_null() {
                let aligned = (cursor as usize).next_multiple_of(layout.align());
                // The chunk is one mapping, so this arithmetic stays inside it.
                if let Some(next) = aligned.checked_add(layout.size()) {
                    if next <= self.end.get() as usize {
                        self.cursor.set(next as *mut u8);
                        return aligned as *mut u8;
                    }
                }
            }
            // Current chunk is full (or there is none yet): map another one.
            let chunk = unsafe { rb_jit_reserve_low_addr_space(ARENA_CHUNK_SIZE) } as *mut u8;
            if chunk.is_null() || layout.size() > ARENA_CHUNK_SIZE {
                self.exhausted.set(true);
                return ptr::null_mut();
            }
            self.mapped_bytes.set(self.mapped_bytes.get() + ARENA_CHUNK_SIZE);
            self.cursor.set(chunk);
            self.end.set(unsafe { chunk.add(ARENA_CHUNK_SIZE) });
        }
    }
}

// Only ever touched with the GVL held, from JITFrame::alloc() on the compiling
// thread and from mem_stats.
unsafe impl Sync for LowArena {}

static LOW_ARENA: LowArena = LowArena::new();

/// Bytes of JITFrame that the arena could not serve and that went to the Rust heap.
struct HeapFallbackBytes(Cell<usize>);
unsafe impl Sync for HeapFallbackBytes {}
static HEAP_FALLBACK_BYTES: HeapFallbackBytes = HeapFallbackBytes(Cell::new(0));

/// Bytes the JITFrame allocator holds: whole arena chunks plus any heap fallback.
/// This counts mapped chunks rather than live frames, so it includes the slack at
/// the end of the current chunk. For --zjit-stats.
pub fn allocated_bytes() -> usize {
    LOW_ARENA.mapped_bytes.get() + HEAP_FALLBACK_BYTES.0.get()
}

impl JITFrame {
    /// Allocate a JITFrame and its trailing stack map on the heap, register it
    /// with ZJITState, and return a raw pointer that remains valid for the
    /// lifetime of the process.
    fn alloc(
        pc: *const VALUE,
        iseq: IseqPtr,
        materialize_block_code: bool,
        stack_size: usize,
    ) -> *const Self {
        // JITFrame ends with a flexible stack[] array, so allocate enough
        // space for the fixed fields plus the requested stack map entries.
        let frame_size = size_of::<JITFrame>()
            .checked_add(stack_size.checked_mul(size_of::<VALUE>()).unwrap())
            .unwrap();
        let layout = Layout::from_size_align(frame_size, align_of::<JITFrame>()).unwrap();
        // Prefer the low-address arena so that call sites can store this pointer
        // as a 32-bit immediate. Falling back to the heap only costs code size.
        let mut raw_ptr = LOW_ARENA.try_alloc(layout) as *mut JITFrame;
        if raw_ptr.is_null() {
            raw_ptr = unsafe { alloc(layout) as *mut JITFrame };
            HEAP_FALLBACK_BYTES.0.set(HEAP_FALLBACK_BYTES.0.get() + layout.size());
        }
        if raw_ptr.is_null() {
            handle_alloc_error(layout);
        }

        unsafe {
            ptr::write(raw_ptr, JITFrame {
                pc,
                iseq,
                materialize_block_code,
                stack_size: stack_size.try_into().unwrap(),
                stack: __IncompleteArrayField::new(),
            });
        }
        // The frame's ISEQ has to stay alive for as long as the frame does, which is
        // forever; the mark phase reaches it through this set rather than by walking
        // every frame. See [`crate::gc::RootIseqs`].
        crate::gc::register_root_iseq(iseq);
        ZJITState::get_jit_frames().push(raw_ptr);
        raw_ptr as *const _
    }

    /// Create a JITFrame for an ISEQ frame.
    pub fn new_iseq(pc: *const VALUE, iseq: IseqPtr, stack_size: usize) -> *const Self {
        let materialize_block_code = !iseq_may_write_block_code(iseq);
        Self::alloc(pc, iseq, materialize_block_code, stack_size)
    }

    /// Bytes this frame occupies on the Rust heap, including the trailing
    /// stack map that [`Self::alloc`] over-allocated for.
    pub fn heap_size(&self) -> usize {
        size_of::<JITFrame>() + self.stack_size as usize * size_of::<VALUE>()
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
