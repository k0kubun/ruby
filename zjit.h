#ifndef ZJIT_H
#define ZJIT_H 1
//
// This file contains definitions ZJIT exposes to the CRuby codebase
//

#include "shape.h" // for shape_id_t

// ZJIT_STATS controls whether to support runtime counters in the interpreter
#ifndef ZJIT_STATS
# define ZJIT_STATS (USE_ZJIT && RUBY_DEBUG)
#endif

// JITFrame is defined here as the single source of truth and imported into
// Rust via bindgen. C code reads fields directly; Rust uses an impl block.
typedef struct zjit_jit_frame {
    // Program counter for this frame, used for backtraces and GC.
    // NULL for C frames (they don't have a Ruby PC).
    const VALUE *pc;
    // The ISEQ this frame belongs to. Marked via rb_execution_context_mark_ctx.
    // NULL for C frames.
    const rb_iseq_t *iseq;
    // Stack map, decoded by zjit_materialize_frames(). Bit 0 of byte 0 is
    // materialize_block_code, see ZJIT_FRAME_MATERIALIZE_BLOCK_CODE_P().
    // The rest are ZJIT_STACK_MAP_OP_* opcodes, each followed by its operands.
    // NULL for C frames.
    const uint8_t *stack_map;
} zjit_jit_frame_t;

// Whether to materialize block_code when this frame is materialized.
// True when the ISEQ doesn't contain send/invokesuper/invokeblock
// (which write block_code themselves), so we must restore it.
// Always false for C frames.
static inline bool
ZJIT_FRAME_MATERIALIZE_BLOCK_CODE_P(const zjit_jit_frame_t *jit_frame)
{
    // Bit 0 of byte 0 is this boolean.
    return jit_frame->stack_map && (jit_frame->stack_map[0] & 1);
}

#if USE_ZJIT
// JITFrame stack map format. See StackMapWriter in zjit/src/jit_frame.rs.
// Byte 0 holds flags. Each opcode after it is a byte, and some are followed by
// unsigned LEB128 operands.
#define ZJIT_STACK_MAP_OP_END       0x00
// 0x01..0xcf: write the native stack slot whose index is the opcode itself
#define ZJIT_STACK_MAP_OP_VREG_MIN  0x01
#define ZJIT_STACK_MAP_OP_VREG_MAX  0xcf
#define ZJIT_STACK_MAP_OP_VREG      0xd0 // operand: slot index
#define ZJIT_STACK_MAP_OP_NIL       0xd1
#define ZJIT_STACK_MAP_OP_IMM       0xd2 // operand: VALUE
#define ZJIT_STACK_MAP_OP_SKIP_VM_ENV_DATA 0xd3
#define ZJIT_STACK_MAP_OP_PREV_FRAME 0xd4
#define ZJIT_STACK_MAP_OP_BASE_PTR  0xd5 // operands: slot index, stack size
#define ZJIT_STACK_MAP_OP_SKIP      0xd6 // operand: number of slots
// 0xd7..0xff: skip (opcode - 0xd6) slots
#define ZJIT_STACK_MAP_OP_SKIP_MIN  0xd7

extern void *rb_zjit_entry;
extern bool rb_zjit_compiling_p;
extern const zjit_jit_frame_t rb_zjit_c_frame;
extern unsigned int rb_zjit_call_threshold;
extern unsigned int rb_zjit_profile_threshold;
extern unsigned int rb_zjit_backedge_count;
extern uint64_t rb_zjit_max_compile_time_ns;
extern uint64_t rb_zjit_total_compile_time_ns;
void rb_zjit_compile_iseq(const rb_iseq_t *iseq, rb_execution_context_t *ec);
void rb_zjit_profile_insn(uint32_t insn, rb_execution_context_t *ec);
void rb_zjit_profile_enable(const rb_iseq_t *iseq);
void rb_zjit_iseq_backedge(const rb_iseq_t *iseq);
const void *rb_zjit_resume_entry(rb_execution_context_t *ec, const rb_iseq_t *iseq, void *payload, size_t insn_idx, size_t stack_size);
void rb_zjit_bop_redefined(int redefined_flag, enum ruby_basic_operators bop);
void rb_zjit_cme_invalidate(const rb_callable_method_entry_t *cme);
void rb_zjit_cme_free(const rb_callable_method_entry_t *cme);
void rb_zjit_klass_free(VALUE klass);
void rb_zjit_invalidate_no_ep_escape(const rb_iseq_t *iseq);
void rb_zjit_constant_state_changed(ID id);
void rb_zjit_iseq_mark(void *payload);
void rb_zjit_iseq_update_references(void *payload);
void rb_zjit_mark_all_writable(void);
void rb_zjit_mark_all_executable(void);
void rb_zjit_iseq_free(const rb_iseq_t *iseq);
void rb_zjit_invalidate_single_ractor(void);
void rb_zjit_tracing_invalidate_all(void);
void rb_zjit_invalidate_newobj_hook(void);
void rb_zjit_invalidate_no_singleton_class(VALUE klass);
void rb_zjit_invalidate_root_box(void);
void rb_zjit_jit_frame_update_references(zjit_jit_frame_t *jit_frame);
void rb_zjit_materialize_frames(const rb_execution_context_t *ec, rb_control_frame_t *cfp);
void rb_zjit_materialize_frames_for_longjmp(const rb_execution_context_t *ec, rb_control_frame_t *cfp);
void rb_zjit_spill_frame(const rb_execution_context_t *ec, rb_control_frame_t *cfp);
size_t rb_zjit_hash_new_size(VALUE *flags_out, size_t size);
VALUE rb_zjit_new_obj_shape(VALUE flags, size_t alloc_size);
bool rb_zjit_class_allocate_instance_fastpath(VALUE klass, size_t *size_out, VALUE *flags_out);
bool rb_zjit_str_resurrect_fastpath(VALUE str, bool chilled, size_t *size_out, VALUE *flags_out, rb_len_t *len_out, size_t *byte_size_out);
bool rb_zjit_array_dup_can_fastpath(VALUE ary, size_t *alloc_size_out, VALUE *flags_out, rb_len_t *len_out);
bool rb_zjit_array_new_can_fastpath(rb_len_t len, size_t *alloc_size_out, VALUE *flags_out);
bool rb_zjit_hash_dup_can_fastpath(VALUE hash, size_t *alloc_size_out, VALUE *flags_out, VALUE *ifnone_out, rb_len_t *bound_out);
void rb_zjit_range_new_fastpath(bool exclude_end, size_t *alloc_size_out, VALUE *flags_out);
void rb_zjit_array_new_fastpath(size_t *alloc_size_out, VALUE *flags_out);
bool rb_zjit_ary_s_new_with_block_p(const rb_callable_method_entry_t *cme, const rb_callable_method_entry_t **init_cme);
bool rb_zjit_newobj_hook_enabled_p(void);

// Special value for cfp->jit_return that means "this is a C method frame, use
// rb_zjit_c_frame as the JITFrame". We don't control the native stack layout
// for C frames, so there's no per-call JITFrame storage; we set this sentinel
// instead of a heap-allocated JITFrame pointer.
#define ZJIT_JIT_RETURN_C_FRAME 0x1

static inline const zjit_jit_frame_t *
CFP_ZJIT_FRAME(const rb_control_frame_t *cfp)
{
    if ((VALUE)cfp->jit_return == ZJIT_JIT_RETURN_C_FRAME) {
        return &rb_zjit_c_frame;
    }
    else {
        // Read JITFrame from this frame's stack slot. cfp->jit_return points at
        // the slot reserved for this frame's inlining depth, so distinct frames in
        // the same JIT function read distinct slots. An initial frame describing
        // the entry PC + iseq is written by gen_entry_point() for the top-level
        // frame and by gen_push_inline_frame() for inlined frames. That entry
        // PC is correct only at the frame's start; because the PC this frame reports
        // must track where execution currently is, later gen_save_pc_for_gc() calls
        // rewrite the slot with the live PC as execution advances through the frame,
        // before any non-leaf C call.
        return (const zjit_jit_frame_t *)((VALUE *)cfp->jit_return)[-1];
    }
}
#else
#define rb_zjit_entry 0
#define rb_zjit_compiling_p false
static inline void rb_zjit_compile_iseq(const rb_iseq_t *iseq, rb_execution_context_t *ec) {}
static inline void rb_zjit_profile_insn(uint32_t insn, rb_execution_context_t *ec) {}
static inline void rb_zjit_profile_enable(const rb_iseq_t *iseq) {}
static inline void rb_zjit_iseq_backedge(const rb_iseq_t *iseq) {}
static inline const void *rb_zjit_resume_entry(rb_execution_context_t *ec, const rb_iseq_t *iseq, void *payload, size_t insn_idx, size_t stack_size) { return NULL; }
static inline void rb_zjit_bop_redefined(int redefined_flag, enum ruby_basic_operators bop) {}
static inline void rb_zjit_cme_invalidate(const rb_callable_method_entry_t *cme) {}
static inline void rb_zjit_invalidate_no_ep_escape(const rb_iseq_t *iseq) {}
static inline void rb_zjit_constant_state_changed(ID id) {}
static inline void rb_zjit_invalidate_single_ractor(void) {}
static inline void rb_zjit_tracing_invalidate_all(void) {}
static inline void rb_zjit_invalidate_newobj_hook(void) {}
static inline void rb_zjit_invalidate_no_singleton_class(VALUE klass) {}
static inline void rb_zjit_invalidate_root_box(void) {}
static inline void rb_zjit_jit_frame_update_references(zjit_jit_frame_t *jit_frame) {}
static inline void rb_zjit_materialize_frames(const rb_execution_context_t *ec, rb_control_frame_t *cfp) {}
static inline void rb_zjit_materialize_frames_for_longjmp(const rb_execution_context_t *ec, rb_control_frame_t *cfp) {}
static inline void rb_zjit_spill_frame(const rb_execution_context_t *ec, rb_control_frame_t *cfp) {}
static inline const zjit_jit_frame_t *CFP_ZJIT_FRAME(const rb_control_frame_t *cfp) { return NULL; }
#endif // #if USE_ZJIT

#define rb_zjit_enabled_p (rb_zjit_entry != 0)

// Return true if a given CFP has ZJIT's JITFrame.
static inline bool
CFP_ZJIT_FRAME_P(const rb_control_frame_t *cfp)
{
    if (!rb_zjit_enabled_p) return false;
    return cfp->jit_return != NULL;
}

static inline const VALUE*
CFP_PC(const rb_control_frame_t *cfp)
{
    if (CFP_ZJIT_FRAME_P(cfp)) {
        return CFP_ZJIT_FRAME(cfp)->pc;
    }
    return cfp->pc;
}

static inline const rb_iseq_t*
CFP_ISEQ(const rb_control_frame_t *cfp)
{
    if (CFP_ZJIT_FRAME_P(cfp)) {
        return CFP_ZJIT_FRAME(cfp)->iseq;
    }
    return cfp->_iseq;
}

#endif // #ifndef ZJIT_H
