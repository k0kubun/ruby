#ifndef ZJIT_H
#define ZJIT_H 1
//
// This file contains definitions ZJIT exposes to the CRuby codebase
//

// ZJIT_STATS controls whether to support runtime counters in the interpreter
#ifndef ZJIT_STATS
# define ZJIT_STATS (USE_ZJIT && RUBY_DEBUG)
#endif

// Maximum number of stack-map entries per array (stack or locals).
// Tripping this cap asserts in gen_prepare_non_leaf_call; raise if needed.
#define ZJIT_STACK_MAP_CAP 32

// How to materialize one stack-map entry back onto the VM stack.
enum zjit_stack_map_kind {
    ZJIT_SME_NONE   = 0,  // unused slot
    ZJIT_SME_VALUE  = 1,  // .value holds the VALUE (Opnd::Value/Imm/UImm)
    ZJIT_SME_CSTACK = 2,  // .payload32 = byte offset from jit_frame->saved_sp
    ZJIT_SME_SPILL  = 3,  // .payload32 = byte disp from jit_frame->saved_fp
};

typedef struct zjit_stack_map_entry {
    uint32_t kind;       // enum zjit_stack_map_kind
    int32_t  payload32;  // CSTACK offset or SPILL disp
    VALUE    value;      // VALUE for SME_VALUE; ignored otherwise
} zjit_stack_map_entry_t;

// JITFrame is defined here as the single source of truth and imported into
// Rust via bindgen. C code reads fields directly; Rust uses an impl block.
typedef struct zjit_jit_frame {
    // Program counter for this frame, used for backtraces and GC.
    // NULL for C frames (they don't have a Ruby PC).
    const VALUE *pc;
    // The ISEQ this frame belongs to. Marked via rb_execution_context_mark.
    // NULL for C frames.
    const rb_iseq_t *iseq;
    // Whether to materialize block_code when this frame is materialized.
    // True when the ISEQ doesn't contain send/invokesuper/invokeblock
    // (which write block_code themselves), so we must restore it.
    // Always false for C frames.
    bool materialize_block_code;

    // Stack map: NATIVE_STACK_PTR and NATIVE_BASE_PTR snapshots taken just
    // before each non-leaf CCall, plus encoded entries describing where the
    // HIR stack and local values live at that call site.
    void *saved_sp;
    void *saved_fp;
    uint8_t stack_len;
    uint8_t locals_len;
    zjit_stack_map_entry_t stack[ZJIT_STACK_MAP_CAP];
    zjit_stack_map_entry_t locals[ZJIT_STACK_MAP_CAP];
} zjit_jit_frame_t;

#if USE_ZJIT
extern void *rb_zjit_entry;
extern uint64_t rb_zjit_call_threshold;
extern uint64_t rb_zjit_profile_threshold;
void rb_zjit_compile_iseq(const rb_iseq_t *iseq, rb_execution_context_t *ec, bool jit_exception);
void rb_zjit_profile_insn(uint32_t insn, rb_execution_context_t *ec);
void rb_zjit_profile_enable(const rb_iseq_t *iseq);
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
void rb_zjit_before_ractor_spawn(void);
void rb_zjit_tracing_invalidate_all(void);
void rb_zjit_invalidate_no_singleton_class(VALUE klass);
void rb_zjit_invalidate_root_box(void);
void rb_zjit_jit_frame_update_references(zjit_jit_frame_t *jit_frame);
void rb_zjit_materialize_frames_for_gc(rb_control_frame_t *cfp);
#else
#define rb_zjit_entry 0
static inline void rb_zjit_compile_iseq(const rb_iseq_t *iseq, rb_execution_context_t *ec, bool jit_exception) {}
static inline void rb_zjit_profile_insn(uint32_t insn, rb_execution_context_t *ec) {}
static inline void rb_zjit_profile_enable(const rb_iseq_t *iseq) {}
static inline void rb_zjit_bop_redefined(int redefined_flag, enum ruby_basic_operators bop) {}
static inline void rb_zjit_cme_invalidate(const rb_callable_method_entry_t *cme) {}
static inline void rb_zjit_invalidate_no_ep_escape(const rb_iseq_t *iseq) {}
static inline void rb_zjit_constant_state_changed(ID id) {}
static inline void rb_zjit_before_ractor_spawn(void) {}
static inline void rb_zjit_tracing_invalidate_all(void) {}
static inline void rb_zjit_invalidate_no_singleton_class(VALUE klass) {}
static inline void rb_zjit_invalidate_root_box(void) {}
static inline void rb_zjit_jit_frame_update_references(zjit_jit_frame_t *jit_frame) {}
static inline void rb_zjit_materialize_frames_for_gc(rb_control_frame_t *cfp) {}
#endif // #if USE_ZJIT

#define rb_zjit_enabled_p (rb_zjit_entry != 0)

// BADFrame. The high bit is set, so likely SEGV on linux and darwin if dereferenced.
#define ZJIT_JIT_RETURN_POISON 0xbadfbadfbadfbadfULL

// Return the JITFrame pointer from cfp->jit_return, or NULL if not present.
// YJIT also uses jit_return (as a return address), so this must only return
// non-NULL when ZJIT is enabled and has set jit_return to a JITFrame pointer.
static inline void *
CFP_ZJIT_FRAME(const rb_control_frame_t *cfp)
{
    if (!rb_zjit_enabled_p) return NULL;
#if USE_ZJIT
    RUBY_ASSERT((unsigned long long)cfp->jit_return != ZJIT_JIT_RETURN_POISON);
#endif
    return cfp->jit_return;
}

static inline const VALUE*
CFP_PC(const rb_control_frame_t *cfp)
{
    if (CFP_ZJIT_FRAME(cfp)) {
        return ((const zjit_jit_frame_t *)cfp->jit_return)->pc;
    }
    return cfp->pc;
}

static inline const rb_iseq_t*
CFP_ISEQ(const rb_control_frame_t *cfp)
{
    if (CFP_ZJIT_FRAME(cfp)) {
        return ((const zjit_jit_frame_t *)cfp->jit_return)->iseq;
    }
    return cfp->_iseq;
}

#endif // #ifndef ZJIT_H
