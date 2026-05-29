#ifndef ZJIT_H
#define ZJIT_H 1
//
// This file contains definitions ZJIT exposes to the CRuby codebase
//

// ZJIT_STATS controls whether to support runtime counters in the interpreter
#ifndef ZJIT_STATS
# define ZJIT_STATS (USE_ZJIT && RUBY_DEBUG)
#endif

typedef enum zjit_opnd_type {
    ZJIT_OPND_VALUE,
    ZJIT_OPND_VREG,
} zjit_opnd_type_t;

typedef struct zjit_opnd {
    zjit_opnd_type_t type;
    union {
        VALUE value;
        size_t vreg_stack_index;
    } as;
} zjit_opnd_t;

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

    uint32_t stack_size;
    zjit_opnd_t *stack;
} zjit_jit_frame_t;

#if USE_ZJIT
extern void *rb_zjit_entry;
extern const zjit_jit_frame_t rb_zjit_c_frame;
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
void rb_zjit_materialize_frames(const rb_execution_context_t *ec, rb_control_frame_t *cfp);

// Special value for cfp->jit_return that means "this is a C method frame, use
// rb_zjit_c_frame as the JITFrame". We don't control the native stack layout
// for C frames, so there's no per-call JITFrame storage; we set this sentinel
// instead of a heap-allocated JITFrame pointer.
#ifndef ZJIT_JIT_RETURN_C_FRAME
# define ZJIT_JIT_RETURN_C_FRAME 0x1
#endif
// Temporary value used between gen_push_frame() and the callee entry point.
// The callee entry point replaces it with NATIVE_BASE_PTR after copying the
// staged env metadata into native-frame slots.
#ifndef ZJIT_JIT_RETURN_DEFERRED_ENV
# define ZJIT_JIT_RETURN_DEFERRED_ENV 0x3
#endif

// BADFrame. The high bit is set, so likely SEGV on linux and darwin if dereferenced.
#define ZJIT_JIT_RETURN_POISON 0xbadfbadfbadfbadfULL

#ifndef ZJIT_JIT_FRAME_INDEX_FRAME
# define ZJIT_JIT_FRAME_INDEX_FRAME          (-1)
# define ZJIT_JIT_FRAME_INDEX_CME           (-2)
# define ZJIT_JIT_FRAME_INDEX_SPECVAL       (-3)
# define ZJIT_JIT_FRAME_INDEX_FLAGS         (-4)
# define ZJIT_JIT_FRAME_INDEX_ENV_MATERIALIZED (-5)
#endif

static inline const zjit_jit_frame_t *
CFP_ZJIT_FRAME(const rb_control_frame_t *cfp)
{
    if ((VALUE)cfp->jit_return == ZJIT_JIT_RETURN_C_FRAME) {
        return &rb_zjit_c_frame;
    }
    else {
        RUBY_ASSERT((VALUE)cfp->jit_return != ZJIT_JIT_RETURN_DEFERRED_ENV);
#if USE_ZJIT
        RUBY_ASSERT((unsigned long long)((VALUE *)cfp->jit_return)[ZJIT_JIT_FRAME_INDEX_FRAME] != ZJIT_JIT_RETURN_POISON);
#endif
        // Read JITFrame from the stack slot. gen_entry_point() writes an initial
        // frame describing the entry PC + iseq; subsequent gen_save_pc_for_gc()
        // calls update it with a more accurate PC before any non-leaf C call.
        return (const zjit_jit_frame_t *)((VALUE *)cfp->jit_return)[ZJIT_JIT_FRAME_INDEX_FRAME];
    }
}
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
static inline void rb_zjit_materialize_frames(const rb_execution_context_t *ec, rb_control_frame_t *cfp) {}
static inline const zjit_jit_frame_t *CFP_ZJIT_FRAME(const rb_control_frame_t *cfp) { return NULL; }
#endif // #if USE_ZJIT

#define rb_zjit_enabled_p (rb_zjit_entry != 0)

// Return true if a given CFP has ZJIT's JITFrame.
static inline bool
CFP_ZJIT_FRAME_P(const rb_control_frame_t *cfp)
{
    if (!rb_zjit_enabled_p) return false;
    return cfp->jit_return != NULL && (VALUE)cfp->jit_return != ZJIT_JIT_RETURN_DEFERRED_ENV;
}

static inline bool
CFP_ZJIT_FRAME_NATIVE_ENV_P(const rb_control_frame_t *cfp)
{
    return CFP_ZJIT_FRAME_P(cfp) && (VALUE)cfp->jit_return != ZJIT_JIT_RETURN_C_FRAME;
}

static inline bool
CFP_ZJIT_FRAME_ENV_MATERIALIZED_P(const rb_control_frame_t *cfp)
{
    if (!CFP_ZJIT_FRAME_NATIVE_ENV_P(cfp)) return true;
    return ((VALUE *)cfp->jit_return)[ZJIT_JIT_FRAME_INDEX_ENV_MATERIALIZED] != 0;
}

static inline bool
CFP_ZJIT_FRAME_UNMATERIALIZED_ENV_P(const rb_control_frame_t *cfp)
{
    return CFP_ZJIT_FRAME_NATIVE_ENV_P(cfp) && !CFP_ZJIT_FRAME_ENV_MATERIALIZED_P(cfp);
}

static inline VALUE
CFP_ZJIT_FRAME_CME(const rb_control_frame_t *cfp)
{
    if (CFP_ZJIT_FRAME_UNMATERIALIZED_ENV_P(cfp)) {
        return ((VALUE *)cfp->jit_return)[ZJIT_JIT_FRAME_INDEX_CME];
    }
    return cfp->ep[VM_ENV_DATA_INDEX_ME_CREF];
}

static inline VALUE
CFP_ZJIT_FRAME_SPECVAL(const rb_control_frame_t *cfp)
{
    if (CFP_ZJIT_FRAME_UNMATERIALIZED_ENV_P(cfp)) {
        return ((VALUE *)cfp->jit_return)[ZJIT_JIT_FRAME_INDEX_SPECVAL];
    }
    return cfp->ep[VM_ENV_DATA_INDEX_SPECVAL];
}

static inline VALUE
CFP_ZJIT_FRAME_FLAGS(const rb_control_frame_t *cfp)
{
    return cfp->ep[VM_ENV_DATA_INDEX_FLAGS];
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
