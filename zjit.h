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
    // The ISEQ this frame belongs to. Marked via rb_execution_context_mark.
    // NULL for C frames.
    const rb_iseq_t *iseq;
    // Whether to materialize block_code when this frame is materialized.
    // True when the ISEQ doesn't contain send/invokesuper/invokeblock
    // (which write block_code themselves), so we must restore it.
    // Always false for C frames.
    bool materialize_block_code;

    // Number of stack map entries in stack[].
    uint32_t stack_size;
    // Flexible array of stack map entries, executed in order by
    // zjit_materialize_frames(). See the ZJIT_STACK_MAP_* opcodes above.
    VALUE stack[];
} zjit_jit_frame_t;

#if USE_ZJIT
// Stack map entries are opcodes for zjit_materialize_frames(), which walks them
// in order while moving a cursor down the VM stack. An untagged entry is an
// immediate Ruby VALUE to store; the tagged forms below copy from the native
// stack, skip slots, or move the cursor. Stack maps never contain heap VALUEs,
// so these tags are available: they are not Qfalse (0), and their low 3 bits
// are zero, so RB_SPECIAL_CONST_P is false. Tags must stay non-zero multiples
// of 8 for that to hold.
#define ZJIT_STACK_MAP_VREG_TAG 0x08
#define ZJIT_STACK_MAP_SKIP_TAG 0x10
#define ZJIT_STACK_MAP_BASE_PTR_TAG 0x18
#define ZJIT_STACK_MAP_TAG_MASK 0xff
#define ZJIT_STACK_MAP_SHIFT 8

// The BASE_PTR payload packs two fields above the tag byte: the slot index in
// bits 8..=31 and the operand stack size in bits 32..=63.
#define ZJIT_STACK_MAP_BASE_PTR_SIZE_SHIFT 32
#define ZJIT_STACK_MAP_BASE_PTR_INDEX_MASK 0xffffff

static inline bool
ZJIT_STACK_MAP_VREG_P(VALUE entry)
{
    return (entry & ZJIT_STACK_MAP_TAG_MASK) == ZJIT_STACK_MAP_VREG_TAG;
}

static inline size_t
ZJIT_STACK_MAP_VREG_INDEX(VALUE entry)
{
    return entry >> ZJIT_STACK_MAP_SHIFT;
}

static inline bool
ZJIT_STACK_MAP_SKIP_P(VALUE entry)
{
    return (entry & ZJIT_STACK_MAP_TAG_MASK) == ZJIT_STACK_MAP_SKIP_TAG;
}

static inline size_t
ZJIT_STACK_MAP_SKIP_SIZE(VALUE entry)
{
    return entry >> ZJIT_STACK_MAP_SHIFT;
}

// Anchor the write cursor using the SP register the JIT saved on its native
// stack, instead of cfp->sp. cfp->sp is not a reliable starting point for a
// frame that is in the middle of a non-leaf C call, as e.g. raising
// ArgumentError can push through and move cfp->sp, then use the stack map.
// gen_prepare_non_leaf_call() emits this opcode, always as stack[0], so the
// entries after decode to the right place.
static inline bool
ZJIT_STACK_MAP_BASE_PTR_P(VALUE entry)
{
    return (entry & ZJIT_STACK_MAP_TAG_MASK) == ZJIT_STACK_MAP_BASE_PTR_TAG;
}

// VALUE index from cfp->jit_return down to the native stack slot holding the
// saved SP register, i.e. base_ptr is `((VALUE **)cfp->jit_return)[-index]`.
// There is one such slot per compiled function, so the index depends on the
// frame's inlining depth.
static inline size_t
ZJIT_STACK_MAP_BASE_PTR_SLOT_INDEX(VALUE entry)
{
    return (entry >> ZJIT_STACK_MAP_SHIFT) & ZJIT_STACK_MAP_BASE_PTR_INDEX_MASK;
}

// Number of VM stack slots above base_ptr, i.e. the frame's operand stack
// depth. The cursor is set to `base_ptr + this`.
static inline size_t
ZJIT_STACK_MAP_BASE_PTR_STACK_SIZE(VALUE entry)
{
    return entry >> ZJIT_STACK_MAP_BASE_PTR_SIZE_SHIFT;
}

// Class -> callcache table for a send site that dispatches over too many
// classes for ZJIT's inline class-guard chain. Allocated and owned by Rust; the
// layout of these fields must stay in step with `struct SendCache` in
// zjit/src/send_cache.rs, which documents what the table caches and why a stale
// entry cannot be wrong.
//
// One slot of the table. Two words so that a hit can answer both questions a
// megamorphic send asks: which callcache dispatches this class (`cc`), and, when
// the answer is an ISEQ method JIT code can enter without the interpreter,
// which method that is (`direct_cme`). See `gen_send_megamorphic_direct()` in
// zjit/src/codegen.rs for the inline probe that reads both.
struct rb_zjit_send_cache_entry {
    // The cached callcache, or NULL when the slot is empty. Validates itself:
    // see zjit_send_cache_search().
    const struct rb_callcache *cc;
    // `vm_cc_cme(cc)` when that method is one JIT code may enter with a direct
    // call (see zjit_send_cache_direct_cme()), NULL otherwise. JIT code checks
    // it against the method entry it read out of `cc`, so a slot caught
    // half-written by another ractor reads as a miss rather than as a call to
    // the wrong method -- which is what lets both words be plain stores.
    const rb_callable_method_entry_t *direct_cme;
};

struct rb_zjit_send_cache {
    // Number of slots, and what zjit_send_cache_slot() scales the hash by. A
    // power of two.
    uint32_t len;
    // Evictions since the table was allocated or last grown. Counted here rather
    // than in Rust because the fill path below still has the evicted slot in
    // hand; only meaningful while `grow_at` is non-zero.
    uint32_t evictions;
    // Slot 0. Moves when the table grows, which is why nothing may hold it
    // across a call that could reach rb_zjit_send_cache_grow().
    struct rb_zjit_send_cache_entry *slots;
    // The ZJIT hit counter under --zjit-stats, NULL otherwise. Doubles as the
    // flag for whether to report misses to rb_zjit_send_cache_record_miss(), so
    // that a build without stats pays a never-taken branch rather than a call.
    uint64_t *hit_counter;
    // `argc` of the call shape this table serves, and whether that shape lets a
    // hit be dispatched with a direct JIT-to-JIT call at all. Both are constant
    // for the life of the table; Rust sets them when it allocates it.
    uint32_t direct_argc;
    // Zero when the call shape rules direct dispatch out (a splat, a block
    // argument, a tail call, ...), so `direct_cme` stays NULL in every slot.
    uint32_t direct_ok;
    // The call shape's `vm_ci_flag()`, for the visibility test the fill path
    // runs (a private method is directly callable only from an FCALL site).
    uint32_t direct_flags;
    // `evictions` at which this table should grow, or 0 for one that will not
    // grow again. See SendCache::grow in zjit/src/send_cache.rs.
    uint32_t grow_at;
};

// Why a probe of a `struct rb_zjit_send_cache` did not produce a callcache.
// Must match the MISS_* constants in zjit/src/send_cache.rs.
#define ZJIT_SEND_CACHE_MISS_FILL        0
#define ZJIT_SEND_CACHE_MISS_EVICT       1
#define ZJIT_SEND_CACHE_MISS_STALE       2
#define ZJIT_SEND_CACHE_MISS_UNCACHEABLE 3

// Multiplier of the Fibonacci hash that turns a class VALUE into a slot index.
// Must match SEND_CACHE_HASH_MULT in zjit/src/send_cache.rs.
#define ZJIT_SEND_CACHE_HASH_MULT 0x9e3779b97f4a7c15ULL

void rb_zjit_send_cache_record_miss(int kind);
void rb_zjit_send_cache_grow(struct rb_zjit_send_cache *cache);

// Field offsets and flag masks the inline send-cache probe in JIT code needs.
// They are functions rather than bindgen constants because the structs they
// reach into (rb_callcache, rb_iseq_constant_body) are opaque to Rust.
size_t rb_zjit_cc_klass_offset(void);
size_t rb_zjit_cc_cme_offset(void);
size_t rb_zjit_iseq_body_offset(void);
size_t rb_zjit_iseq_body_jit_entry_offset(void);
size_t rb_zjit_send_cache_entry_size(void);
size_t rb_zjit_send_cache_entry_direct_cme_offset(void);
size_t rb_zjit_cme_def_offset(void);
size_t rb_zjit_def_iseqptr_offset(void);
VALUE rb_zjit_method_entry_invalidated_flag(void);
size_t rb_zjit_mega_direct_max_stack(void);

// Field offsets and flag masks the inline block dispatch in JIT code needs to
// decide, from a run-time block ISEQ, whether it may push the block frame
// itself. Same reason as above: rb_iseq_constant_body is opaque to Rust, and
// the `param.flags` bitfields have no layout Rust could reproduce.
size_t rb_zjit_iseq_body_param_flags_offset(void);
size_t rb_zjit_iseq_body_param_lead_num_offset(void);
size_t rb_zjit_iseq_body_local_table_size_offset(void);
size_t rb_zjit_iseq_body_stack_max_offset(void);
uint32_t rb_zjit_iseq_param_flags_not_simple_mask(void);
uint32_t rb_zjit_iseq_param_flags_ambiguous_param0_mask(void);

// Largest `stack_max` a callee may have and still be entered by the inline
// megamorphic dispatch path, which checks for stack overflow against this bound
// instead of the callee's own (unknown at compile time) requirement.
#define ZJIT_MEGA_DIRECT_MAX_STACK 64

extern void *rb_zjit_entry;
extern const zjit_jit_frame_t rb_zjit_c_frame;
extern uint64_t rb_zjit_call_threshold;
extern uint64_t rb_zjit_profile_threshold;
void rb_zjit_compile_iseq(const rb_iseq_t *iseq, rb_execution_context_t *ec, bool jit_exception);
void rb_zjit_profile_insn(uint32_t insn, rb_execution_context_t *ec);
void rb_zjit_profile_enable(const rb_iseq_t *iseq);
void rb_zjit_bop_redefined(int redefined_flag, enum ruby_basic_operators bop);
void rb_zjit_cme_invalidate(const rb_callable_method_entry_t *cme);
void rb_zjit_method_lookup_changed(ID mid);
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
size_t rb_zjit_hash_new_size(VALUE *flags_out, size_t size);
VALUE rb_zjit_new_obj_shape(VALUE flags, size_t alloc_size);
bool rb_zjit_class_allocate_instance_fastpath(VALUE klass, size_t *size_out, VALUE *flags_out);
bool rb_zjit_str_resurrect_fastpath(VALUE str, bool chilled, size_t *size_out, VALUE *flags_out, long *len_out, size_t *byte_size_out);
bool rb_zjit_array_dup_can_fastpath(VALUE ary, size_t *alloc_size_out, VALUE *flags_out, long *len_out);
bool rb_zjit_array_new_can_fastpath(long len, size_t *alloc_size_out, VALUE *flags_out);
bool rb_zjit_hash_dup_can_fastpath(VALUE hash, size_t *alloc_size_out, VALUE *flags_out, VALUE *ifnone_out, long *bound_out);
void rb_zjit_range_new_fastpath(bool exclude_end, size_t *alloc_size_out, VALUE *flags_out);
void rb_zjit_array_new_fastpath(size_t *alloc_size_out, VALUE *flags_out);
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
static inline void rb_zjit_compile_iseq(const rb_iseq_t *iseq, rb_execution_context_t *ec, bool jit_exception) {}
static inline void rb_zjit_profile_insn(uint32_t insn, rb_execution_context_t *ec) {}
static inline void rb_zjit_profile_enable(const rb_iseq_t *iseq) {}
static inline void rb_zjit_bop_redefined(int redefined_flag, enum ruby_basic_operators bop) {}
static inline void rb_zjit_cme_invalidate(const rb_callable_method_entry_t *cme) {}
static inline void rb_zjit_method_lookup_changed(ID mid) {}
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
