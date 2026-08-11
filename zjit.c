#include "internal.h"
#include "internal/sanitizers.h"
#include "internal/string.h"
#include "internal/hash.h"
#include "internal/variable.h"
#include "internal/compile.h"
#include "internal/class.h"
#include "internal/fixnum.h"
#include "internal/numeric.h"
#include "internal/gc.h"
#include "internal/vm.h"
#include "yjit.h"
#include "vm_core.h"
#include "vm_callinfo.h"
#include "builtin.h"
#include "insns.inc"
#include "insns_info.inc"
#include "zjit.h"
#include "vm_insnhelper.h"
#include "probes.h"
#include "probes_helper.h"
#include "constant.h"
#include "iseq.h"
#include "ruby/debug.h"
#include "internal/cont.h"
#include "ractor_core.h"
#include "shape.h"

// This build config impacts the pointer tagging scheme and we only want to
// support one scheme for simplicity.
STATIC_ASSERT(pointer_tagging_scheme, USE_FLONUM);

enum zjit_struct_offsets {
    ISEQ_BODY_OFFSET_PARAM = offsetof(struct rb_iseq_constant_body, param),
    ISEQ_BODY_OFFSET_OUTER_VARIABLES = offsetof(struct rb_iseq_constant_body, outer_variables),
    RUBY_OFFSET_THREAD_RACTOR = offsetof(rb_thread_t, ractor),
};

// Struct offsets that cannot be constants in the checked-in bindgen output
// (zjit/src/cruby_bindings.inc.rs) because they vary with the build target
// and configuration. For example, offsetof(rb_ractor_t, newobj_cache) depends
// on the sizes of pthread types embedded in rb_ractor_t, which differ across
// architectures and OSes, as well as on VM_CHECK_MODE and RACTOR_CHECK_MODE.
// This table is filled out at C compile time and read by Rust at JIT compile
// time. Offsets that are identical on all supported builds should be added to
// enum zjit_struct_offsets above instead.
struct rb_zjit_runtime_offsets {
    int32_t ractor_newobj_cache;
    int32_t ractor_objspace;
};
const struct rb_zjit_runtime_offsets rb_zjit_runtime_offsets = {
    .ractor_newobj_cache = offsetof(rb_ractor_t, newobj_cache),
    .ractor_objspace = offsetof(rb_ractor_t, objspace),
};

// Special JITFrame used by all C method calls. We don't control the native
// stack layout for C frames, so cfp->jit_return points at this static frame
// via the ZJIT_JIT_RETURN_C_FRAME sentinel instead of a per-call allocation.
const zjit_jit_frame_t rb_zjit_c_frame = (zjit_jit_frame_t) {
    .pc = 0,
    .iseq = 0,
    .materialize_block_code = false,
};

void rb_zjit_profile_disable(const rb_iseq_t *iseq);
int rb_zjit_insn_to_bare_insn(int insn);

void
rb_zjit_compile_iseq(const rb_iseq_t *iseq, rb_execution_context_t *ec, bool jit_exception)
{
    RB_VM_LOCKING() {
        rb_vm_barrier();

        // Compile a block version starting at the current instruction
        uint8_t *rb_zjit_iseq_gen_entry_point(const rb_iseq_t *iseq, rb_execution_context_t *ec, bool jit_exception); // defined in Rust
        uintptr_t code_ptr = (uintptr_t)rb_zjit_iseq_gen_entry_point(iseq, ec, jit_exception);

        if (jit_exception) {
            ISEQ_BODY(iseq)->jit_exception = (rb_jit_func_t)code_ptr;
        }
        else {
            ISEQ_BODY(iseq)->jit_entry = (rb_jit_func_t)code_ptr;
        }
    }
}

extern VALUE *rb_vm_base_ptr(struct rb_control_frame_struct *cfp);

// Convert a given ISEQ's instructions to zjit_* instructions
void
rb_zjit_profile_enable(const rb_iseq_t *iseq)
{
    // This table encodes an opcode into the instruction's address
    const void *const *insn_table = rb_vm_get_insns_address_table();

    unsigned int insn_idx = 0;
    while (insn_idx < ISEQ_BODY(iseq)->iseq_size) {
        int insn = rb_vm_insn_addr2opcode((void *)ISEQ_BODY(iseq)->iseq_encoded[insn_idx]);
        int zjit_insn = vm_bare_insn_to_zjit_insn(insn);
        if (insn != zjit_insn) {
            ISEQ_BODY(iseq)->iseq_encoded[insn_idx] = (VALUE)insn_table[zjit_insn];
        }
        insn_idx += insn_len(insn);
    }
}

// Convert a given ISEQ's ZJIT instructions to bare instructions
void
rb_zjit_profile_disable(const rb_iseq_t *iseq)
{
    // This table encodes an opcode into the instruction's address
    const void *const *insn_table = rb_vm_get_insns_address_table();

    unsigned int insn_idx = 0;
    while (insn_idx < ISEQ_BODY(iseq)->iseq_size) {
        int insn = rb_vm_insn_addr2opcode((void *)ISEQ_BODY(iseq)->iseq_encoded[insn_idx]);
        int bare_insn = vm_zjit_insn_to_bare_insn(insn);
        if (insn != bare_insn) {
            ISEQ_BODY(iseq)->iseq_encoded[insn_idx] = (VALUE)insn_table[bare_insn];
        }
        insn_idx += insn_len(insn);
    }
}

// Map `zjit_* instructions back to their bare form. This is an identity function for all others.
int
rb_zjit_insn_to_bare_insn(int insn)
{
    return vm_zjit_insn_to_bare_insn(insn);
}

// Update a YARV instruction to a given opcode (to disable ZJIT profiling).
void
rb_zjit_iseq_insn_set(const rb_iseq_t *iseq, unsigned int insn_idx, enum ruby_vminsn_type bare_insn)
{
#if RUBY_DEBUG
    int insn = rb_vm_insn_addr2opcode((void *)ISEQ_BODY(iseq)->iseq_encoded[insn_idx]);
    RUBY_ASSERT(vm_zjit_insn_to_bare_insn(insn) == (int)bare_insn);
#endif
    const void *const *insn_table = rb_vm_get_insns_address_table();
    ISEQ_BODY(iseq)->iseq_encoded[insn_idx] = (VALUE)insn_table[bare_insn];
}

// Get profiling information for ISEQ
void *
rb_iseq_get_zjit_payload(const rb_iseq_t *iseq)
{
    RUBY_ASSERT_ALWAYS(IMEMO_TYPE_P(iseq, imemo_iseq));
    if (ISEQ_BODY(iseq)) {
        return ISEQ_BODY(iseq)->zjit_payload;
    }
    else {
        // Body is NULL when constructing the iseq.
        return NULL;
    }
}

// Set profiling information for ISEQ
void
rb_iseq_set_zjit_payload(const rb_iseq_t *iseq, void *payload)
{
    RUBY_ASSERT_ALWAYS(IMEMO_TYPE_P(iseq, imemo_iseq));
    RUBY_ASSERT_ALWAYS(ISEQ_BODY(iseq));
    RUBY_ASSERT_ALWAYS(NULL == ISEQ_BODY(iseq)->zjit_payload);
    ISEQ_BODY(iseq)->zjit_payload = payload;
}

void
rb_zjit_print_exception(void)
{
    VALUE exception = rb_errinfo();
    rb_set_errinfo(Qnil);
    assert(RTEST(exception));
    rb_warn("Ruby error: %"PRIsVALUE"", rb_funcall(exception, rb_intern("full_message"), 0));
}

bool
rb_zjit_singleton_class_p(VALUE klass)
{
    return RCLASS_SINGLETON_P(klass);
}

/* Sets all of the required shape flags for the object including the layout type,
 * the frozen status, and the slot size. Mimics `rb_newobj`.
 */
VALUE
rb_zjit_new_obj_shape(VALUE flags, size_t alloc_size)
{
    shape_id_t shape_id;
    switch (flags & T_MASK) {
      case T_OBJECT:
        shape_id = ROOT_SHAPE_ID;
        break;
      case T_STRUCT:
        shape_id = ROOT_SHAPE_ID | SHAPE_ID_LAYOUT_EXTENDED;
        break;
      case T_DATA:
        shape_id = ROOT_SHAPE_ID | SHAPE_ID_LAYOUT_RDATA;
        break;
      default:
        shape_id = ROOT_SHAPE_ID | SHAPE_ID_LAYOUT_OTHER;
        break;
    }

    if (flags & FL_FREEZE) {
        shape_id = rb_shape_transition_frozen(shape_id);
    }

    shape_id = rb_shape_transition_slot_size(shape_id, rb_gc_size_slot_size(alloc_size));

    return (flags & SHAPE_FLAG_MASK) | ((VALUE)shape_id << SHAPE_FLAG_SHIFT);
}

VALUE
rb_zjit_defined_ivar(VALUE obj, ID id, VALUE pushval)
{
    VALUE result = rb_ivar_defined(obj, id);
    return result ? pushval : Qnil;
}

bool
rb_zjit_method_tracing_currently_enabled(void)
{
    rb_event_flag_t tracing_events;
    if (rb_multi_ractor_p()) {
        tracing_events = ruby_vm_event_enabled_global_flags;
    }
    else {
        // At the time of writing, events are never removed from
        // ruby_vm_event_enabled_global_flags so always checking using it would
        // mean we don't compile even after tracing is disabled.
        tracing_events = rb_ec_ractor_hooks(GET_EC())->events;
    }

    return tracing_events & (RUBY_EVENT_C_CALL | RUBY_EVENT_C_RETURN);
}

// Check if any ISEQ trace events are currently enabled.
// Used to prevent ZJIT from compiling while tracing is active, since ZJIT's
// send fallback (rb_vm_opt_send_without_block) uses VM_EXEC which sets
// VM_FRAME_FLAG_FINISH on the callee frame, changing exception handling
// semantics for throw TAG_RETURN (e.g. return from rescue).
bool
rb_zjit_iseq_tracing_currently_enabled(void)
{
    rb_event_flag_t tracing_events;
    if (rb_multi_ractor_p()) {
        tracing_events = ruby_vm_event_enabled_global_flags;
    }
    else {
        tracing_events = rb_ec_ractor_hooks(GET_EC())->events;
    }

    return tracing_events & ISEQ_TRACE_EVENTS;
}

bool
rb_zjit_insn_leaf(int insn, const VALUE *opes)
{
    return insn_leaf(insn, opes);
}

ID
rb_zjit_local_id(const rb_iseq_t *iseq, unsigned idx)
{
    return ISEQ_BODY(iseq)->local_table[idx];
}

bool rb_zjit_cme_is_cfunc(const rb_callable_method_entry_t *me, const void *func);

const struct rb_callable_method_entry_struct *
rb_zjit_vm_search_method(VALUE cd_owner, struct rb_call_data *cd, VALUE recv);

bool
rb_zjit_class_initialized_p(VALUE klass)
{
    return RCLASS_INITIALIZED_P(klass);
}

rb_alloc_func_t rb_zjit_class_get_alloc_func(VALUE klass);

VALUE rb_class_allocate_instance(VALUE klass);

bool
rb_zjit_class_has_default_allocator(VALUE klass)
{
    assert(RCLASS_INITIALIZED_P(klass));
    assert(!RCLASS_SINGLETON_P(klass));
    rb_alloc_func_t alloc = rb_zjit_class_get_alloc_func(klass);
    return alloc == rb_class_allocate_instance;
}


// The class an ICLASS was included into.
VALUE
rb_zjit_iclass_includer(VALUE iclass)
{
    RUBY_ASSERT(RB_TYPE_P(iclass, T_ICLASS));
    return RCLASS_INCLUDER(iclass);
}

// The depth of `klass` in its superclass chain, as used by
// class_search_class_ancestor() for the constant-time `is_a?` check.
unsigned int
rb_zjit_class_superclass_depth(VALUE klass)
{
    RUBY_ASSERT(RB_TYPE_P(klass, T_CLASS));
    return (unsigned int)RCLASS_SUPERCLASS_DEPTH(klass);
}

struct zjit_override_search {
    ID mid;
    // Remaining number of classes we are willing to visit. Goes negative when
    // the hierarchy below the class is too large to scan.
    int budget;
    bool found;
};

// True if the segment of the ancestor chain owned by `klass` (its own method
// table plus any module included into or prepended to it) defines `mid`.
// The segment ends at the next T_CLASS, which is `klass`'s superclass.
static bool
zjit_class_segment_defines_method(VALUE klass, ID mid)
{
    VALUE p = klass;
    do {
        VALUE unused;
        struct rb_id_table *tbl = RCLASS_M_TBL(p);
        if (tbl && rb_id_table_lookup(tbl, mid, &unused)) return true;
        p = RCLASS_SUPER(p);
    } while (p && !RB_TYPE_P(p, T_CLASS));
    return false;
}

static void zjit_search_override_i(VALUE klass, VALUE data);

// Walk every class below `klass` looking for a definition of `mid`. Subclass
// lists only track T_CLASS -> T_CLASS links, so each visited class also has to
// account for the ICLASSes include/prepend inserted directly above it.
//
// Singleton classes are deliberately kept out of the subclass lists, so a
// `def obj.mid` on an instance of a class below `klass` is invisible here. The
// generated guard makes up for that by rejecting receivers whose class is a
// singleton class.
static bool
zjit_method_overridden_below(VALUE klass, struct zjit_override_search *search)
{
    rb_class_foreach_subclass(klass, zjit_search_override_i, (VALUE)search);
    return search->found || search->budget < 0;
}

static void
zjit_search_override_i(VALUE klass, VALUE data)
{
    struct zjit_override_search *search = (struct zjit_override_search *)data;
    if (search->found || search->budget < 0) return;
    search->budget--;
    if (search->budget < 0) return;
    // Refinement ICLASSes are reachable from a module's subclass list, but a
    // T_CLASS's list only ever holds T_CLASS entries.
    if (!RB_TYPE_P(klass, T_CLASS)) {
        search->found = true;
        return;
    }
    if (zjit_class_segment_defines_method(klass, search->mid)) {
        search->found = true;
        return;
    }
    zjit_method_overridden_below(klass, search);
}

// True when no class below `klass` overrides `mid`, so `mid` resolves to the
// same method entry for every instance of `klass` and of its subclasses.
// Conservatively returns false when the hierarchy is larger than `budget`
// classes or when we cannot enumerate part of it.
bool
rb_zjit_no_method_override_below(VALUE klass, ID mid, unsigned int budget)
{
    if (!RB_TYPE_P(klass, T_CLASS)) return false;
    if (RCLASS_SINGLETON_P(klass)) return false;

    bool result;
    RB_VM_LOCKING() {
        struct zjit_override_search search = { .mid = mid, .budget = (int)budget, .found = false };
        result = !zjit_method_overridden_below(klass, &search);
    }
    return result;
}

VALUE rb_vm_untag_block_handler(VALUE block_handler);
VALUE rb_vm_get_untagged_block_handler(rb_control_frame_t *reg_cfp);

// Primitives used by zjit.rb. Don't put other functions below, which wouldn't use them.
VALUE rb_zjit_enable(rb_execution_context_t *ec, VALUE self);
VALUE rb_zjit_assert_compiles(rb_execution_context_t *ec, VALUE self);
VALUE rb_zjit_stats(rb_execution_context_t *ec, VALUE self, VALUE target_key);
VALUE rb_zjit_reset_stats_bang(rb_execution_context_t *ec, VALUE self);
VALUE rb_zjit_stats_enabled_p(rb_execution_context_t *ec, VALUE self);
VALUE rb_zjit_print_stats_p(rb_execution_context_t *ec, VALUE self);
VALUE rb_zjit_get_stats_file_path_p(rb_execution_context_t *ec, VALUE self);
VALUE rb_zjit_trace_exit_locations_enabled_p(rb_execution_context_t *ec, VALUE self);
VALUE rb_zjit_get_exit_locations(rb_execution_context_t *ec, VALUE self);

// Preprocessed zjit.rb generated during build
#include "zjit.rbinc"
