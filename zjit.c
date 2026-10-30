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
#include "internal/jit.h"
#include "ractor_core.h"
#include "shape.h"
#include "eval_intern.h"

#ifndef _WIN32
#include <sys/mman.h>
#endif

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

#if !defined(_WIN32) && defined(MAP_ANONYMOUS)
uint8_t *rb_jit_align_ptr(uint8_t *ptr, uint32_t multiple); // defined in jit.c

// Reserve address space that lives entirely below INT32_MAX for JITFrame.
//
// When a JITFrame pointer fits in 32 bits, x86_64 can encode the store
// as `mov qword ptr [mem], imm32` (8 bytes) instead of `movabs` + a store,
// and arm64 materializes it in two instructions instead of four.
//
// Like rb_jit_reserve_addr_space in jit.c, this only reserves address space (PROT_NONE).
// VirtualMem is in charge of mapping physical memory into the reserved space page by page.
void *
rb_zjit_reserve_low_addr_space(size_t size)
{
    void *mem_block = MAP_FAILED;

    // Linux (x86_64): Use MAP_32BIT to map within the first 2GiB of address space.
    // This works only for x86_64, and the kernel restricts it to [1GiB, 2GiB).
  #ifdef MAP_32BIT
    mem_block = mmap(NULL, size, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
  #endif

    // Linux (all arch): Probe a free hole below 2GiB if MAP_32BIT is not possible.
    // MAP_FIXED_NOREPLACE fails rather than clobbering an existing mapping.
  #if defined(MAP_FIXED_NOREPLACE) && defined(_SC_PAGESIZE)
    if (mem_block == MAP_FAILED) {
        // Distance between probes. 64MiB sweeps the usable 2GiB in at most 32 mmap calls.
        const uintptr_t probe_stride = 64 * 1024 * 1024;
        const uint32_t page_size = (uint32_t)sysconf(_SC_PAGESIZE);
        const uintptr_t limit = (uintptr_t)INT32_MAX - size;
        for (uintptr_t addr = probe_stride; addr < limit; addr += probe_stride) {
            // mmap only honors a hint that is page-aligned.
            void *req = rb_jit_align_ptr((uint8_t *)addr, page_size);
            mem_block = mmap(req, size, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (mem_block != MAP_FAILED) break;
        }
    }
  #endif

    if (mem_block == MAP_FAILED) return NULL;

    // Both MAP_32BIT and MAP_FIXED_NOREPLACE are advisory in some platforms, e.g.
    // sandboxes or older kernels. Fallback to normal allocation if it doesn't work.
    if ((uintptr_t)mem_block + size > (uintptr_t)INT32_MAX) {
        munmap(mem_block, size);
        return NULL;
    }
    ruby_annotate_mmap(mem_block, size, "Ruby:rb_zjit_reserve_low_addr_space");
    return mem_block;
}

#else

// Windows not supported for now
void *rb_zjit_reserve_low_addr_space(size_t size) { return NULL; }

#endif

static VALUE
rb_zjit_total_compile_time_ns_get(rb_execution_context_t *ec, VALUE self)
{
    return ULL2NUM(rb_zjit_total_compile_time_ns);
}

static VALUE
rb_zjit_max_compile_time_ns_get(rb_execution_context_t *ec, VALUE self)
{
    return ULL2NUM(rb_zjit_max_compile_time_ns);
}

void rb_zjit_update_max_compile_time_ns(uint64_t max_compile_time_ns);

static VALUE
rb_zjit_max_compile_time_ns_set(rb_execution_context_t *ec, VALUE self, VALUE max_compile_time_ns)
{
    rb_zjit_update_max_compile_time_ns(NUM2ULL(max_compile_time_ns));
    return max_compile_time_ns;
}

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

// This is used by a function stub to install compiled code as the ISEQ's entry point.
void
rb_zjit_iseq_set_jit_entry(const rb_iseq_t *iseq, void *code_ptr)
{
    ISEQ_BODY(iseq)->jit_entry = (rb_jit_func_t)code_ptr;
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

// Return false if a function stub has not collected enough profiles yet, enabling
// profiling instructions as needed. Return true once enough profiles are collected.
bool
rb_zjit_iseq_has_profiled_enough(const rb_iseq_t *iseq)
{
    struct rb_iseq_constant_body *body = ISEQ_BODY(iseq);

    if (body->jit_entry_calls < rb_zjit_profile_threshold) {
        // Skip the unprofiled warmup. The compiled caller already establishes
        // that the callee is hot, so go straight to the profiling window.
        body->jit_entry_calls = rb_zjit_profile_threshold;
        rb_zjit_profile_enable(iseq);
    }
    else {
        body->jit_entry_calls++;
    }

    return body->jit_entry_calls >= rb_zjit_call_threshold;
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

rb_alloc_func_t rb_zjit_class_get_alloc_func(VALUE klass);

// Defined in struct.c, where struct_alloc and the members ivar are visible.
bool rb_zjit_class_has_struct_allocator(VALUE klass);
rb_len_t rb_zjit_struct_num_members(VALUE klass);
ID rb_zjit_struct_member_id(VALUE klass, rb_len_t index);
bool rb_zjit_struct_embedded_p(rb_len_t num_members);
VALUE rb_struct_s_keyword_init(VALUE klass);

VALUE rb_class_allocate_instance(VALUE klass);

bool
rb_zjit_class_has_default_allocator(VALUE klass)
{
    assert(RCLASS_INITIALIZED_P(klass));
    assert(!RCLASS_SINGLETON_P(klass));
    rb_alloc_func_t alloc = rb_zjit_class_get_alloc_func(klass);
    return alloc == rb_class_allocate_instance;
}

bool
rb_zjit_callable_method_entry_is_undefined(const rb_callable_method_entry_t *cme)
{
    return UNDEFINED_METHOD_ENTRY_P(cme);
}

VALUE rb_vm_get_untagged_block_handler(rb_control_frame_t *reg_cfp);
bool rb_vm_once_done_value(ISE is, VALUE *result);

bool
rb_zjit_array_aref_with_adjusted_index(VALUE ary, rb_len_t index, VALUE *out)
{
    RUBY_ASSERT(RB_TYPE_P(ary, T_ARRAY));
    if (index >= 0 && index < RARRAY_LEN(ary)) {
        *out = RARRAY_AREF(ary, index);
        return true;
    }
    else {
        return false;
    }
}

rb_cref_t *rb_vm_get_cref(const VALUE *ep);

// Return the lexical scope (cref) of an ISEQ frame. ZJIT records it with folded
// constant paths so that it can re-resolve them in rb_zjit_constant_path_resolves_to().
const rb_cref_t *
rb_zjit_frame_cref(const rb_control_frame_t *cfp)
{
    return rb_vm_get_cref(cfp->ep);
}

// Return the lexical scope (cref) a method entry runs with, or NULL if unknown.
const rb_cref_t *
rb_zjit_cme_cref(const rb_callable_method_entry_t *cme)
{
    if (cme && cme->def && cme->def->type == VM_METHOD_TYPE_ISEQ) {
        return cme->def->body.iseq.cref;
    }
    return NULL;
}

// Result of rb_zjit_constant_path_resolves_to()
enum zjit_const_resolution {
    ZJIT_CONST_SAME,
    ZJIT_CONST_DIFFERENT,
    ZJIT_CONST_UNKNOWN,
};

enum zjit_const_lookup {
    ZJIT_CONST_LOOKUP_FOUND,
    ZJIT_CONST_LOOKUP_MISS,
    ZJIT_CONST_LOOKUP_UNKNOWN,
};

// Side-effect-free mirror of rb_const_search_from(). Never autoloads, never warns.
static enum zjit_const_lookup
zjit_const_search_from(VALUE klass, ID id, bool exclude, bool visibility, VALUE *value)
{
    bool first_iteration = true;
    for (VALUE current = klass; RTEST(current); current = RCLASS_SUPER(current), first_iteration = false) {
        if (!first_iteration && RCLASS_ORIGIN(current) != current) continue;

        VALUE tmp = current;
        if (BUILTIN_TYPE(tmp) == T_ICLASS) {
            tmp = RBASIC(tmp)->klass;
            // A module with prepended modules is included through several iclasses whose
            // origins are fixed up only at the end of do_include_modules_at(), which
            // invalidates constants in the middle of it. Don't try to be precise there.
            if (RB_TYPE_P(tmp, T_MODULE) && RCLASS_ORIGIN(tmp) != tmp) return ZJIT_CONST_LOOKUP_UNKNOWN;
        }

        rb_const_entry_t *ce = rb_const_lookup(tmp, id);
        if (ce) {
            if (visibility && RB_CONST_PRIVATE_P(ce)) return ZJIT_CONST_LOOKUP_MISS;
            if (UNDEF_P(ce->value)) return ZJIT_CONST_LOOKUP_UNKNOWN; // autoload
            if (exclude && tmp == rb_cObject) return ZJIT_CONST_LOOKUP_MISS;
            *value = ce->value;
            return ZJIT_CONST_LOOKUP_FOUND;
        }
    }
    return ZJIT_CONST_LOOKUP_MISS;
}

// Side-effect-free mirror of rb_const_search()
static enum zjit_const_lookup
zjit_const_search(VALUE klass, ID id, bool exclude, bool visibility, VALUE *value)
{
    if (klass == rb_cObject) exclude = false;
    enum zjit_const_lookup result = zjit_const_search_from(klass, id, exclude, visibility, value);
    if (result != ZJIT_CONST_LOOKUP_MISS || exclude || BUILTIN_TYPE(klass) != T_MODULE) return result;
    return zjit_const_search_from(rb_cObject, id, false, visibility, value);
}

static bool
zjit_namespace_p(VALUE klass)
{
    return !SPECIAL_CONST_P(klass) && (BUILTIN_TYPE(klass) == T_CLASS || BUILTIN_TYPE(klass) == T_MODULE);
}

// Side-effect-free mirror of the lexical scope search in vm_get_ev_const(), for a cref chain
// with no CREF_PUSHED_BY_EVAL crefs (checked by the caller)
static enum zjit_const_lookup
zjit_const_lexical_search(const rb_cref_t *root_cref, ID id, VALUE *value)
{
    for (const rb_cref_t *cref = root_cref; CREF_NEXT(cref); cref = CREF_NEXT(cref)) {
        VALUE klass = CREF_CLASS(cref);
        if (!zjit_namespace_p(klass)) return ZJIT_CONST_LOOKUP_UNKNOWN;
        rb_const_entry_t *ce = rb_const_lookup(klass, id);
        if (ce) {
            if (UNDEF_P(ce->value)) return ZJIT_CONST_LOOKUP_UNKNOWN; // autoload
            *value = ce->value;
            return ZJIT_CONST_LOOKUP_FOUND;
        }
    }

    // Search the ancestors of the innermost class. (The interpreter would use CLASS_OF(self)
    // if it was nil, which we don't know.)
    if (!zjit_namespace_p(CREF_CLASS(root_cref))) return ZJIT_CONST_LOOKUP_UNKNOWN;
    return zjit_const_search(CREF_CLASS(root_cref), id, false, false, value);
}

// Check whether the constant path `segments`, evaluated in the lexical scope `cref` the way
// vm_get_ev_const_chain() does, still resolves to `expected`, without side effects (no autoload,
// no const_missing, no deprecation warnings, no allocation). This must be called after the
// constant table update that triggered rb_clear_constant_cache_for_id().
// Returns ZJIT_CONST_SAME, ZJIT_CONST_DIFFERENT, or ZJIT_CONST_UNKNOWN.
enum zjit_const_resolution
rb_zjit_constant_path_resolves_to(const rb_cref_t *cref, const ID *segments, VALUE expected)
{
    // Constant tables and their ownership may be updated by other Ractors.
    if (rb_multi_ractor_p()) return ZJIT_CONST_UNKNOWN;
    if (!cref) return ZJIT_CONST_UNKNOWN;
    for (const rb_cref_t *c = cref; c; c = CREF_NEXT(c)) {
        // A dynamic cref makes the IC check the cref on every hit, so we never fold such paths.
        // An ISEQ run by class_eval/instance_eval blocks may have other crefs that share the IC,
        // so we can't resolve the path using only the one we've seen.
        if (CREF_DYNAMIC(c) || CREF_PUSHED_BY_EVAL(c)) return ZJIT_CONST_UNKNOWN;
    }

    VALUE value = Qnil;
    size_t idx = 0;
    bool lexical = true;
    if (segments[0] == idNULL) {
        value = rb_cObject;
        idx++;
        lexical = false;
    }
    while (segments[idx]) {
        ID id = segments[idx++];
        enum zjit_const_lookup result;
        if (lexical) {
            result = zjit_const_lexical_search(cref, id, &value);
            lexical = false;
        }
        else {
            // vm_check_if_namespace() would raise
            if (!RB_TYPE_P(value, T_CLASS) && !RB_TYPE_P(value, T_MODULE)) return ZJIT_CONST_DIFFERENT;
            // rb_public_const_get_from()
            result = zjit_const_search(value, id, true, true, &value);
        }
        switch (result) {
          case ZJIT_CONST_LOOKUP_FOUND: break;
          case ZJIT_CONST_LOOKUP_MISS: return ZJIT_CONST_DIFFERENT; // const_missing, not cached
          case ZJIT_CONST_LOOKUP_UNKNOWN: return ZJIT_CONST_UNKNOWN;
        }
    }
    return value == expected ? ZJIT_CONST_SAME : ZJIT_CONST_DIFFERENT;
}

VALUE rb_nil_to_s(VALUE obj);
VALUE rb_true_to_s(VALUE obj);
VALUE rb_false_to_s(VALUE obj);

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
