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
#include "internal/imemo.h"
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
#include "internal/object.h"
#include "id_table.h"

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
    .stack_map = NULL,
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

VALUE rb_vm_exec_interpreter(rb_execution_context_t *ec);

// Run the frame of a callee that JIT code pushed for a JIT-to-JIT call in the interpreter, and
// return the callee's result to the JIT code, which continues running the caller.
VALUE
rb_zjit_exec_callee_frame(rb_execution_context_t *ec)
{
    // Return from the interpreter when the callee returns, like VM_EXEC() does after vm_sendish()
    VM_ENV_FLAGS_SET(ec->cfp->ep, VM_FRAME_FLAG_FINISH);
    return rb_vm_exec_interpreter(ec);
}

// The number of backward branches taken by the interpreter. See JIT_BACKEDGE().
unsigned int rb_zjit_backedge_count = 0;

// Count a sampled backward branch of an interpreted ISEQ like a call to it, so that an ISEQ
// entered rarely but looping a lot, e.g. a block called once per request that walks a large
// Array, doesn't need call-threshold separate calls to get compiled. Like a call, it enables
// profiling at the profile threshold, which profiles the rest of the loop. It stops one short
// of the call threshold, so that the next call compiles the ISEQ: ZJIT can't enter the compiled
// code in the middle of the running loop.
void
rb_zjit_iseq_backedge(const rb_iseq_t *iseq)
{
    struct rb_iseq_constant_body *body = ISEQ_BODY(iseq);
    if (body->jit_entry != NULL || body->jit_entry_calls + 1 >= rb_zjit_call_threshold) {
        return;
    }
    body->jit_entry_calls++;
    if (body->jit_entry_calls == rb_zjit_profile_threshold) {
        rb_zjit_profile_enable(iseq);
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

// Iterate the (literal, offset) pairs of an opt_case_dispatch CDHASH.
void
rb_zjit_cdhash_foreach(VALUE cdhash, st_foreach_callback_func *func, st_data_t arg)
{
    st_foreach(rb_imemo_cdhash_tbl(cdhash), func, arg);
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

bool
rb_zjit_iclass_for_refinement_p(VALUE iclass)
{
    return RICLASS_FOR_REFINEMENT_P(iclass);
}

// Whether `klass` is the singleton class of an object that isn't a class or module
static bool
object_singleton_class_p(VALUE klass)
{
    if (!RCLASS_SINGLETON_P(klass)) return false;
    VALUE attached = RCLASS_ATTACHED_OBJECT(klass);
    return !RB_TYPE_P(attached, T_CLASS) && !RB_TYPE_P(attached, T_MODULE);
}

// Return the method entry for the Symbol `mid_sym` cached on the class of `recv`, or NULL if the class's method
// cache doesn't have a valid one. Unlike rb_callable_method_entry(), this never fills the cache,
// so it never allocates or takes the VM lock. JIT code calls it to check the callee of a call site
// whose receiver classes vary but which always calls the same method.
//
// `klass_cache` is the call site's cache of up to ZJIT_KLASS_CACHE_SIZE classes found to resolve
// to `expected_cme`. The JIT code checks the entry that the receiver's class hashes to, see
// zjit_klass_cache_index(), and calls this on a miss. A hit skips the method cache lookup, whose
// hash table probe and the loads of the class's method cache entries miss the CPU cache on most
// calls. `cache_owner` is the ISEQ whose JIT code owns the cache and marks the class in it. The JIT code that calls this is invalidated
// when `expected_cme` is, and a class that resolves `mid_sym` to `expected_cme` keeps doing so
// until `expected_cme` is invalidated, which is what makes a cached class a valid hit.
//
// With `match_def`, a method entry of the same method definition, e.g. of a module method included
// in another class, is a hit as well. It isn't cached since `expected_cme` doesn't guard it.
#define ZJIT_KLASS_CACHE_SIZE 32 // KLASS_CACHE_SIZE in zjit/src/hir.rs

// The entry of a klass_cache that JIT code checks for `klass`. Keep it in sync with
// KLASS_CACHE_INDEX_SHIFT1 in zjit/src/hir.rs.
static inline int
zjit_klass_cache_index(VALUE klass)
{
    return (int)(((klass >> 5) ^ (klass >> 10)) & (ZJIT_KLASS_CACHE_SIZE - 1));
}

const rb_callable_method_entry_t *
rb_zjit_cached_cme(VALUE recv, VALUE mid_sym, VALUE *klass_cache, VALUE cache_owner, const rb_callable_method_entry_t *expected_cme, bool match_def)
{
    VALUE klass = CLASS_OF(recv);
    int home = zjit_klass_cache_index(klass);
    // A class is cached in its entry or the entry paired with it, like a 2-way set-associative
    // cache. JIT code doesn't check the cache for immediates, so check their entry here too.
    if (klass_cache[home] == klass) {
        return expected_cme;
    }
    int pair = home ^ 1;
    if (klass_cache[pair] == klass) {
        // Swap it into its entry for the JIT code to find it inline next time
        klass_cache[pair] = klass_cache[home];
        klass_cache[home] = klass;
        return expected_cme;
    }

    ID mid = rb_sym2id(mid_sym);
    VALUE cc_tbl = RUBY_ATOMIC_VALUE_LOAD(RCLASS_WRITABLE_CC_TBL(klass));
    VALUE ccs_data;
    if (cc_tbl && rb_managed_id_table_lookup(cc_tbl, mid, &ccs_data)) {
        const struct rb_class_cc_entries *ccs = (const struct rb_class_cc_entries *)ccs_data;
        if (LIKELY(!METHOD_ENTRY_INVALIDATED(ccs->cme))) {
            if (ccs->cme == expected_cme) {
                // The cache keeps the class alive. Don't let it keep a short-lived object alive
                // through its singleton class.
                if (!object_singleton_class_p(klass) || RB_OBJ_PROMOTED(klass)) {
                    // Put it in its entry and move the class there to the paired entry, evicting
                    // the class in it. Sites whose receiver classes are, e.g., the Relation
                    // subclasses of every Active Record model see many classes in turn.
                    if (klass_cache[home] != Qfalse) {
                        klass_cache[pair] = klass_cache[home];
                    }
                    klass_cache[home] = klass;
                    RB_OBJ_WRITTEN(cache_owner, Qundef, klass);
                }
            }
            else if (match_def && ccs->cme->def == expected_cme->def) {
                return expected_cme;
            }
            return ccs->cme;
        }
    }
    return NULL;
}

// The method entry that the call cache of `cd` holds, or NULL if it's empty or invalidated. It's
// the method that the last call from the interpreter or a dynamic send of JIT code dispatched to.
const rb_callable_method_entry_t *
rb_zjit_call_data_cme(const struct rb_call_data *cd)
{
    const struct rb_callcache *cc = cd->cc;
    if (cc && vm_cc_markable(cc) && !vm_cc_invalidated_p(cc)) {
        // JIT code that refers to a method of a singleton class keeps the object alive
        const rb_callable_method_entry_t *cme = vm_cc_cme(cc);
        if (!object_singleton_class_p(cme->owner) && !object_singleton_class_p(cme->defined_class)) {
            return cme;
        }
    }
    return NULL;
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

// A cache of instance variable indexes for the reads of T_OBJECT ivars that JIT code
// didn't specialize for the object's shape, e.g. at sites whose receivers have more shapes
// than a guard chain covers. Like V8's megamorphic stub cache, it's a direct-mapped table
// keyed by (shape, name), so such a read doesn't walk the shape tree every time.
// Shapes are never freed and an ivar's index on a shape never changes, so an entry is valid
// forever once written. It's only used while a single Ractor runs, where the GVL serializes
// reads and writes of an entry.
#define ZJIT_IVAR_CACHE_BITS 12
struct zjit_ivar_cache_entry {
    ID id;
    shape_id_t shape_id;
    attr_index_t index; // ATTR_INDEX_NOT_SET if the shape doesn't have the ivar
};
static struct zjit_ivar_cache_entry zjit_ivar_cache[1 << ZJIT_IVAR_CACHE_BITS];

// Find the fields of a T_OBJECT `obj` and the index of `id` in them (ATTR_INDEX_NOT_SET if
// unset). Return false if `obj` is not a T_OBJECT with a non-complex shape or multiple
// Ractors are running.
static inline bool
zjit_obj_ivar_index(VALUE obj, ID id, VALUE *fields_obj, attr_index_t *index)
{
    if (SPECIAL_CONST_P(obj) || BUILTIN_TYPE(obj) != T_OBJECT || rb_multi_ractor_p()) {
        return false;
    }
    *fields_obj = ROBJECT_FIELDS_OBJ(obj);
    shape_id_t shape_id = RBASIC_SHAPE_ID(*fields_obj);
    if (rb_shape_complex_p(shape_id)) {
        return false;
    }

    shape_id_t read_shape_id = shape_id & SHAPE_ID_READ_ONLY_MASK;
    // Fibonacci hashing. The low bits of an ID are its scope, so drop them.
    uint32_t hash = (read_shape_id ^ (uint32_t)(id >> 4)) * 2654435761u;
    struct zjit_ivar_cache_entry *entry = &zjit_ivar_cache[hash >> (32 - ZJIT_IVAR_CACHE_BITS)];
    if (entry->id != id || entry->shape_id != read_shape_id) {
        attr_index_t new_index;
        if (!rb_shape_get_iv_index(shape_id, id, &new_index)) {
            new_index = ATTR_INDEX_NOT_SET;
        }
        *entry = (struct zjit_ivar_cache_entry){ .id = id, .shape_id = read_shape_id, .index = new_index };
    }
    *index = entry->index;
    return true;
}

// rb_ivar_get() for a read without an inline cache
VALUE
rb_zjit_getivar(VALUE obj, ID id)
{
    VALUE fields_obj;
    attr_index_t index;
    if (zjit_obj_ivar_index(obj, id, &fields_obj, &index)) {
        return index == ATTR_INDEX_NOT_SET ? Qnil : rb_imemo_fields_ptr(fields_obj)[index];
    }
    return rb_ivar_get(obj, id);
}

VALUE rb_vm_getinstancevariable(const rb_iseq_t *iseq, VALUE obj, ID id, IVC ic);

// rb_vm_getinstancevariable() that doesn't use the inline cache for T_OBJECTs. The inline
// cache holds one shape, so it misses on every read at a site whose receivers alternate.
VALUE
rb_zjit_getinstancevariable(const rb_iseq_t *iseq, VALUE obj, ID id, IVC ic)
{
    VALUE fields_obj;
    attr_index_t index;
    if (zjit_obj_ivar_index(obj, id, &fields_obj, &index)) {
        return index == ATTR_INDEX_NOT_SET ? Qnil : rb_imemo_fields_ptr(fields_obj)[index];
    }
    return rb_vm_getinstancevariable(iseq, obj, id, ic);
}

VALUE
rb_zjit_defined_ivar(VALUE obj, ID id, VALUE pushval)
{
    VALUE fields_obj;
    attr_index_t index;
    if (zjit_obj_ivar_index(obj, id, &fields_obj, &index)) {
        return index == ATTR_INDEX_NOT_SET ? Qnil : pushval;
    }
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

// Return true if unwinding a `break` or non-local `return` through a frame of `iseq`
// stopped at `epc` (the PC offset after the current instruction) would run an ensure
// clause, like vm_exec_handle_exception() does.
bool
rb_zjit_iseq_ensure_covers_p(const rb_iseq_t *iseq, unsigned int epc)
{
    const struct iseq_catch_table *ct = ISEQ_BODY(iseq)->catch_table;
    if (!ct) return false;
    for (unsigned int i = 0; i < ct->size; i++) {
        const struct iseq_catch_table_entry *entry = UNALIGNED_MEMBER_PTR(ct, entries[i]);
        if (entry->type == CATCH_TYPE_ENSURE && entry->start < epc && entry->end >= epc) {
            return true;
        }
    }
    return false;
}

// Return true if a `break` from `block_iseq` lands at `epc` (the PC offset after the
// call that was passed `block_iseq`) in a frame of `iseq`. This mirrors the catch
// table lookup that vm_throw_start() does to check for an orphan `break`.
bool
rb_zjit_iseq_break_lands_p(const rb_iseq_t *iseq, const rb_iseq_t *block_iseq, unsigned int epc)
{
    const struct iseq_catch_table *ct = ISEQ_BODY(iseq)->catch_table;
    if (!ct) return false;
    for (unsigned int i = 0; i < ct->size; i++) {
        const struct iseq_catch_table_entry *entry = UNALIGNED_MEMBER_PTR(ct, entries[i]);
        if (entry->type == CATCH_TYPE_BREAK && entry->iseq == block_iseq &&
            entry->start < epc && entry->end >= epc) {
            return entry->cont == epc;
        }
    }
    return false;
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
rb_cref_t *rb_vm_get_cref(const VALUE *ep);
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

// Return the lexical scope (cref) of an ISEQ frame, or NULL if a scope in it is pushed by eval
const rb_cref_t *
rb_zjit_frame_cref(const rb_control_frame_t *cfp)
{
    const rb_cref_t *cref = rb_vm_get_cref(cfp->ep);
    for (const rb_cref_t *c = cref; c; c = CREF_NEXT(c)) {
        // rb_zjit_cref_may_search_class_p() can't use such a cref, and its class is the receiver's
        // singleton class for instance_eval, so pinning it would keep the receiver alive.
        if (CREF_PUSHED_BY_EVAL(c)) return NULL;
    }
    return cref;
}

// Return true if a constant lookup in the lexical scope `cref` may search `klass`. It checks the
// scopes and the ancestors that vm_get_ev_const() searches for a constant without a scope (`Foo`).
bool
rb_zjit_cref_may_search_class_p(const rb_cref_t *cref, VALUE klass)
{
    for (const rb_cref_t *c = cref; c; c = CREF_NEXT(c)) {
        // vm_get_ev_const() skips scopes pushed by class_eval, so give up on them.
        if (CREF_CLASS(c) == klass || CREF_PUSHED_BY_EVAL(c)) return true;
    }
    VALUE cref_class = CREF_CLASS(cref);
    if (!RB_TYPE_P(cref_class, T_CLASS) && !RB_TYPE_P(cref_class, T_MODULE)) return true;
    if (rb_class_search_ancestor(cref_class, klass)) return true;
    // A module falls back to Object (rb_const_search())
    return RB_TYPE_P(cref_class, T_MODULE) && rb_class_search_ancestor(rb_cObject, klass);
}

VALUE rb_nil_to_s(VALUE obj);
VALUE rb_true_to_s(VALUE obj);
VALUE rb_false_to_s(VALUE obj);
VALUE rb_int_to_s(int argc, VALUE *argv, VALUE x);
VALUE rb_fix_to_s(VALUE x);
VALUE rb_mod_to_s(VALUE klass);

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
