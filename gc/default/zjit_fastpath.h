#ifndef RUBY_GC_DEFAULT_ZJIT_FASTPATH_H
#define RUBY_GC_DEFAULT_ZJIT_FASTPATH_H

#include <stddef.h>

#include "gc/gc_impl.h"
#include "ruby/internal/static_assert.h"
#include "ruby/ruby.h"

struct rb_gc_zjit_default_new_obj_fastpath {
    size_t cursor_offset;
    size_t cursor_end_offset;
    size_t slot_size;
    size_t total_allocated_objects_offset;
    VALUE flags;
    VALUE klass;
};

RBIMPL_STATIC_ASSERT(zjit_default_fastpath_fits,
                     sizeof(struct rb_gc_zjit_default_new_obj_fastpath) <= sizeof(union rb_gc_zjit_fastpath_data));

/* Everything ZJIT needs to inline the fast path of rb_gc_impl_writebarrier(a, b).
 * The barrier can be skipped entirely when
 *
 *     *incremental_marking_count == 0 &&
 *     (RBASIC(a)->flags & recv_slowpath_flags) == 0 &&
 *     (!(RBASIC(a)->flags & promoted_flag) || (RBASIC(b)->flags & promoted_flag))
 *
 * i.e. when no incremental mark is running, `a` does not need the shareable-object
 * bookkeeping, and the generational barrier has nothing to remember. */
struct rb_gc_zjit_default_writebarrier_fastpath {
    /* Address of a counter of the objspaces that are in the middle of an incremental
     * mark.  While it is non-zero, the write barrier must always be called. */
    const void *incremental_marking_count;
    /* Width of the counter above, in bits. */
    size_t incremental_marking_count_num_bits;
    /* Receiver flags that force the call even when the generational rules say otherwise. */
    VALUE recv_slowpath_flags;
    /* The RUBY_FL_PROMOTED bit, i.e. RVALUE_OLD_P() for a non-special-const object. */
    VALUE promoted_flag;
};

RBIMPL_STATIC_ASSERT(zjit_default_writebarrier_fastpath_fits,
                     sizeof(struct rb_gc_zjit_default_writebarrier_fastpath) <= sizeof(union rb_gc_zjit_fastpath_data));

#endif /* RUBY_GC_DEFAULT_ZJIT_FASTPATH_H */
