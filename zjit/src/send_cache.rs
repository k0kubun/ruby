//! Class -> callcache tables for send sites that dispatch over too many classes
//! for an inline guard chain.
//!
//! # Why
//!
//! ZJIT specializes a call site by profiling the receiver's classes and emitting
//! a chain of class guards, one arm per profiled class, with a dynamic send as
//! the fallthrough (see [`crate::hir::Function::send_chain_plan`]). That works
//! when a site sees a handful of classes. It does not work for code like
//! Shopify's Storefront Renderer, where a single Liquid dispatch site sees
//! hundreds of unrelated receiver classes: the chain cannot be widened far
//! enough, the ancestor guard only applies when every observed class resolves
//! the name to the *same* method, and everything else falls off the end into
//! [`crate::codegen::gen_send_without_block`], a plain call to
//! `rb_vm_opt_send_without_block`. On SFR that is ~14.1M calls per benchmark
//! run, 41.9% of all dynamic sends.
//!
//! Each of those calls pays for a method *lookup* that has already been done
//! thousands of times. `cd->cc`, the interpreter's inline method cache, holds
//! one class, so a megamorphic site misses it on nearly every call and runs
//! `vm_search_method_slowpath0` -> `vm_search_cc` -> `vm_lookup_cc`: an atomic
//! load of the class's callcache table, an id-table lookup keyed on the method
//! name, a linear scan of that name's entries comparing argc/flags, and then a
//! store of `cd->cc` with a write barrier -- whose only effect at a megamorphic
//! site is to dirty the ISEQ for a cache line that will miss again next call.
//!
//! The step between a one-entry inline cache and that search is a *lookup
//! table*: hash the receiver's class and look the answer up. This module is that
//! table, and it is the send-side counterpart of [`crate::ivar_cache`].
//!
//! # Shape of the fast path
//!
//! Each *call shape* -- a `(method name, argc, call flags)` triple -- gets one
//! [`SendCache`]: a fixed-size, direct-mapped array of callcache pointers keyed
//! by the receiver's class. `rb_zjit_send_cached_without_block` in
//! `vm_insnhelper.c` (and its with-block sibling) hashes `CLASS_OF(recv)`, loads
//! one slot, and on a hit goes straight to `vm_cc_call(cc)` -- the same dispatch
//! `vm_sendish` reaches after its own search. A miss runs exactly the search the
//! old code ran, `vm_search_method_fastpath`, and stores the result in the slot.
//!
//! So the table does not change *how* a megamorphic send is performed, only how
//! its target is found. The call itself, the frame push and the callee are
//! untouched, which is why this is safe to turn on for every megamorphic site
//! and why it composes with the guard chain rather than replacing it.
//!
//! # Dispatching a hit without leaving JIT code
//!
//! Finding the target cheaply still left the *call* expensive: a C call, a
//! generic argument setup in `vm_call_iseq_setup()`, an interpreter frame push,
//! the `setjmp` in `vm_exec()`, and a trip through `jit_exec()` and the entry
//! trampoline -- all to arrive at code ZJIT had already compiled. On lobsters
//! 45% of megamorphic sends resolve to an ISEQ method with fixed arity, no
//! locals beyond its parameters and compiled code waiting for it.
//!
//! For those, [`crate::codegen::gen_send_megamorphic_direct`] does the probe
//! inline and calls the callee itself. Each slot therefore carries a second
//! word, the method entry the target resolves to *when the target is one of
//! those* (`zjit_send_cache_direct_cme()` in `vm_insnhelper.c` decides), and JIT
//! code checks that word against the method entry it read out of the callcache
//! before trusting it. That check is what keeps the two-word slot as safe as the
//! one-word slot was: a reader that catches a fill half-done sees a mismatched
//! pair and searches, so neither word needs a barrier or an atomic.
//!
//! The compiled entry point itself is never cached. JIT code loads
//! `ISEQ_BODY(iseq)->jit_entry` on every call, which `rb_iseq_reset_jit_func()`
//! clears whenever that code stops being valid, so an invalidated callee reads
//! as a null and takes the slow path.
//!
//! # What the table is keyed on, and why that is enough
//!
//! The table memoizes `vm_lookup_cc(klass, ci)`. That function's answer depends
//! on the receiver class and on four properties of the call site: the method
//! name, `argc`, the call flags, and the number of keyword arguments -- it
//! compares exactly those when scanning a class's entries, and returns one
//! shared callcache to every site that agrees on them. So a table keyed on
//! `(mid, argc, flags)` and indexed by class is memoizing a pure function of its
//! own key, and sites that share a call shape share a table and warm it for each
//! other, exactly as ivar sites reading the same name share one shape table.
//!
//! Sites with an explicit keyword-argument list are excluded ([`cache_for`]), so
//! the keyword count is always zero and drops out of the key. That is not
//! required for correctness -- the interpreter shares a callcache across sites
//! that differ only in *which* keywords they pass -- but it keeps the key small
//! and costs nothing: a megamorphic Liquid-style dispatch site passes positional
//! arguments.
//!
//! Keying per shape rather than per site also bounds the memory. Per site, the
//! table size would be multiplied by the number of compiled megamorphic sites
//! and again by every recompile; per call shape it is multiplied by the number
//! of distinct `(name, argc, flags)` triples that a program actually dispatches
//! megamorphically, which is small and does not grow over time.
//!
//! # Why a stale entry can never be wrong
//!
//! An entry is trusted under exactly the condition `vm_cc_hit_p` trusts
//! `cd->cc` under, and for exactly the same reasons:
//!
//! * `cc->klass == klass`. A callcache stores its own key, so the entry
//!   validates itself and no separate key word is needed.
//! * `!METHOD_ENTRY_INVALIDATED(vm_cc_cme(cc))`. Every definition, `undef`,
//!   alias, visibility change, `include`, `prepend` and refinement of a method
//!   runs `rb_clear_method_cache` -> `vm_cme_invalidate`, which sets that flag on
//!   the method entry the cache resolved to. A cached entry for a redefined
//!   method therefore fails the probe and is replaced by the fresh lookup, in the
//!   same GC-free, lock-free way the interpreter's own inline cache is.
//!
//! Those two checks also cover object death, in the right order. If the class or
//! the method entry is collected, GC's weak-reference pass calls
//! `vm_cc_invalidate`, which sets `cc->klass = Qundef`; no live class is
//! `Qundef`, so the class compare fails *before* the method entry is read, which
//! is what the "cc->cme must not be accessed after invalidation" rule in
//! `imemo.c` requires.
//!
//! There is therefore no patch point, no [`crate::invariants`] entry and no
//! on-redefinition hook for this table. What it does need, unlike
//! [`crate::ivar_cache`], is the GC's attention: it holds `VALUE`s.
//!
//! # GC
//!
//! A callcache is an `imemo_callcache` object, reachable only from the class's
//! callcache table (which drops it when the method is invalidated) and from
//! `cd->cc`. Neither keeps it alive for us, so [`mark`] marks every cached entry
//! from [`crate::gc::rb_zjit_root_mark`], which the GC runs as a root on every
//! collection, minor or major.
//!
//! Marking a callcache retains nothing else: `imemo.c` marks a normal
//! callcache's `klass` and `cme_` weakly on purpose, and only `super` and
//! `refinement` caches mark their method entry. So a table cannot pin a dead
//! class or a dead method -- it can only hold onto a fixed number of two-word
//! imemos, which is exactly what makes the self-invalidation above work. (Those
//! `super` and `refinement` caches are the ones we refuse to store, in the C
//! helper's `zjit_send_cache_cacheable_p`, precisely because they *do* retain
//! their method entry.)
//!
//! Compaction moves classes, and a class address is the hash key here, so
//! [`update_references`] drops every entry instead of rehashing: after a
//! compaction the keys are wrong anyway, the tables refill from the interpreter's
//! own search, and compaction is rare enough that keeping the warm entries is not
//! worth the code.
//!
//! # Ractors
//!
//! A slot is one naturally-aligned pointer, published with a single plain store
//! -- the same way `vm_search_method_slowpath0` publishes `cd->cc` -- so a
//! reader sees either the old callcache or the new one, never a mix. Two ractors
//! racing on one slot both store a valid callcache for the class they looked up,
//! and a reader validates whichever it sees against its own receiver's class
//! before using it.

use std::collections::HashMap;

use crate::cruby::*;
use crate::options::get_option;
use crate::stats::{Counter, counter_ptr, incr_counter, incr_counter_by};
use crate::state::ZJITState;

/// Multiplier of the Fibonacci hash in [`slot_of`], which must match
/// `zjit_send_cache_slot` in `vm_insnhelper.c`. The key is a class `VALUE`, i.e.
/// a heap address whose low bits are zero from object alignment and whose high
/// bits barely vary within one heap page, so neither end of the word can be used
/// as an index directly. A multiply mixes every input bit into the high bits,
/// which is where the index is taken from.
pub const SEND_CACHE_HASH_MULT: u64 = 0x9e37_79b9_7f4a_7c15;

/// Index a class maps to in a table of `len` slots: the top `log2(len)` bits of
/// the hash product, taken with a second multiply rather than with a shift by
/// `64 - log2(len)`.
///
/// The two are the same number -- for a power-of-two `len`, `(t * len) >> 32` is
/// `t >> (32 - log2(len))` -- but only one of them can be emitted without
/// knowing `len` when the code is written. Tables are sized adaptively (see
/// [`SendCache::grow`]), so the length has to come from the table's header at
/// run time, and a *shift* by a loaded amount needs the count in a fixed
/// register on x86-64 while a *multiply* by a loaded amount does not.
///
/// Masking the product's low bits instead, the way [`crate::ivar_cache`] indexes
/// its fixed-size tables, is not an option here: it would index a small table on
/// a narrow window in the middle of the product, and classes are allocated at a
/// fixed stride from a heap page, so for any such window there is a stride that
/// collapses it (measured: 40-byte-strided classes reach 4 of 32 slots through
/// bits 52..57, against 26 through the top 5). Only the top bits are safe for
/// every stride, which is what this keeps.
#[inline]
pub fn slot_of(klass: u64, len: u32) -> usize {
    let hash = klass.wrapping_mul(SEND_CACHE_HASH_MULT);
    (((hash >> 32) * len as u64) >> 32) as usize
}

/// log2 of `size_of::<SendCacheEntry>()`. Asserted against the C `sizeof` in
/// [`SendCacheLayout::get`].
pub const SEND_CACHE_ENTRY_SHIFT: u32 = 4;

/// Slots a table starts with, i.e. 512 bytes per call shape.
///
/// Almost every call shape a program dispatches megamorphically sees a handful
/// of classes: on `benchmarks/lobsters` 1607 tables hold 4.3 live entries each
/// on average, so a fixed 512-slot table left 98% of 13.3MB empty -- memory the
/// GC then walked as a root on every collection. Starting small and growing the
/// few shapes that actually thrash (see [`SendCache::grow`]) keeps the hit rate
/// of a big table at a fraction of the footprint.
pub const SEND_CACHE_INITIAL_ENTRIES: usize = 32;

/// How much a table grows when it thrashes, and -- multiplied by the current
/// length -- how many evictions it takes to decide that it does. Four evictions
/// per slot is well past what a table holding its working set produces and is
/// reached in the first few thousand calls by one that does not.
const SEND_CACHE_GROWTH_FACTOR: usize = 4;

/// Default of `--zjit-send-cache-entries`: the size a table is allowed to grow
/// *to*, i.e. at most 8KiB per call shape.
///
/// A direct-mapped table smaller than a site's class working set thrashes rather
/// than degrading smoothly: under a cyclic access pattern every slot holding two
/// live classes misses on *every* call. Table hit rate measured on
/// `benchmark/zjit_megamorphic_send.rb`, cyclic / skewed receiver order:
///
/// | slots | 50 classes  | 200 classes | 500 classes |
/// |-------|-------------|-------------|-------------|
/// |   128 | 96% / 100%  |  15% /  75% |   1% /  64% |
/// |   256 | 100% / 100% |  31% /  85% |  10% /  74% |
/// |   512 | 100% / 100% | 100% /  96% |  57% /  94% |
/// |  1024 | 100% / 100% | 100% / 100% |  87% /  97% |
///
/// 512 is where a few hundred classes stop colliding without spending 16KiB a
/// shape. Note this is a softer decision than the ivar table's: a miss here runs
/// exactly the search the site ran before the table existed, so the probe pays
/// for itself from a hit rate of a few percent up. Undersizing costs the win,
/// not performance.
pub const DEFAULT_CACHE_ENTRIES: usize = 512;

/// Largest a table may grow, from `--zjit-send-cache-entries`.
pub fn cache_entries() -> usize {
    get_option!(send_cache_entries, DEFAULT_CACHE_ENTRIES)
}

/// Slots a fresh table gets, never more than the ceiling above.
fn initial_cache_entries() -> usize {
    SEND_CACHE_INITIAL_ENTRIES.min(cache_entries())
}

/// The call shape a table is for: the inputs, other than the receiver class,
/// that `vm_lookup_cc` compares when it picks a callcache. See the module docs
/// for why keyword-argument sites are excluded rather than keyed on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SendCacheKey {
    pub mid: ID,
    pub argc: u32,
    pub flags: u32,
}

/// One slot: the two words `struct rb_zjit_send_cache_entry` in `zjit.h`
/// declares. Rust only ever reads word 0 (the callcache, which is the only one
/// the GC has to hear about); word 1 is written and validated in C and in JIT
/// code. Keep the two declarations in step.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SendCacheEntry {
    /// The cached callcache as a raw word, 0 when the slot is empty.
    cc: usize,
    /// `vm_cc_cme(cc)` when the target is directly callable from JIT code, 0
    /// otherwise. Never dereferenced from Rust.
    direct_cme: usize,
}

/// One class table, shared by every compiled site with the same call shape.
///
/// The first seven fields are read by `vm_insnhelper.c` through
/// `struct rb_zjit_send_cache` in `zjit.h`; keep the two declarations in step.
#[repr(C)]
pub struct SendCache {
    /// Number of slots, and the multiplier [`slot_of`] scales the hash by. A
    /// power of two.
    ///
    /// Written *after* `slots` when the table grows, so that a reader is never
    /// handed a length longer than the table it reads from.
    len: u32,
    /// Evictions since the table was allocated or last grown, counted by the C
    /// fill path. Only meaningful while `grow_at` is non-zero.
    evictions: u32,
    /// Address of slot 0. Points into `storage`, which only moves in
    /// [`SendCache::grow`], and only when no other thread can be mid-probe.
    slots: *mut SendCacheEntry,
    /// `counter_ptr(send_cache_hit)` under `--zjit-stats`, null otherwise. The
    /// C helper uses it both as the counter and as the flag that says whether to
    /// call [`rb_zjit_send_cache_record_miss`] at all, so that a build without
    /// stats pays one never-taken branch per send rather than a call per miss.
    hit_counter: *mut u64,
    /// `argc` of the call shape, for the arity test the C fill path runs before
    /// it marks a target directly callable.
    direct_argc: u32,
    /// 1 when this call shape may use the inline direct-dispatch path at all,
    /// 0 when the shape rules it out. See [`shape_allows_direct`].
    direct_ok: u32,
    /// The call shape's `vm_ci_flag()`, which the C fill path consults for the
    /// visibility test.
    direct_flags: u32,
    /// Evictions at which the C fill path calls [`rb_zjit_send_cache_grow`], or
    /// 0 for a table that will not grow again -- it is already at the ceiling,
    /// or the program has more than one ractor. See [`SendCache::grow`].
    grow_at: u32,

    /// Backing store for `slots`, all-zero for empty.
    storage: Box<[SendCacheEntry]>,
    /// The call shape this table is for. Only read by debug output.
    key: SendCacheKey,
}

impl SendCache {
    fn new(key: SendCacheKey) -> Box<Self> {
        let len = initial_cache_entries();
        debug_assert!(len.is_power_of_two());
        let mut cache = Box::new(SendCache {
            len: 0,
            evictions: 0,
            slots: std::ptr::null_mut(),
            hit_counter: if get_option!(stats) {
                counter_ptr(Counter::send_cache_hit)
            } else {
                std::ptr::null_mut()
            },
            direct_argc: key.argc,
            direct_ok: u32::from(shape_allows_direct(key.flags)),
            direct_flags: key.flags,
            grow_at: 0,
            storage: empty_storage(len),
            key,
        });
        // Only now that `storage` has its final address: the C helper indexes
        // `slots` without going through the `Box`.
        cache.publish(len);
        cache
    }

    /// Point `slots`, `len` and `grow_at` at the current `storage`, whose length
    /// must be `len`.
    ///
    /// `slots` is stored before `len` so that a reader racing with
    /// [`SendCache::grow`] can only ever pair a length with a table at least
    /// that large: tables only grow, so an old length against the new table is a
    /// probe of a prefix, and the new length against the old table -- the
    /// pairing that would read out of bounds -- is the one this order rules out.
    /// See `grow` for why the race cannot happen in the first place.
    fn publish(&mut self, len: usize) {
        debug_assert_eq!(len, self.storage.len());
        self.slots = self.storage.as_mut_ptr();
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        self.len = len as u32;
        self.evictions = 0;
        self.grow_at = if len >= cache_entries() {
            0
        } else {
            (len * SEND_CACHE_GROWTH_FACTOR) as u32
        };
    }

    /// Replace the table with a larger one because it is thrashing: the classes
    /// this call shape dispatches over do not fit, so slots are being evicted
    /// rather than hit.
    ///
    /// The cached entries are dropped rather than rehashed. They are a memo of a
    /// search the interpreter can redo, the table refills within a few thousand
    /// calls, and a table grows at most twice in its life.
    ///
    /// Growth frees the old storage, so it must not run while another thread can
    /// be between loading `slots` and reading the slot. Ruby threads within a
    /// ractor are serialized and take no interrupt check inside a probe, so the
    /// only way to have a concurrent reader is a second ractor -- and a table
    /// simply stops growing once the program has one.
    fn grow(&mut self) {
        let max = cache_entries();
        let len = self.len as usize;
        if len >= max || unsafe { rb_jit_multi_ractor_p() } {
            self.grow_at = 0;
            return;
        }
        let new_len = (len * SEND_CACHE_GROWTH_FACTOR).min(max);
        self.storage = empty_storage(new_len);
        self.publish(new_len);
        incr_counter!(send_cache_grow_count);
    }

    /// Bytes this table owns on the Rust heap.
    pub fn heap_size(&self) -> usize {
        size_of::<SendCache>() + self.storage.len() * size_of::<SendCacheEntry>()
    }

    /// Address of the table's header, which the inline probe bakes in and reads
    /// `slots` and `len` out of.
    pub fn header_ptr(&self) -> *const u8 {
        self as *const SendCache as *const u8
    }

    /// Whether this table's call shape can use the inline direct-dispatch path.
    pub fn direct_ok(&self) -> bool {
        self.direct_ok != 0
    }

    /// The call shape this table serves.
    pub fn key(&self) -> SendCacheKey {
        self.key
    }

    /// Slot a class maps to. Must match `zjit_send_cache_slot` in
    /// `vm_insnhelper.c`, or the helper would fill and probe different slots.
    #[allow(dead_code)]
    fn slot_of(&self, klass: VALUE) -> usize {
        slot_of(klass.as_u64(), self.len)
    }

    /// Mark every cached callcache. See the module docs on why this retains
    /// nothing but the callcaches themselves.
    fn mark(&self) -> usize {
        let mut marked = 0;
        for slot in self.storage.iter() {
            if slot.cc != 0 {
                marked += 1;
                unsafe { rb_gc_mark_movable(VALUE(slot.cc)) };
            }
        }
        marked
    }

    /// Drop every entry, because compaction has moved the classes the slots were
    /// hashed from. Returns how many entries were live.
    fn clear(&mut self) -> usize {
        let mut dropped = 0;
        for slot in self.storage.iter_mut() {
            if slot.cc != 0 {
                dropped += 1;
            }
            // Drop the pair together: `direct_cme` is only meaningful next to
            // the `cc` it was derived from.
            *slot = SendCacheEntry { cc: 0, direct_cme: 0 };
        }
        dropped
    }

    /// Number of occupied slots, for `--zjit-stats`.
    pub fn occupancy(&self) -> usize {
        self.storage.iter().filter(|slot| slot.cc != 0).count()
    }
}

/// `len` empty slots. Zeroed rather than filled so that the allocator can hand
/// back fresh pages it already knows are zero.
fn empty_storage(len: usize) -> Box<[SendCacheEntry]> {
    debug_assert!(len.is_power_of_two());
    vec![SendCacheEntry { cc: 0, direct_cme: 0 }; len].into_boxed_slice()
}

/// Grow a thrashing table. Called from `zjit_send_cache_search` once its
/// eviction count reaches `grow_at`; see [`SendCache::grow`].
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_send_cache_grow(cache: *mut SendCache) {
    unsafe { (*cache).grow() };
}

/// Table for `key`, creating it on first use. The `Box` is owned by
/// [`ZJITState`] so that [`crate::mem_stats`] can account for it; the returned
/// pointer is stable for the life of the process, which is what lets JIT code
/// bake it in.
fn send_cache_for(key: SendCacheKey) -> *const SendCache {
    let caches = ZJITState::get_send_caches();
    let cache = caches.entry(key).or_insert_with(|| {
        incr_counter!(send_cache_alloc_count);
        SendCache::new(key)
    });
    cache.as_ref() as *const SendCache
}

/// Whether a call shape lets a table hit be dispatched with a direct call into
/// the callee's JIT code, rather than through `vm_cc_call()` and the
/// interpreter.
///
/// [`crate::codegen::gen_send_megamorphic_direct`] pushes the callee frame
/// itself, with the arguments left exactly where the caller's operand stack
/// already put them and the frame size fixed at compile time. Every flag here
/// would move an argument, change `argc`, or replace the frame:
///
/// * `ARGS_SPLAT` / `KW_SPLAT` / `KWARG`: `CALLER_SETUP_ARG()` rewrites the
///   argument list before the arity is known, so the site's `argc` is not the
///   callee's.
/// * `ARGS_BLOCKARG`: the block argument has to be popped and converted, which
///   can run `to_proc`, i.e. arbitrary Ruby.
/// * `TAILCALL`: dispatches to `vm_call_iseq_setup_tailcall()`, which reuses the
///   caller's frame instead of pushing one.
/// * `SUPER` / `ZSUPER` / `FORWARDING`: never reach this helper at all
///   ([`cache_for`] already refuses them), listed so the set is complete.
///
/// A site with a literal block passes no flag of its own, so it can share a
/// table with a block-less site of the same shape. That is harmless: only
/// [`crate::codegen::gen_send_without_block`] reads `direct_cme`, and it is
/// never used for a send that has a block.
fn shape_allows_direct(flags: u32) -> bool {
    const REJECTED: u32 = VM_CALL_ARGS_SPLAT
        | VM_CALL_KW_SPLAT
        | VM_CALL_KWARG
        | VM_CALL_ARGS_BLOCKARG
        | VM_CALL_TAILCALL
        | VM_CALL_SUPER
        | VM_CALL_ZSUPER
        | VM_CALL_FORWARDING;
    flags & REJECTED == 0
}

/// Offsets and masks the inline probe bakes into JIT code, read once from C
/// because the structs they reach into are opaque to Rust. See the declarations
/// in `zjit.h`.
pub struct SendCacheLayout {
    /// `offsetof(struct rb_zjit_send_cache_entry, direct_cme)`
    pub entry_direct_cme: i32,
    /// `sizeof(struct rb_zjit_send_cache_entry)`
    pub entry_size: usize,
    /// `offsetof(struct rb_zjit_send_cache, slots)`
    pub cache_slots: i32,
    /// `offsetof(struct rb_zjit_send_cache, len)`
    pub cache_len: i32,
    /// `offsetof(struct rb_callcache, klass)`
    pub cc_klass: i32,
    /// `offsetof(struct rb_callcache, cme_)`
    pub cc_cme: i32,
    /// `offsetof(rb_callable_method_entry_t, def)`
    pub cme_def: i32,
    /// `offsetof(rb_method_definition_t, body.iseq.iseqptr)`
    pub def_iseqptr: i32,
    /// `offsetof(struct rb_iseq_struct, body)`
    pub iseq_body: i32,
    /// `offsetof(struct rb_iseq_constant_body, jit_entry)`
    pub body_jit_entry: i32,
    /// `IMEMO_FL_USER5`, the bit `METHOD_ENTRY_INVALIDATED()` tests
    pub cme_invalidated_flag: u64,
    /// `ZJIT_MEGA_DIRECT_MAX_STACK`: the `stack_max` bound the C fill path holds
    /// directly-callable callees to, which is what the call site's stack
    /// overflow check is compiled against.
    pub direct_max_stack: usize,
}

impl SendCacheLayout {
    pub fn get() -> Self {
        unsafe extern "C" {
            fn rb_zjit_cc_klass_offset() -> usize;
            fn rb_zjit_cc_cme_offset() -> usize;
            fn rb_zjit_cme_def_offset() -> usize;
            fn rb_zjit_def_iseqptr_offset() -> usize;
            fn rb_zjit_iseq_body_offset() -> usize;
            fn rb_zjit_iseq_body_jit_entry_offset() -> usize;
            fn rb_zjit_send_cache_entry_size() -> usize;
            fn rb_zjit_send_cache_entry_direct_cme_offset() -> usize;
            fn rb_zjit_method_entry_invalidated_flag() -> VALUE;
            fn rb_zjit_mega_direct_max_stack() -> usize;
        }
        // The C declaration of a slot has to agree with the Rust one, or the
        // helper would fill slots the probe never reads.
        unsafe {
            assert_eq!(rb_zjit_send_cache_entry_size(), size_of::<SendCacheEntry>());
            assert_eq!(SEND_CACHE_ENTRY_SHIFT, size_of::<SendCacheEntry>().trailing_zeros());
        }
        unsafe {
            SendCacheLayout {
                entry_direct_cme: rb_zjit_send_cache_entry_direct_cme_offset() as i32,
                entry_size: rb_zjit_send_cache_entry_size(),
                cache_slots: std::mem::offset_of!(SendCache, slots) as i32,
                cache_len: std::mem::offset_of!(SendCache, len) as i32,
                cc_klass: rb_zjit_cc_klass_offset() as i32,
                cc_cme: rb_zjit_cc_cme_offset() as i32,
                cme_def: rb_zjit_cme_def_offset() as i32,
                def_iseqptr: rb_zjit_def_iseqptr_offset() as i32,
                iseq_body: rb_zjit_iseq_body_offset() as i32,
                body_jit_entry: rb_zjit_iseq_body_jit_entry_offset() as i32,
                cme_invalidated_flag: rb_zjit_method_entry_invalidated_flag().as_u64(),
                direct_max_stack: rb_zjit_mega_direct_max_stack(),
            }
        }
    }
}

/// The table a compiled send site should probe, or `None` to keep calling the
/// plain interpreter entry point.
///
/// Called at compile time, once per site, so it can afford the hash lookup that
/// the shared-per-shape design needs.
pub fn cache_for(cd: *const rb_call_data, reason: crate::hir::SendFallbackReason) -> Option<*const SendCache> {
    use crate::hir::SendFallbackReason::*;

    if get_option!(disable_send_cache) {
        return None;
    }

    // Only sites that dispatch over many classes. A site that falls back for a
    // reason unrelated to its receiver -- an unsupported method type, a
    // visibility check, no profile at all -- may well be monomorphic, and there
    // `cd->cc` already hits on every call and the table would be pure overhead.
    match reason {
        SendMegamorphic | SendPolymorphic => {}
        _ => return None,
    }

    let ci = unsafe { (*cd).ci };
    let flags = unsafe { vm_ci_flag(ci) };

    // `super` does not resolve through `vm_search_method_fastpath` at all, and
    // `...` forwarding rewrites the call data before dispatching; neither reaches
    // the helper this table feeds.
    if flags & (VM_CALL_SUPER | VM_CALL_ZSUPER | VM_CALL_FORWARDING) != 0 {
        return None;
    }
    // See the module docs: excluded to keep the keyword count out of the key.
    if flags & VM_CALL_KWARG != 0 || !unsafe { rb_vm_ci_kwarg(ci) }.is_null() {
        return None;
    }

    Some(send_cache_for(SendCacheKey {
        mid: unsafe { vm_ci_mid(ci) },
        argc: unsafe { vm_ci_argc(ci) },
        flags,
    }))
}

/// Mark the callcaches every table holds. Called from
/// [`crate::gc::rb_zjit_root_mark`].
pub fn mark_all() {
    for cache in ZJITState::get_send_caches().values() {
        cache.mark();
    }
}

/// Drop every table's entries after a compacting GC moved the classes they are
/// keyed on. Called from [`crate::gc::rb_zjit_root_update_references`].
pub fn update_references() {
    let mut dropped = 0;
    for cache in ZJITState::get_send_caches().values_mut() {
        dropped += cache.clear();
    }
    if dropped != 0 {
        incr_counter_by(Counter::send_cache_compaction_drop, dropped as u64);
    }
}

/// Why the C helper's probe did not produce a callcache. Values must match the
/// `ZJIT_SEND_CACHE_MISS_*` constants in `zjit.h`.
const MISS_FILL: i32 = 0;
const MISS_EVICT: i32 = 1;
const MISS_STALE: i32 = 2;
const MISS_UNCACHEABLE: i32 = 3;

/// Record which kind of miss the C helper just took. Only called when
/// `--zjit-stats` is on, which is why the classification happens in C (where the
/// old slot is still in hand) rather than here.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_send_cache_record_miss(kind: i32) {
    incr_counter_by(
        match kind {
            MISS_FILL => Counter::send_cache_fill,
            MISS_EVICT => Counter::send_cache_evict,
            MISS_STALE => Counter::send_cache_stale,
            MISS_UNCACHEABLE => Counter::send_cache_uncacheable,
            _ => return,
        },
        1,
    );
}

/// Total slots and occupied slots across every table, for `--zjit-stats`.
pub fn occupancy_totals() -> (usize, usize) {
    let caches = ZJITState::get_send_caches();
    let total = caches.values().map(|cache| cache.len as usize).sum();
    let used = caches.values().map(|cache| cache.occupancy()).sum();
    (total, used)
}

/// Every table, for [`crate::mem_stats`] and `--zjit-stats`.
pub type SendCaches = HashMap<SendCacheKey, Box<SendCache>>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every table size a program can reach, from the smallest a table starts
    /// at to the largest `--zjit-send-cache-entries` is ever set to.
    const SIZES: [u32; 10] = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    /// The entry-size constant JIT code shifts by has to be the real one, or the
    /// probe reads between slots.
    #[test]
    fn entry_shift_matches_the_entry_size() {
        assert_eq!(SEND_CACHE_ENTRY_SHIFT, size_of::<SendCacheEntry>().trailing_zeros());
    }

    /// Scaling by the length has to be the shift it replaces: `slot_of` is
    /// emitted as two multiplies in JIT code because the length is only known at
    /// run time, and the two have to be the same number at every size.
    #[test]
    fn scaling_by_the_length_is_a_shift_by_the_log() {
        for len in SIZES {
            let shift = 64 - len.trailing_zeros();
            for klass in [0u64, 8, 0x10, 0xffff_ffff_ffff_fff8, 0x5555_5555_5555_5550, 0x7f00_0000_1000] {
                let expected = (klass.wrapping_mul(SEND_CACHE_HASH_MULT) >> shift) as usize;
                assert_eq!(slot_of(klass, len), expected, "len {len} klass {klass:#x}");
            }
        }
    }

    /// Every slot index a class address can produce must be in range, at every
    /// size a table passes through, including for the extreme words a `VALUE`
    /// can hold.
    #[test]
    fn slots_are_in_range() {
        for len in SIZES {
            for klass in [0u64, 8, 0x10, 0xffff_ffff_ffff_fff8, 0x5555_5555_5555_5550] {
                let slot = slot_of(klass, len);
                assert!(slot < len as usize, "len {len} klass {klass:#x} slot {slot}");
            }
        }
    }

    /// Classes are allocated at a fixed stride from a heap page, so a table has
    /// to spread a strided run of addresses at *every* size it can be -- a small
    /// table just as much as a big one, since every table now starts small.
    #[test]
    fn strided_addresses_do_not_share_a_slot() {
        // Sizes of the objects classes are allocated from, and then some.
        for stride in [40u64, 80, 160, 320] {
            for len in SIZES {
                let slots: std::collections::HashSet<usize> = (0..len as u64)
                    .map(|i| slot_of(0x7f00_0000_1000 + i * stride, len))
                    .collect();
                // Filling `len` slots with `len` random keys leaves ~63% of them
                // distinct, which is the ceiling for a direct-mapped table; the
                // bar is deliberately well under it so the test does not pin the
                // exact hash. What it does catch is a index that stops mixing:
                // masking a middle window of the product instead of taking its
                // top bits reaches 4 of 32 slots at this stride.
                assert!(
                    slots.len() * 5 >= len as usize * 2,
                    "stride {stride}: only {} distinct slots for {len} classes in {len} slots",
                    slots.len()
                );
            }
        }
    }

    /// Every slot of a table has to be reachable at every size it grows through,
    /// or growing it would buy fewer slots than it paid for.
    #[test]
    fn growth_reaches_every_slot_at_every_size() {
        let mut len = SEND_CACHE_INITIAL_ENTRIES;
        while len <= 4096 {
            let reached: std::collections::HashSet<usize> =
                (0..64u64 * len as u64).map(|k| slot_of(k * 8, len as u32)).collect();
            assert_eq!(reached.len(), len, "len {len}");
            len *= SEND_CACHE_GROWTH_FACTOR;
        }
    }
}
