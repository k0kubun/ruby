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

/// Default of `--zjit-send-cache-entries`, i.e. 4KiB per call shape.
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
/// 512 is where a few hundred classes stop colliding without spending 8KiB a
/// shape. Note this is a softer decision than the ivar table's: a miss here runs
/// exactly the search the site ran before the table existed, so the probe pays
/// for itself from a hit rate of a few percent up. Undersizing costs the win,
/// not performance.
pub const DEFAULT_CACHE_ENTRIES: usize = 512;

/// Number of slots in each table, from `--zjit-send-cache-entries`.
pub fn cache_entries() -> usize {
    get_option!(send_cache_entries, DEFAULT_CACHE_ENTRIES)
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

/// One class table, shared by every compiled site with the same call shape.
///
/// The first four fields are read by `vm_insnhelper.c` through
/// `struct rb_zjit_send_cache` in `zjit.h`; keep the two declarations in step.
#[repr(C)]
pub struct SendCache {
    /// Number of slots. A power of two, so `len - 1` is a mask.
    len: u32,
    /// `64 - log2(len)`: the right shift that turns the hash product into a slot
    /// index, with no mask needed because the shift keeps only that many bits.
    shift: u32,
    /// Address of slot 0. Points into `storage`, which is never reallocated
    /// because the C helper holds this pointer directly.
    slots: *mut usize,
    /// `counter_ptr(send_cache_hit)` under `--zjit-stats`, null otherwise. The
    /// C helper uses it both as the counter and as the flag that says whether to
    /// call [`rb_zjit_send_cache_record_miss`] at all, so that a build without
    /// stats pays one never-taken branch per send rather than a call per miss.
    hit_counter: *mut u64,

    /// Backing store for `slots`: one callcache pointer per slot, 0 for empty.
    storage: Box<[usize]>,
    /// The call shape this table is for. Only read by debug output.
    key: SendCacheKey,
}

impl SendCache {
    fn new(key: SendCacheKey) -> Box<Self> {
        let len = cache_entries();
        debug_assert!(len.is_power_of_two());
        let mut cache = Box::new(SendCache {
            len: len as u32,
            shift: 64 - len.trailing_zeros(),
            slots: std::ptr::null_mut(),
            hit_counter: if get_option!(stats) {
                counter_ptr(Counter::send_cache_hit)
            } else {
                std::ptr::null_mut()
            },
            storage: vec![0usize; len].into_boxed_slice(),
            key,
        });
        // Only now that `storage` has its final address: the C helper indexes
        // `slots` without going through the `Box`.
        cache.slots = cache.storage.as_mut_ptr();
        cache
    }

    /// Bytes this table owns on the Rust heap.
    pub fn heap_size(&self) -> usize {
        size_of::<SendCache>() + self.storage.len() * size_of::<usize>()
    }

    /// The call shape this table serves.
    pub fn key(&self) -> SendCacheKey {
        self.key
    }

    /// Slot a class maps to. Must match `zjit_send_cache_slot` in
    /// `vm_insnhelper.c`, or the helper would fill and probe different slots.
    fn slot_of(&self, klass: VALUE) -> usize {
        ((klass.as_u64().wrapping_mul(SEND_CACHE_HASH_MULT)) >> self.shift) as usize
    }

    /// Mark every cached callcache. See the module docs on why this retains
    /// nothing but the callcaches themselves.
    fn mark(&self) -> usize {
        let mut marked = 0;
        for &slot in self.storage.iter() {
            if slot != 0 {
                marked += 1;
                unsafe { rb_gc_mark_movable(VALUE(slot)) };
            }
        }
        marked
    }

    /// Drop every entry, because compaction has moved the classes the slots were
    /// hashed from. Returns how many entries were live.
    fn clear(&mut self) -> usize {
        let mut dropped = 0;
        for slot in self.storage.iter_mut() {
            if *slot != 0 {
                *slot = 0;
                dropped += 1;
            }
        }
        dropped
    }

    /// Number of occupied slots, for `--zjit-stats`.
    pub fn occupancy(&self) -> usize {
        self.storage.iter().filter(|&&slot| slot != 0).count()
    }
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
        SendMegamorphic | SendAncestorGuardFallback | SendPolymorphic => {}
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
    let mut marked = 0u64;
    let mut probed = 0u64;
    for cache in ZJITState::get_send_caches().values() {
        marked += cache.mark() as u64;
        probed += cache.storage.len() as u64;
    }
    if get_option!(stats, /*default=*/false) {
        incr_counter_by(Counter::gc_mark_send_cache_slot_count, marked);
        incr_counter_by(Counter::gc_mark_send_cache_probe_count, probed);
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

    /// The shift has to keep exactly log2(len) bits, or the helper indexes out of
    /// bounds (too few) or wastes half the table (too many).
    #[test]
    fn shift_covers_the_whole_table() {
        for log2 in 3..=12u32 {
            let len = 1usize << log2;
            let shift = 64 - len.trailing_zeros();
            let max = u64::MAX >> shift;
            assert_eq!(max as usize, len - 1, "len {len}");
        }
    }

    /// Every slot index a class address can produce must be in range, including
    /// for the extreme words a `VALUE` can hold.
    #[test]
    fn slots_are_in_range() {
        for log2 in 3..=12u32 {
            let len = 1usize << log2;
            let shift = 64 - len.trailing_zeros();
            for klass in [0u64, 8, 0x10, 0xffff_ffff_ffff_fff8, 0x5555_5555_5555_5550] {
                let slot = (klass.wrapping_mul(SEND_CACHE_HASH_MULT) >> shift) as usize;
                assert!(slot < len, "len {len} klass {klass:#x} slot {slot}");
            }
        }
    }

    /// Nearby class addresses must not collide: they differ only in bits the
    /// multiply has to spread into the top of the word.
    #[test]
    fn adjacent_addresses_do_not_share_a_slot() {
        let len = 256usize;
        let shift = 64 - len.trailing_zeros();
        let slot = |klass: u64| (klass.wrapping_mul(SEND_CACHE_HASH_MULT) >> shift) as usize;
        // A heap page of 40-byte slots, which is what classes are allocated from.
        let slots: std::collections::HashSet<usize> =
            (0..64u64).map(|i| slot(0x7f00_0000_1000 + i * 40)).collect();
        // With a mask of the raw address instead of a multiply this would be a
        // handful of distinct slots; the bar is deliberately loose so the test
        // does not pin the exact hash.
        assert!(slots.len() > 48, "only {} distinct slots for 64 classes", slots.len());
    }
}
