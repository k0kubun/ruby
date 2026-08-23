//! Shape-id -> ivar-location tables for ivar accesses that miss their inline
//! shape guard chain.
//!
//! # Why
//!
//! ZJIT specializes an ivar access by profiling the receiver's shapes and
//! emitting a chain of `shape_id` guards, one arm per profiled shape (see
//! [`crate::hir::Function::dispatch_ivar`]). That works when a site sees a
//! handful of shapes. It does not work for code like Shopify's Storefront
//! Renderer, where a single class has hundreds of live shapes because instances
//! stop at different points in a long chain of conditionally-assigned ivars: the
//! chain cannot be widened far enough (each arm is code, and
//! [`crate::payload::MAX_IVAR_RESPECIALIZATIONS`] deliberately bounds the
//! recompiles), so the site falls off the end of the chain into a generic
//! `rb_ivar_get` / `rb_vm_getinstancevariable` call. On SFR that is ~13M generic
//! calls per 350 requests.
//!
//! The step between a guard chain and a megamorphic call is a *lookup table*:
//! instead of comparing the receiver's shape against N compile-time constants,
//! hash it and look the answer up. This module is that table.
//!
//! # Shape of the fast path
//!
//! Each ivar *name* read or written by such a site gets one [`IvarCache`]: a
//! fixed-size, direct-mapped array of 8-byte entries keyed by `shape_id`, each
//! holding the byte offset of that ivar within the receiver. Reads probe it
//! inline: `gen_ivar_cache_probe` in [`crate::codegen`] hashes the shape id it
//! already loaded for the guard chain, loads one entry, compares the key, and on
//! a hit produces the value with no call at all -- including when the answer is
//! `nil` because the shape does not have the ivar. Anything the probe cannot
//! serve calls [`rb_zjit_getivar_cached`], which fills the entry and answers.
//! Writes have no inline probe (a store also needs a frozen check and a write
//! barrier) but [`rb_zjit_setivar_cached`] resolves out of the same table.
//!
//! Keying by name rather than by site is deliberate. `shape_id -> location of
//! @name` does not depend on the site, so sites reading the same ivar want the
//! same table and warm it for each other; and the table has to be large, because
//! a direct-mapped table smaller than the shape working set thrashes instead of
//! degrading (see [`DEFAULT_CACHE_ENTRIES`]). Per site, that size would be
//! multiplied by the number of compiled sites -- and again by every recompile.
//! Per name it is multiplied by the number of ivar names the program actually
//! accesses polymorphically, which is far smaller and does not grow over time.
//!
//! # Why a stale entry can never be wrong
//!
//! An entry maps `shape_id -> (kind, offset)`. That mapping is immutable:
//!
//! * Shapes are immutable once created. `rb_shape_t::next_field_index` and
//!   `edge_name` are written when the node is allocated and never change, and
//!   the shape tree is append-only -- ids are never recycled or renumbered, so
//!   `rb_shape_get_iv_index(shape_id, id)` is a pure function of its arguments.
//! * Every mutation of an *object* that could move its ivars changes the
//!   object's `shape_id`, which changes the key:
//!   - adding an ivar transitions to a child shape;
//!   - `remove_instance_variable` goes through `rb_ivar_delete`, which
//!     transitions to a rebuilt shape (or to a complex shape);
//!   - `freeze` sets `SHAPE_ID_FL_FROZEN`, part of `shape_id`;
//!   - `object_id` sets `SHAPE_ID_FL_HAS_OBJECT_ID`, part of `shape_id`;
//!   - going "too complex" sets `SHAPE_ID_FL_COMPLEX`, part of `shape_id`;
//!   - GC compaction can change an object's embedded capacity and layout, both
//!     of which are part of `shape_id`.
//!   The key we store is the *raw* `shape_id` word as JIT code loads it from the
//!   object, with no bits masked off, so any of those changes is a table miss
//!   rather than a stale hit.
//!
//! There is therefore nothing to invalidate: no patch points, no
//! [`crate::invariants`] entry, no on-shape-redefinition hook. The table also
//! stores no `VALUE`s -- only a `shape_id`, a small offset, and a kind tag -- so
//! it keeps nothing alive and the GC never has to mark or update it.
//!
//! # Ractors
//!
//! An entry is one naturally-aligned `u64` and is published with a single
//! relaxed atomic store, so a reader (JIT code doing a plain 8-byte load, or
//! another ractor in [`rb_zjit_getivar_cached`]) sees either the old entry or
//! the new one, never a mix of the two. There is no other mutable state: the
//! table is the whole cache.
//!
//! The fast path reads a receiver's fields without a ractor check, which is what
//! the interpreter's own `vm_getivar` does for the same shapes, and what ZJIT's
//! shape-specialized path already does for the shapes its guard chain covers.
//! Classes, modules, structs and generic-ivar objects are the cases where the
//! check has teeth, and those are marked uncacheable and keep going through
//! `rb_ivar_get`, preserving its `RactorIsolationError` behaviour exactly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::cruby::*;
use crate::options::get_option;
use crate::stats::{Counter, incr_counter, incr_counter_by};
use crate::state::ZJITState;

/// Multiplier of the Fibonacci hash in [`slot_of`]. A `shape_id`'s bits are
/// grouped by meaning -- tree offset in 0..19, embedded capacity in 19..26, then
/// the complex/frozen/object-id/layout flags -- so an xor-fold cannot be trusted
/// to move the flag bits into the low bits the slot index comes from. (An
/// earlier version folded by 13, which left the frozen bit out of the index
/// entirely: a frozen and an unfrozen instance of the same shape shared a slot
/// and evicted each other on every access.) A multiply mixes every input bit
/// into the high bits, which is where the index is taken from.
pub const IVAR_CACHE_HASH_MULT: u64 = 0x9e37_79b9_7f4a_7c15;

/// Key of an unoccupied slot. `shape_id_t` only uses bits 0..31, so no real
/// shape id can collide with it.
pub const IVAR_CACHE_EMPTY_KEY: u32 = u32::MAX;

/// Entry bit layout, shared with the code `gen_ivar_cache_probe` emits.
pub const IVAR_CACHE_OFFSET_SHIFT: u64 = 32;
pub const IVAR_CACHE_KIND_SHIFT: u64 = 48;
/// Byte offsets of the two halves of an entry, which JIT code reads as separate
/// 32-bit loads: the shape id key, then the offset-and-kind word.
pub const IVAR_CACHE_KEY_OFFSET: i32 = 0;
pub const IVAR_CACHE_INFO_OFFSET: i32 = 4;
/// Mask JIT code tests the info word with to reject the kinds it cannot serve
/// inline, i.e. everything but [`EntryKind::Direct`] and [`EntryKind::Nil`]
/// (which is why those two are numbered 0 and 1).
pub const IVAR_CACHE_NOT_INLINE_MASK: u64 = 0xfffe_0000;
/// Bit JIT code tests the info word with to tell [`EntryKind::Nil`] from
/// [`EntryKind::Direct`] once [`IVAR_CACHE_NOT_INLINE_MASK`] has passed.
pub const IVAR_CACHE_NIL_BIT: u64 = 0x0001_0000;

/// Default of `--zjit-ivar-cache-entries`, i.e. 4KiB per ivar name.
///
/// This is the one number that decides whether the whole mechanism pays. A
/// direct-mapped table smaller than the shape working set does not degrade
/// gracefully: under a cyclic access pattern every slot with two live shapes in
/// it misses on *every* access, so the table stops paying for itself and the
/// probe becomes pure overhead. Measured on `benchmark/zjit_ivar_megashape.rb`,
/// 220 shapes per name, instructions retired per read relative to the generic
/// call: 64 slots 1.00, 256 slots 0.69, 512 slots 0.38, 1024 slots 0.24. 512 is
/// where the curve flattens for a few hundred shapes without spending 8KiB a
/// name.
pub const DEFAULT_CACHE_ENTRIES: usize = 512;

/// Number of slots in each table, from `--zjit-ivar-cache-entries`.
pub fn cache_entries() -> usize {
    crate::options::get_option!(ivar_cache_entries, DEFAULT_CACHE_ENTRIES)
}

/// Right shift JIT code applies to the hash product to get a *byte* offset into
/// the table, before masking with [`cache_byte_mask`].
pub fn cache_hash_shift() -> u64 {
    64 - cache_entries().trailing_zeros() as u64 - 3
}

/// Mask that turns the shifted hash into an in-range byte offset.
pub fn cache_byte_mask() -> u64 {
    (cache_entries() as u64 - 1) * 8
}

/// What an entry says about how to read the ivar.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum EntryKind {
    /// `T_OBJECT` with embedded fields: the ivar is at `recv + offset`. Served
    /// inline, which is why it is 0 (see [`IVAR_CACHE_NOT_INLINE_MASK`]).
    Direct = 0,
    /// This shape has no such ivar, so a read is `nil` (and a write is a shape
    /// transition). Reads are served inline, which is why it is 1.
    Nil = 1,
    /// An object whose fields live in a separate imemo/fields object -- a
    /// `T_OBJECT` that outgrew its embedded capacity, or a `T_DATA` -- so the
    /// ivar is at `*(recv + ROBJECT_OFFSET_AS_HEAP_FIELDS) + offset`. Both put
    /// the fields object at that same offset (`shape.h` asserts
    /// `offsetof(RObject, as.extended) == offsetof(RTypedData, fields_obj)`),
    /// which matters because a `T_OBJECT` and a `T_DATA` can share a shape id.
    /// Reads take it (like the shape-specialized path in
    /// [`crate::hir::Function::load_ivar`]); writes do not.
    Extended = 2,
    /// A class or module. Its fields live behind a writable-fields indirection
    /// that depends on the current namespace, so the location is an *index* into
    /// the fields object rather than a byte offset off the receiver, and reads go
    /// through `rb_ivar_get_at_no_ractor_check` -- the same helper the
    /// shape-specialized path uses (see
    /// [`crate::hir::Function::load_ivar_c_call`]) -- rather than being loaded
    /// here. Writes do not take it.
    RClass = 3,
    /// Complex (hash-backed) shapes, immediates, `T_STRUCT`, generic-ivar objects,
    /// and anything else this module declines to model. Serve with `rb_ivar_get`,
    /// but remember the decision so we do not re-derive it.
    Uncacheable = 4,
}

/// One decoded table entry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Entry {
    pub shape_id: u32,
    /// Byte offset of the ivar slot from the base the kind selects.
    pub offset: u16,
    pub kind: EntryKind,
}

impl Entry {
    pub fn pack(&self) -> u64 {
        (self.shape_id as u64)
            | ((self.offset as u64) << IVAR_CACHE_OFFSET_SHIFT)
            | ((self.kind as u64) << IVAR_CACHE_KIND_SHIFT)
    }

    pub fn unpack(word: u64) -> Self {
        let kind = match (word >> IVAR_CACHE_KIND_SHIFT) as u8 {
            0 => EntryKind::Direct,
            1 => EntryKind::Nil,
            2 => EntryKind::Extended,
            3 => EntryKind::RClass,
            _ => EntryKind::Uncacheable,
        };
        Entry {
            shape_id: word as u32,
            offset: (word >> IVAR_CACHE_OFFSET_SHIFT) as u16,
            kind,
        }
    }

    fn empty() -> Self {
        Entry { shape_id: IVAR_CACHE_EMPTY_KEY, offset: 0, kind: EntryKind::Direct }
    }
}

/// Slot a shape id maps to. Must match the arithmetic `gen_ivar_cache_probe`
/// emits, or JIT code and the helper will fill and probe different slots.
pub fn slot_of(shape_id: u32) -> usize {
    let hash = (shape_id as u64).wrapping_mul(IVAR_CACHE_HASH_MULT);
    (((hash >> cache_hash_shift()) & cache_byte_mask()) / 8) as usize
}

/// Word JIT code reads instead of a receiver field when an entry says the shape
/// does not have the ivar. Serving `nil` from the table matters because
/// conditionally-assigned ivars are exactly what makes a class shape-polymorphic
/// in the first place, so "absent here" is a common answer, and resolving it
/// generically is the most expensive lookup of all (the search has to fail).
#[used]
pub static IVAR_CACHE_NIL_SLOT: VALUE = Qnil;

/// One shape table, shared by every compiled site that reads the same ivar name.
///
/// Sharing by name rather than by site is both cheaper and more effective. The
/// mapping it caches, `shape_id -> location of @name`, does not depend on the
/// site at all, so two sites reading `@name` want exactly the same table and
/// warm it for each other. And the table has to be *large*: a shape-polymorphic
/// site walks through hundreds of shapes, and a direct-mapped table smaller than
/// the working set thrashes completely (measured: 64 slots against 220 shapes
/// evicted on 99.8% of accesses). Sizing per site would multiply that size by
/// the number of compiled sites; sizing per name multiplies it by the number of
/// ivar names the program actually reads polymorphically, which is far smaller
/// and does not grow when an ISEQ is recompiled.
pub struct IvarCache {
    /// Direct-mapped slots, `cache_entries()` of them. JIT code holds
    /// [`IvarCache::table_ptr`] directly and indexes it with a masked hash, so
    /// this must never be reallocated.
    entries: Box<[AtomicU64]>,
    /// The ivar this table is for.
    id: ID,
}

impl IvarCache {
    fn new(id: ID) -> Self {
        let empty = Entry::empty().pack();
        IvarCache {
            entries: (0..cache_entries()).map(|_| AtomicU64::new(empty)).collect(),
            id,
        }
    }

    /// Address of slot 0, which is what JIT code bakes in.
    pub fn table_ptr(&self) -> *const u8 {
        self.entries.as_ptr() as *const u8
    }

    fn load(&self, slot: usize) -> Entry {
        Entry::unpack(self.entries[slot].load(Ordering::Relaxed))
    }

    fn store(&self, slot: usize, entry: Entry) {
        self.entries[slot].store(entry.pack(), Ordering::Relaxed);
    }

    /// Bytes this cache owns on the Rust heap.
    pub fn heap_size(&self) -> usize {
        size_of::<IvarCache>() + self.entries.len() * size_of::<AtomicU64>()
    }
}

/// Table for `id`, creating it on first use. The `Box` is owned by [`ZJITState`]
/// so that [`crate::mem_stats`] can account for it; the returned pointer is
/// stable for the life of the process.
pub fn ivar_cache_for(id: ID) -> *const IvarCache {
    let caches = ZJITState::get_ivar_caches();
    let cache = caches.entry(id).or_insert_with(|| {
        incr_counter!(ivar_cache_alloc_count);
        Box::new(IvarCache::new(id))
    });
    cache.as_ref() as *const IvarCache
}

/// Decide how a shape id locates `id`. Depends on `shape_id` alone -- not on the
/// receiver -- which is what makes the answer safe to cache under that key even
/// though several classes, and even several `BUILTIN_TYPE`s, can share a shape.
fn resolve(shape_id: u32, id: ID) -> Entry {
    let uncacheable = Entry { shape_id, offset: 0, kind: EntryKind::Uncacheable };

    // Hash-backed shapes have no index to cache, and `rb_shape_get_iv_index`
    // asserts against them.
    if unsafe { rb_jit_shape_complex_p(shape_id) } {
        return uncacheable;
    }

    // T_STRUCT and generic-ivar objects are not worth another kind. Classes and
    // modules are: on a Rails workload they are most of what reaches this helper
    // at all -- reading `@abstract_class`, `@primary_key` and friends off model
    // classes accounted for 761K of the 800K calls to it on lobsters -- and
    // caching their index turns each of those from a shape-tree lookup into an
    // indexed load.
    let kind = match ShapeId(shape_id).layout() {
        ShapeLayout::RObject => EntryKind::Direct,
        ShapeLayout::Extended => EntryKind::Extended,
        ShapeLayout::RClass => EntryKind::RClass,
        ShapeLayout::Other => return uncacheable,
    };

    // Plain rb_shape_get_iv_index, not rb_shape_get_iv_index_with_hint. The
    // interpreter's IVC keeps a (shape, index) hint because its cache holds one
    // shape, so a miss is usually a *near relative* of what it holds. A table
    // miss here is the opposite: it means this shape is not among the hundreds
    // the table already knows, so the hint's ancestor walk is work thrown away
    // ahead of the ancestor-index lookup that actually answers the question.
    // Measured on the megashape benchmark, hinting cost 15-45% more instructions
    // per access than not hinting, and turned an undersized table from break-even
    // into a 15% regression.
    let mut index: attr_index_t = 0;
    if !unsafe { rb_shape_get_iv_index(shape_id, id, &mut index) } {
        return Entry { shape_id, offset: 0, kind: EntryKind::Nil };
    }

    // Fields live at the same offset in a T_OBJECT's embedded array and in the
    // imemo/fields object an extended one points at.
    let offset = ROBJECT_OFFSET_AS_ARY as usize + SIZEOF_VALUE * (index as usize);
    debug_assert!(offset <= u16::MAX as usize);
    Entry { shape_id, offset: offset as u16, kind }
}

/// Read the ivar an entry describes.
///
/// # Safety
/// `entry` must have been derived from `recv`'s current shape id.
unsafe fn load_entry(recv: VALUE, entry: Entry) -> VALUE {
    let base = match entry.kind {
        EntryKind::Direct => recv.as_usize(),
        EntryKind::Extended => unsafe {
            *((recv.as_usize() + ROBJECT_OFFSET_AS_HEAP_FIELDS as usize) as *const usize)
        },
        _ => unreachable!("load_entry on {:?}", entry.kind),
    };
    unsafe { *((base + entry.offset as usize) as *const VALUE) }
}

/// Slow path of a table-backed ivar read: called from JIT code when the inline
/// probe did not produce a value. Fills the site's table so the next receiver
/// with this shape is served inline.
///
/// Raises exactly what `rb_ivar_get` raises, and only from the paths that
/// delegate to it.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_getivar_cached(recv: VALUE, cache: *const IvarCache) -> VALUE {
    let cache = unsafe { &*cache };
    let id = cache.id;

    // Immediates have no ivars. `rb_ivar_get` returns nil for them too.
    if recv.special_const_p() {
        count(Counter::getivar_cache_immediate);
        return Qnil;
    }

    // Read the shape id field directly rather than calling rb_obj_shape_id():
    // this is the same word, at the same offset, that the inline probe loaded.
    let shape_id = unsafe { *((recv.as_usize() as isize + rb_shape_id_offset() as isize) as *const u32) };
    let slot = slot_of(shape_id);
    let entry = cache.load(slot);
    let entry = if entry.shape_id == shape_id {
        // The inline probe serves Direct and Nil; Extended and Uncacheable land
        // here on every access, but still skip re-deriving the location.
        count(Counter::getivar_cache_helper_hit);
        entry
    } else {
        count(if entry.shape_id == IVAR_CACHE_EMPTY_KEY {
            Counter::getivar_cache_fill
        } else {
            Counter::getivar_cache_evict
        });
        let entry = resolve(shape_id, id);
        cache.store(slot, entry);
        entry
    };

    match entry.kind {
        EntryKind::Direct | EntryKind::Extended => unsafe { load_entry(recv, entry) },
        EntryKind::Nil => Qnil,
        // `rb_ivar_get_at_no_ractor_check` skips the `RactorIsolationError` that reading a
        // class's unshareable ivar raises off the main ractor. That check can only fire once a
        // second ractor exists, so testing for one is exactly as strict as `rb_ivar_get` -- and
        // strictly stricter than the shape-specialized path, which reads the fields object
        // inline under nothing but the `SingleRactorMode` patch point.
        EntryKind::RClass if !unsafe { rb_jit_multi_ractor_p() } => {
            let index = (entry.offset as usize - ROBJECT_OFFSET_AS_ARY as usize) / SIZEOF_VALUE;
            unsafe { rb_ivar_get_at_no_ractor_check(recv, index as attr_index_t) }
        }
        EntryKind::RClass | EntryKind::Uncacheable => {
            count(Counter::getivar_cache_uncacheable);
            unsafe { rb_ivar_get(recv, id) }
        }
    }
}

/// Slow path of a table-backed ivar write, called from JIT code in place of the
/// generic `rb_ivar_set` / `rb_vm_setinstancevariable`.
///
/// Reads and writes want exactly the same answer -- where does `@name` live in
/// this shape -- so this shares the read path's table, and a read warms the table
/// for a write and vice versa. Only the interpretation differs: a shape that does
/// not have the ivar yet is `nil` to a read but a shape transition to a write, and
/// a write additionally has to reject frozen receivers.
///
/// Raises exactly what `rb_ivar_set` raises, and only by delegating to it.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_setivar_cached(recv: VALUE, val: VALUE, cache: *const IvarCache) {
    let cache = unsafe { &*cache };
    let id = cache.id;

    // Immediates raise FrozenError; let rb_ivar_set do it.
    if recv.special_const_p() {
        count(Counter::setivar_cache_uncacheable);
        unsafe { rb_ivar_set(recv, id, val); }
        return;
    }

    let shape_id = unsafe { *((recv.as_usize() as isize + rb_shape_id_offset() as isize) as *const u32) };
    // A frozen receiver has to raise, and a complex one has no index. Both are
    // visible in the shape id, so neither costs a table probe. Note this is a
    // property of the shape rather than of the entry: the entry for a frozen
    // shape is a perfectly good *read* location, and the read path uses it.
    if shape_id & (SHAPE_ID_FL_FROZEN | SHAPE_ID_FL_COMPLEX) != 0 {
        count(Counter::setivar_cache_uncacheable);
        unsafe { rb_ivar_set(recv, id, val); }
        return;
    }

    let slot = slot_of(shape_id);
    let entry = cache.load(slot);
    let entry = if entry.shape_id == shape_id {
        count(Counter::setivar_cache_hit);
        entry
    } else {
        count(if entry.shape_id == IVAR_CACHE_EMPTY_KEY {
            Counter::setivar_cache_fill
        } else {
            Counter::setivar_cache_evict
        });
        let entry = resolve(shape_id, id);
        cache.store(slot, entry);
        entry
    };

    match entry.kind {
        EntryKind::Direct => {
            // Exactly what vm_setivar does on an inline cache hit: the shape
            // already has the ivar, so the store needs no transition.
            unsafe { *((recv.as_usize() + entry.offset as usize) as *mut VALUE) = val };
            recv.write_barrier(val);
        }
        // Nil means the shape does not have this ivar yet, so the write is a
        // shape transition and possibly a reallocation. Extended means the
        // fields live in a separate object which may be a T_DATA needing
        // `rb_ivar_set`'s type dispatch, and RClass a writable-fields
        // indirection that copies on write in a non-root namespace. None is
        // worth modelling here: all are far off the hot path this table exists
        // for.
        EntryKind::Nil | EntryKind::Extended | EntryKind::RClass | EntryKind::Uncacheable => {
            count(Counter::setivar_cache_transition);
            unsafe { rb_ivar_set(recv, id, val); }
        }
    }
}

/// Bump a counter, but only when `--zjit-stats` asked for them: this runs on
/// every miss of a hot ivar site, and the whole point of the table is to keep
/// that path short.
fn count(counter: Counter) {
    if get_option!(stats, false) {
        incr_counter_by(counter, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_roundtrip() {
        for kind in [EntryKind::Direct, EntryKind::Extended, EntryKind::Nil, EntryKind::Uncacheable] {
            let entry = Entry { shape_id: 0x1234_5678, offset: 0x0abc, kind };
            assert_eq!(Entry::unpack(entry.pack()), entry);
        }
    }

    #[test]
    fn empty_entry_never_matches_a_shape_id() {
        // Shape ids only use bits 0..31, and the flag bits stop at bit 30.
        assert_eq!(Entry::empty().shape_id, IVAR_CACHE_EMPTY_KEY);
        assert!(SHAPE_ID_LAYOUT_MASK < IVAR_CACHE_EMPTY_KEY);
    }

    #[test]
    fn only_direct_and_nil_entries_pass_the_inline_probe() {
        // What JIT code tests: `(entry >> 32) & IVAR_CACHE_NOT_INLINE_MASK`, then
        // `& IVAR_CACHE_NIL_BIT` to pick between the receiver and the nil slot.
        let direct = Entry { shape_id: 7, offset: 0x18, kind: EntryKind::Direct };
        let word = direct.pack() >> IVAR_CACHE_OFFSET_SHIFT;
        assert_eq!(word & IVAR_CACHE_NOT_INLINE_MASK, 0);
        assert_eq!(word & IVAR_CACHE_NIL_BIT, 0);
        // The surviving low half is exactly the offset it should load with.
        assert_eq!(word & 0xffff, 0x18);

        let nil = Entry { shape_id: 7, offset: 0, kind: EntryKind::Nil };
        let word = nil.pack() >> IVAR_CACHE_OFFSET_SHIFT;
        assert_eq!(word & IVAR_CACHE_NOT_INLINE_MASK, 0);
        assert_ne!(word & IVAR_CACHE_NIL_BIT, 0);

        for kind in [EntryKind::Extended, EntryKind::Uncacheable] {
            let entry = Entry { shape_id: 7, offset: 0x18, kind };
            assert_ne!((entry.pack() >> IVAR_CACHE_OFFSET_SHIFT) & IVAR_CACHE_NOT_INLINE_MASK, 0);
        }
    }

    #[test]
    fn the_nil_slot_holds_nil() {
        assert_eq!(unsafe { *std::ptr::addr_of!(IVAR_CACHE_NIL_SLOT) }, Qnil);
    }

    // slot_of() reads --zjit-ivar-cache-entries, which is only initialized once
    // ZJIT has parsed its options, so this asserts the range against whatever
    // the default is rather than hard-coding a size.
    #[test]
    fn slots_are_in_range() {
        for shape_id in [0u32, 1, 0x1234, 0x7ffff, 0xffff_ffff, 0x2000_0001] {
            assert!(slot_of(shape_id) < cache_entries());
        }
    }
}
