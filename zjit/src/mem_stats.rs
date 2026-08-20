//! Breakdown of `zjit_alloc_bytes` (ZJIT's Rust heap usage) by subsystem.
//!
//! `zjit_alloc_bytes` is a single number produced by the global allocator
//! wrapper in the `jit` crate, so it tells you *how much* ZJIT has allocated
//! but not *what for*. On large applications the non-code metadata is several
//! times the size of the generated code, and `--zjit-mem-size` caps the sum of
//! the two, so knowing the composition is the difference between guessing and
//! fixing.
//!
//! This module walks the structures ZJIT retains and reports the bytes each
//! subsystem owns. Vec-backed sizes are exact (capacity times element size);
//! hash-table sizes are close approximations of what hashbrown asks the
//! allocator for (see [`hash_table_bytes`]). The residual between
//! `zjit_alloc_bytes` and the sum of the categories is reported as
//! `mem_unaccounted_bytes`.

use crate::cruby::{IseqPtr, for_each_iseq, rb_iseq_get_jit_payload};
use crate::payload::IseqPayload;
use crate::state::ZJITState;

/// Number of control bytes hashbrown allocates past the end of the bucket
/// array. This is `Group::WIDTH`, 16 on x86-64 (SSE2) and aarch64 (NEON).
const HASHBROWN_GROUP_WIDTH: usize = 16;

/// Approximate the bytes hashbrown (the backing store of `std`'s `HashMap` and
/// `HashSet`) requests from the allocator for a table whose `capacity()` is
/// `capacity` and whose element type is `T`.
///
/// hashbrown rounds the requested capacity up to a power-of-two bucket count
/// and allocates `buckets * size_of::<T>()` for the entries plus
/// `buckets + Group::WIDTH` control bytes. Padding for alignment is ignored, so
/// this can be a few bytes low per table.
pub fn hash_table_bytes<T>(capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    // Inverse of hashbrown's capacity_to_buckets(): buckets 4 hold 3 items,
    // buckets 8 hold 7, and beyond that capacity is 7/8 of the bucket count.
    let buckets = if capacity <= 3 {
        4
    } else if capacity <= 7 {
        8
    } else {
        (capacity * 8 / 7).next_power_of_two()
    };
    buckets * size_of::<T>() + buckets + HASHBROWN_GROUP_WIDTH
}

/// Bytes retained by each ZJIT subsystem, in the same units as
/// `zjit_alloc_bytes`.
#[derive(Default, Debug)]
pub struct MemoryBreakdown {
    /// `IseqPayload` structs, one per ISEQ ZJIT has ever touched.
    pub iseq_payload_bytes: usize,
    /// Per-instruction profiling data (type/shape distributions) inside payloads.
    pub profile_bytes: usize,
    /// `IseqVersion` structs plus the `Vec` of version pointers in each payload.
    pub iseq_version_bytes: usize,
    /// GC offset tables (addresses of `VALUE`s baked into JIT code).
    pub gc_offset_bytes: usize,
    /// JIT-to-JIT call metadata: `IseqCall` allocations and the incoming and
    /// outgoing edge vectors that point at them.
    pub iseq_call_bytes: usize,
    /// Patch-point tables used to invalidate speculative code (`Invariants`).
    pub invariant_bytes: usize,
    /// `JITFrame`s: compile-time frame metadata plus their trailing stack maps.
    pub jit_frame_bytes: usize,
    /// `CodeBlock` bookkeeping: label tables and (with `--zjit-dump-disasm`)
    /// assembly comments.
    pub code_block_bytes: usize,
    /// String-keyed counter tables that only `--zjit-stats` populates.
    pub stats_counter_bytes: usize,
    /// Shape tables for ivar accesses that miss their inline guard chain.
    pub ivar_cache_bytes: usize,
    /// Class tables for send sites that dispatch over more classes than an
    /// inline guard chain can cover.
    pub send_cache_bytes: usize,
    /// Interpreter state for compiled side exits. Trading these bytes for
    /// executable ones is the point of the exercise: they used to be immediates
    /// in the exit stubs. See [`crate::exit_meta`].
    pub exit_meta_bytes: usize,
    /// The deduplicated set of ISEQs the JITFrame and ExitMeta tables reference,
    /// which is what the GC mark phase walks in their place. See
    /// [`crate::gc::RootIseqs`].
    pub root_iseq_bytes: usize,

    /// Number of ISEQ payloads walked, for per-ISEQ math.
    pub payload_count: usize,
    /// Number of `IseqVersion`s reachable from those payloads.
    pub version_count: usize,
    /// Number of per-instruction profile entries in those payloads.
    pub profile_entry_count: usize,
    /// Bytes sitting in the unused tail of profile `entries` vectors.
    pub profile_entry_slack_bytes: usize,
    /// Number of operand type distributions across all profiles.
    pub profile_distribution_count: usize,
    /// How many of those distributions saw at most one type.
    pub profile_monomorphic_distribution_count: usize,
    /// Number of patch points in `Invariants`.
    pub patch_point_count: usize,
    /// Number of live `JITFrame`s.
    pub jit_frame_count: usize,
    /// Number of ivar shape tables, i.e. distinct ivar names with one.
    pub ivar_cache_count: usize,
    /// Number of send class tables, i.e. distinct call shapes with one.
    pub send_cache_count: usize,
    /// Number of interned `ExitMeta` records.
    pub exit_meta_count: usize,
    /// Number of distinct ISEQs in the root set. The ratio against
    /// `jit_frame_count + 2 * exit_meta_count` is what deduplication saves the
    /// mark phase on every collection.
    pub root_iseq_count: usize,
    /// Number of objects in the dense arrays GC marking walks, i.e. the distinct
    /// objects the profiles reference. See
    /// [`crate::profile::IseqProfile::marked_objects`].
    pub profile_marked_object_count: usize,
}

impl MemoryBreakdown {
    /// Sum of every byte category above.
    pub fn accounted_bytes(&self) -> usize {
        self.iseq_payload_bytes
            + self.profile_bytes
            + self.iseq_version_bytes
            + self.gc_offset_bytes
            + self.iseq_call_bytes
            + self.invariant_bytes
            + self.jit_frame_bytes
            + self.code_block_bytes
            + self.stats_counter_bytes
            + self.ivar_cache_bytes
            + self.send_cache_bytes
            + self.exit_meta_bytes
            + self.root_iseq_bytes
    }
}

/// Walk everything ZJIT retains on the Rust heap and attribute it to a
/// subsystem. Requires the VM lock (it iterates over every live ISEQ).
pub fn memory_breakdown() -> MemoryBreakdown {
    let mut out = MemoryBreakdown::default();

    // Per-ISEQ payloads. Only ISEQs that are still alive are visited; ZJIT
    // currently never frees the payload of a dead ISEQ, so those bytes show up
    // in mem_unaccounted_bytes rather than here.
    for_each_iseq(|iseq: IseqPtr| {
        let payload = unsafe { rb_iseq_get_jit_payload(iseq) } as *const IseqPayload;
        if payload.is_null() {
            return;
        }
        let payload = unsafe { &*payload };
        out.payload_count += 1;
        out.iseq_payload_bytes += size_of::<IseqPayload>();

        let profile = payload.profile.heap_size();
        out.profile_bytes += profile.bytes;
        out.profile_entry_count += profile.entry_count;
        out.profile_entry_slack_bytes += profile.entry_slack_bytes;
        out.profile_distribution_count += profile.distribution_count;
        out.profile_monomorphic_distribution_count += profile.monomorphic_distribution_count;
        out.profile_marked_object_count += profile.marked_object_count;

        out.iseq_version_bytes += payload.versions.capacity() * size_of::<crate::payload::IseqVersionRef>();
        for version in payload.versions.iter() {
            let version = unsafe { version.as_ref() };
            out.version_count += 1;
            out.iseq_version_bytes += size_of::<crate::payload::IseqVersion>();
            out.gc_offset_bytes += version.gc_offsets.heap_size();
            out.iseq_call_bytes += version.iseq_call_heap_size();
        }
    });

    let invariants = ZJITState::get_invariants();
    let (invariant_bytes, patch_point_count) = invariants.heap_size();
    out.invariant_bytes = invariant_bytes;
    out.patch_point_count = patch_point_count;

    let jit_frames = ZJITState::get_jit_frames();
    out.jit_frame_count = jit_frames.len();
    out.jit_frame_bytes = jit_frames.capacity() * size_of::<*mut crate::jit_frame::JITFrame>()
        + jit_frames.iter().map(|&frame| unsafe { &*frame }.heap_size()).sum::<usize>();

    let exit_metas = ZJITState::get_exit_metas();
    out.exit_meta_count = exit_metas.len();
    out.exit_meta_bytes = exit_metas.capacity() * size_of::<crate::exit_meta::ExitMeta>();

    let root_iseqs = ZJITState::get_root_iseqs();
    out.root_iseq_count = root_iseqs.len();
    out.root_iseq_bytes = root_iseqs.heap_size();

    let ivar_caches = ZJITState::get_ivar_caches();
    out.ivar_cache_count = ivar_caches.len();
    out.ivar_cache_bytes = hash_table_bytes::<(crate::cruby::ID, Box<crate::ivar_cache::IvarCache>)>(ivar_caches.capacity())
        + ivar_caches.values().map(|cache| size_of::<crate::ivar_cache::IvarCache>() + cache.heap_size()).sum::<usize>();

    let send_caches = ZJITState::get_send_caches();
    out.send_cache_count = send_caches.len();
    out.send_cache_bytes = crate::send_cache::send_caches_heap_size(send_caches);

    out.code_block_bytes = ZJITState::get_code_block().heap_size();
    out.stats_counter_bytes = ZJITState::counter_table_heap_size();

    out
}
