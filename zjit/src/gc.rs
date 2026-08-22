//! This module is responsible for marking/moving objects on GC.

use std::collections::HashSet;
use std::ptr::null;
use std::{ffi::c_void, ops::Range};
use crate::{cruby::*, state::ZJITState, stats::with_time_stat, virtualmem::CodePtr};
use crate::payload::{IseqPayload, IseqVersionRef, get_iseq_payload_ptr};
use crate::options::get_option;
use crate::stats::{Counter, incr_counter, incr_counter_by};
use crate::stats::Counter::*;
use crate::bg_assume::Assumption;

/// The `VALUE`s ZJIT baked into one compiled version's machine code, and where in
/// the code region each one sits.
///
/// Two parallel arrays rather than an array of pairs, because the mark phase only
/// needs the objects: it hands `objects` to [`rb_gc_mark_values`] in a single call
/// and never touches the code region at all. It used to read every baked `VALUE`
/// back out of the code, which is a cache miss per object per major GC, scattered
/// over tens of megabytes of executable memory -- by far the most expensive thing
/// per object that ZJIT's GC hooks did.
///
/// The shadow copies are safe to mark in place of the code's because a baked `VALUE`
/// is only ever written twice: once by codegen, before [`GcOffsets::append`] records
/// it, and once by [`GcOffsets::update_references`] on a compacting GC, which writes
/// the shadow and the code from the same `rb_gc_location` result. Invalidation drops
/// entries from both arrays together ([`GcOffsets::remove_overlapping`]). Debug
/// builds re-read the code on every mark and assert the two still agree.
#[derive(Debug, Default)]
pub struct GcOffsets {
    /// Addresses of the baked `VALUE`s inside the code region.
    offsets: Vec<CodePtr>,
    /// The `VALUE` at each of those addresses, in the same order.
    objects: Vec<VALUE>,
}

impl GcOffsets {
    /// Record the `VALUE`s the backend baked into `offsets`, and write-barrier them
    /// against `iseq` (the payload that owns this version hangs off it, so the ISEQ
    /// is the old object that now points at them).
    fn append(&mut self, iseq: IseqPtr, offsets: &[CodePtr]) {
        let cb = ZJITState::get_code_block();
        self.offsets.extend(offsets);
        self.objects.reserve_exact(offsets.len());
        for &offset in offsets.iter() {
            // Creating an unaligned pointer is well defined unlike in C.
            let value_ptr = offset.raw_ptr(cb) as *const VALUE;
            let object = unsafe { value_ptr.read_unaligned() };
            self.objects.push(object);
            VALUE::from(iseq).write_barrier(object);
        }
        // These tables are written once per version and then only read, so hand back
        // whatever Vec's growth strategy over-reserved.
        self.offsets.shrink_to_fit();
        self.objects.shrink_to_fit();
    }

    /// Keep every baked object alive. One FFI call for the whole array, reading only
    /// this version's own dense array rather than the code region.
    fn mark(&self) {
        if self.objects.is_empty() {
            return;
        }
        self.debug_assert_shadow_matches_code();
        unsafe { rb_gc_mark_values(self.objects.len() as std::ffi::c_long, self.objects.as_ptr()) };
    }

    /// Follow every baked object to its new address after a compacting GC, writing
    /// the result to both the shadow array and the code.
    ///
    /// The code region is already writable because rb_zjit_mark_all_writable() was
    /// called before the GC update_references phase. We write directly to avoid
    /// per-page mprotect calls.
    fn update_references(&mut self) {
        let cb = ZJITState::get_code_block();
        for (&offset, object) in self.offsets.iter().zip(self.objects.iter_mut()) {
            let new_addr = unsafe { rb_gc_location(*object) };
            // Only write when the VALUE moves, to be copy-on-write friendly.
            if new_addr != *object {
                *object = new_addr;
                let value_ptr = offset.raw_ptr(cb) as *mut VALUE;
                unsafe { value_ptr.write_unaligned(new_addr) };
            }
        }
    }

    /// Drop the entries whose `VALUE` lands inside `removed_range`, which
    /// invalidation has just overwritten with a jump.
    fn remove_overlapping(&mut self, removed_range: &Range<CodePtr>) {
        debug_assert_eq!(self.offsets.len(), self.objects.len());
        let mut kept = 0;
        for index in 0..self.offsets.len() {
            let gc_offset = self.offsets[index];
            let offset_range = gc_offset..(gc_offset.add_bytes(SIZEOF_VALUE));
            if ranges_overlap(&offset_range, removed_range) {
                continue;
            }
            self.offsets[kept] = gc_offset;
            self.objects[kept] = self.objects[index];
            kept += 1;
        }
        self.offsets.truncate(kept);
        self.objects.truncate(kept);
    }

    /// Forget every entry. Used when the owning ISEQ is freed: the table is only
    /// read through the ISEQ, so with the ISEQ gone it is dead weight.
    pub fn clear(&mut self) {
        self.offsets = Vec::new();
        self.objects = Vec::new();
    }

    /// Number of baked objects this version has.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Bytes the two arrays own on the Rust heap.
    pub fn heap_size(&self) -> usize {
        self.offsets.capacity() * size_of::<CodePtr>()
            + self.objects.capacity() * size_of::<VALUE>()
    }

    /// Catch any future path that writes a baked `VALUE` without telling us, which
    /// would leave the mark phase marking a stale object. Debug builds only: this
    /// reads the code region, which is exactly what marking no longer does.
    #[inline(always)]
    fn debug_assert_shadow_matches_code(&self) {
        if cfg!(debug_assertions) {
            let cb = ZJITState::get_code_block();
            for (&offset, &object) in self.offsets.iter().zip(self.objects.iter()) {
                let value_ptr = offset.raw_ptr(cb) as *const VALUE;
                let in_code = unsafe { value_ptr.read_unaligned() };
                debug_assert_eq!(in_code, object,
                    "baked VALUE at {offset:?} changed without updating the GC shadow copy");
            }
        }
    }
}

/// The ISEQs the process-wide, append-only root tables keep raw pointers to,
/// deduplicated: one entry per distinct ISEQ rather than one per table entry.
///
/// [`crate::exit_meta`] and [`crate::jit_frame`] both hold raw `IseqPtr`s in tables
/// that are never freed, and [`rb_zjit_root_mark`] has to keep every one of those
/// ISEQs alive on *every* collection, minor ones included, because they are roots
/// and no write barrier covers them. Marking them entry by entry is what that used
/// to mean: on a large application the two tables hold hundreds of thousands of
/// entries between them, pointing at only as many distinct ISEQs as ZJIT has
/// compiled -- a factor of twenty or more of pure repetition, paid on every GC, plus
/// a cache miss per `JITFrame` because each one is its own heap allocation.
///
/// So the tables register their ISEQ with this set when they are appended to, and
/// the mark phase walks the set instead. That is sound because it marks a *superset*
/// of what walking the entries would: entries are never removed from either table,
/// so an ISEQ that ever entered the set has to stay alive for as long as the process
/// does either way. Retention is therefore unchanged; only the walk is smaller.
///
/// Compaction is handled in [`RootIseqs::update_references`]: the pointers here move
/// with the ISEQs, and the individual table entries are still fixed up one by one
/// (that phase runs only on a compacting GC, not on every collection).
#[derive(Default, Debug)]
pub struct RootIseqs {
    /// The distinct ISEQs, as `VALUE` so that the mark phase can hand the whole
    /// array to [`rb_gc_mark_values`] in one call.
    iseqs: Vec<VALUE>,
    /// Membership index over `iseqs`, so registering stays O(1) while compiling.
    /// Rebuilt from `iseqs` after compaction.
    seen: HashSet<VALUE>,
}

impl RootIseqs {
    /// Note that some root table now points at `iseq`. Cheap and idempotent; called
    /// once per table append, always with the GVL held (both tables are only
    /// appended to from phases 1 and 3 of a compile, never from the GVL-free phase).
    pub fn register(&mut self, iseq: IseqPtr) {
        if iseq.is_null() {
            return;
        }
        let value = VALUE::from(iseq);
        if self.seen.insert(value) {
            self.iseqs.push(value);
        }
    }

    /// Keep every ISEQ in the set alive. One FFI call for the whole array.
    fn mark(&self) {
        if self.iseqs.is_empty() {
            return;
        }
        unsafe { rb_gc_mark_values(self.iseqs.len() as std::ffi::c_long, self.iseqs.as_ptr()) };
    }

    /// Follow the ISEQs to their new addresses after a compacting GC and reindex.
    fn update_references(&mut self) {
        for iseq in self.iseqs.iter_mut() {
            *iseq = unsafe { rb_gc_location(*iseq) };
        }
        self.seen.clear();
        self.seen.extend(self.iseqs.iter().copied());
    }

    /// Number of distinct ISEQs the root tables reference.
    pub fn len(&self) -> usize {
        self.iseqs.len()
    }

    /// Bytes this set owns on the Rust heap.
    pub fn heap_size(&self) -> usize {
        self.iseqs.capacity() * size_of::<VALUE>()
            + crate::mem_stats::hash_table_bytes::<VALUE>(self.seen.capacity())
    }
}

/// Note that a root table entry now points at `iseq`. See [`RootIseqs`].
pub fn register_root_iseq(iseq: IseqPtr) {
    crate::bgcompile::assert_gvl_held("gc::register_root_iseq");
    ZJITState::get_root_iseqs().register(iseq);
}

/// Whether the GC callbacks should account for their own time and the objects they
/// visit. `rb_zjit_iseq_mark` runs once per live ISEQ payload per collection --
/// tens of thousands of calls per GC on a large application -- so an unconditional
/// `Instant::now()` pair plus a counter bump per callback is itself a measurable
/// slice of the GC pause we are trying to shrink. Under `--zjit-stats` we pay it to
/// get the breakdown; otherwise the hooks run untimed.
#[inline(always)]
fn gc_stats_p() -> bool {
    get_option!(stats, /*default=*/false)
}

/// [`with_time_stat`], but only when `--zjit-stats` asked for the numbers, and
/// charging the elapsed time to `gc_time_ns` as well as to `counter` so that the
/// per-callback breakdown always sums to the total.
#[inline(always)]
fn time_gc_hook<F, R>(counter: Counter, func: F) -> R where F: FnOnce() -> R {
    if !gc_stats_p() {
        return func();
    }
    let start = std::time::Instant::now();
    let ret = func();
    let nanos = start.elapsed().as_nanos() as u64;
    incr_counter_by(counter, nanos);
    incr_counter_by(gc_time_ns, nanos);
    ret
}

/// Like [`time_gc_hook`] but charges only `counter`, for a phase nested inside a
/// callback that `time_gc_hook` already charged to `gc_time_ns`.
#[inline(always)]
fn time_gc_phase<F, R>(counter: Counter, func: F) -> R where F: FnOnce() -> R {
    if !gc_stats_p() {
        return func();
    }
    with_time_stat(counter, func)
}

/// GC callback for marking GC objects in the per-ISEQ payload.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_iseq_mark(payload: *mut c_void) {
    let payload = if payload.is_null() {
        return; // nothing to mark
    } else {
        // SAFETY: The GC takes the VM lock while marking, which
        // we assert, so we should be synchronized and data race free.
        //
        // For aliasing, having the VM lock hopefully also implies that no one
        // else has an overlapping &mut IseqPayload.
        // A mutable borrow, unlike the name "mark" suggests: the profile caches the
        // dense array of objects it references (see
        // [`crate::profile::IseqProfile::marked_objects`]) and rebuilds it here when
        // a mutation invalidated it. The aliasing argument is the same as for
        // `rb_zjit_iseq_update_references` below, and nothing this callback reaches
        // runs Ruby code or re-enters ZJIT.
        unsafe {
            rb_assert_holding_vm_lock();
            &mut *(payload as *mut IseqPayload)
        }
    };
    if gc_stats_p() { incr_counter!(gc_iseq_mark_count); }
    time_gc_hook(gc_iseq_mark_time_ns, || iseq_mark(payload));
}

/// GC callback for updating GC objects in the per-ISEQ payload.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_iseq_update_references(payload: *mut c_void) {
    let payload = if payload.is_null() {
        return; // nothing to update
    } else {
        // SAFETY: The GC takes the VM lock while marking, which
        // we assert, so we should be synchronized and data race free.
        //
        // For aliasing, having the VM lock hopefully also implies that no one
        // else has an overlapping &mut IseqPayload.
        unsafe {
            rb_assert_holding_vm_lock();
            &mut *(payload as *mut IseqPayload)
        }
    };
    if gc_stats_p() { incr_counter!(gc_iseq_update_count); }
    time_gc_hook(gc_iseq_update_time_ns, || iseq_update_references(payload));
}

/// GC callback for finalizing an ISEQ
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_iseq_free(iseq: IseqPtr) {
    if !ZJITState::has_instance() {
        return;
    }

    // A background compilation in its GVL-free phase may hold this ISEQ as a
    // JIT-to-JIT callee or an EP-escape assumption. It would bake the pointer into
    // machine code, so discard it instead. See [`crate::bg_assume`].
    crate::bgcompile::note_invalidation(Assumption::Iseq(iseq));

    ZJITState::get_invariants().forget_iseq(iseq);

    // Do *not* create a payload here. Most ISEQs a process frees were never hot
    // enough for ZJIT to touch, and allocating a payload for one on its way out
    // used to retain `size_of::<IseqPayload>()` forever for nothing. On a Rails
    // app that was the single largest ZJIT heap consumer.
    let payload_ptr = get_iseq_payload_ptr(iseq);
    if payload_ptr.is_null() {
        return;
    }
    crate::stats::incr_counter!(dead_iseq_payload_count);

    // Take ownership of the payload and unhook it from the ISEQ, so the GC
    // callbacks above can no longer reach it while we tear it down. Dropping the
    // `Box` at the end of this function frees the `IseqPayload` allocation, the
    // profile, the exception entry table and the version vectors' buffers.
    let mut payload = unsafe { Box::from_raw(payload_ptr) };
    unsafe { rb_iseq_set_zjit_payload(iseq, std::ptr::null_mut()) };

    for version in payload.all_versions_mut() {
        let version = unsafe { version.as_mut() };
        version.iseq = null();
        // GC offsets are only read by iseq_mark(), which the GC reaches through
        // the ISEQ. With the ISEQ gone nothing marks this code again, so the
        // table is dead weight.
        version.gc_offsets.clear();
        // The `IseqVersion` allocation itself has to outlive the ISEQ: patch
        // points in `Invariants` hold raw pointers to it and are only dropped
        // when the assumption they guard is broken. Dropping the payload below
        // frees the `Vec` of pointers, not the pointees, so record what stays
        // behind or it would vanish from the `mem_*` breakdown.
        ZJITState::add_dead_iseq_version_bytes(version.total_heap_size());
    }

    // Everything the payload owns outright is reachable only through the ISEQ,
    // which is gone: the profile is only read while building HIR for this ISEQ
    // or for a caller that inlines it (the ISEQ can no longer be called nor be
    // the target of a send in newly compiled code), and the exception entry
    // table is only reached through `body->jit_exception`.
    //
    // The re-profiling paths (`rb_zjit_ivar_reprofile`) cannot resurrect the
    // profile either: they reach one only through
    // `get_or_create_iseq_payload(frame_iseq)` of a *running* frame's ISEQ,
    // which is by definition alive, and they hold no pointer into any
    // `IseqProfile` across a call.
    drop(payload);
}

/// GC callback for finalizing a CME
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_cme_free(cme: *const rb_callable_method_entry_struct) {
    if !ZJITState::has_instance() {
        return;
    }
    crate::bgcompile::note_invalidation(Assumption::Cme(cme));
    let invariants = ZJITState::get_invariants();
    invariants.forget_cme(cme);
}

/// GC callback for finalizing a class
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_klass_free(klass: VALUE) {
    if !ZJITState::has_instance() {
        return;
    }
    crate::bgcompile::note_invalidation(Assumption::Klass(klass));
    let invariants = ZJITState::get_invariants();
    invariants.forget_klass(klass);
}

/// GC callback for updating object references after all object moves
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_root_update_references() {
    if !ZJITState::has_instance() {
        return;
    }
    if gc_stats_p() { incr_counter!(gc_root_update_count); }
    time_gc_hook(gc_root_update_time_ns, root_update_references);
}

fn root_update_references() {
    let invariants = ZJITState::get_invariants();
    invariants.update_references();

    // The deduplicated set the mark phase walks, and then the tables it summarizes.
    ZJITState::get_root_iseqs().update_references();

    // Update iseq pointers in all JITFrames for GC compaction.
    // rb_execution_context_update only updates JITFrames currently on the stack,
    // but JITFrames not on the stack also need their iseq pointers updated
    // because the JIT code will reuse them on the next call.
    for &jit_frame in ZJITState::get_jit_frames().iter() {
        unsafe { &mut *jit_frame }.update_references();
    }

    // Side-exit metadata holds raw ISEQ pointers for exits that have not run yet.
    for meta in ZJITState::get_exit_metas().iter_mut() {
        meta.update_references();
    }

    // Send class tables are keyed on class addresses, which this compaction has
    // just changed, so they are dropped rather than rehashed.
    crate::send_cache::update_references();

    // ISEQs waiting for the background compile thread, and the thread itself.
    crate::bgcompile::update_references();

    // A compilation in its GVL-free phase holds raw pointers to ISEQs, classes,
    // CMEs and baked-in objects that this compaction has just moved, and cannot be
    // walked from here because the thread that owns it is running. Discard it.
    crate::bgcompile::note_compaction();
}

fn iseq_mark(payload: &mut IseqPayload) {
    // Mark objects retained by profiling instructions
    let profile = &mut payload.profile;
    time_gc_phase(gc_iseq_mark_profile_time_ns, || {
        let objects = profile.marked_objects();
        if !objects.is_empty() {
            unsafe { rb_gc_mark_values(objects.len() as std::ffi::c_long, objects.as_ptr()) };
        }
        if gc_stats_p() { incr_counter_by(gc_mark_profile_object_count, objects.len() as u64); }
    });

    // Mark objects baked in JIT code
    time_gc_phase(gc_iseq_mark_offsets_time_ns, || {
        let mut marked = 0u64;
        for version in payload.all_versions() {
            let gc_offsets = &unsafe { version.as_ref() }.gc_offsets;
            marked += gc_offsets.len() as u64;
            gc_offsets.mark();
        }
        if gc_stats_p() { incr_counter_by(gc_mark_offset_object_count, marked); }
    });
}

/// This is a mirror of [iseq_mark].
fn iseq_update_references(payload: &mut IseqPayload) {
    // Move objects retained by profiling instructions
    payload.profile.each_object_mut(|old_object| {
        let new_object = unsafe { rb_gc_location(*old_object) };
        if *old_object != new_object {
            *old_object = new_object;
        }
    });

    for &version in payload.all_versions() {
        iseq_version_update_references(version);
    }
}

fn iseq_version_update_references(mut version: IseqVersionRef) {
    // Move ISEQ in the payload
    unsafe { version.as_mut() }.iseq = unsafe { rb_gc_location(version.as_ref().iseq.into()) }.as_iseq();

    // Move ISEQ references in incoming IseqCalls
    for iseq_call in unsafe { version.as_mut() }.incoming.iter_mut() {
        let old_iseq = iseq_call.iseq.get();
        let new_iseq = unsafe { rb_gc_location(VALUE(old_iseq as usize)) }.0 as IseqPtr;
        if old_iseq != new_iseq {
            iseq_call.iseq.set(new_iseq);
        }
    }

    // Move ISEQ references in outgoing IseqCalls
    for iseq_call in unsafe { version.as_mut() }.outgoing.iter_mut() {
        let old_iseq = iseq_call.iseq.get();
        let new_iseq = unsafe { rb_gc_location(VALUE(old_iseq as usize)) }.0 as IseqPtr;
        if old_iseq != new_iseq {
            iseq_call.iseq.set(new_iseq);
        }
    }

    // Move objects baked in JIT code
    unsafe { version.as_mut() }.gc_offsets.update_references();
}

/// Append a set of gc_offsets to the iseq's payload
pub fn append_gc_offsets(iseq: IseqPtr, mut version: IseqVersionRef, offsets: &Vec<CodePtr>) {
    unsafe { version.as_mut() }.gc_offsets.append(iseq, offsets);
}

/// Remove GC offsets that overlap with a given removed_range.
/// We do this when invalidation rewrites some code with a jump instruction
/// and GC offsets are corrupted by the rewrite, assuming no on-stack code
/// will step into the instruction with the GC offsets after invalidation.
pub fn remove_gc_offsets(mut version: IseqVersionRef, removed_range: &Range<CodePtr>) {
    unsafe { version.as_mut() }.gc_offsets.remove_overlapping(removed_range);
}

/// Return true if given `Range<CodePtr>` ranges overlap with each other
fn ranges_overlap<T>(left: &Range<T>, right: &Range<T>) -> bool where T: PartialOrd {
    left.start < right.end && right.start < left.end
}

/// GC callback for making all JIT code writable before updating references in bulk.
/// This avoids toggling W^X permissions per-page during GC compaction.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_mark_all_writable() {
    if !ZJITState::has_instance() {
        return;
    }
    ZJITState::get_code_block().mark_all_writable();
}

/// GC callback for making all JIT code executable after updating references in bulk.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_mark_all_executable() {
    if !ZJITState::has_instance() {
        return;
    }
    ZJITState::get_code_block().mark_all_executable();
}

/// Callback for marking GC objects inside [crate::invariants::Invariants].
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_root_mark() {
    if !ZJITState::has_instance() {
        return;
    }
    if gc_stats_p() { incr_counter!(gc_root_mark_count); }
    time_gc_hook(gc_root_mark_time_ns, || {
        // Keep alive every ISEQ a JITFrame or a side exit holds a raw pointer to.
        //
        // JITFrames that are currently on the stack are also marked via
        // rb_execution_context_mark, but JITFrames not on the stack still need their
        // iseqs kept alive because JIT code will reuse them; likewise a side exit
        // that has not run yet has to be able to resume into (or recompile) its
        // ISEQ. Both tables register into [`RootIseqs`] as they grow, so this marks
        // one entry per distinct ISEQ instead of one per table entry.
        time_gc_phase(gc_root_mark_iseq_time_ns, || {
            let root_iseqs = ZJITState::get_root_iseqs();
            root_iseqs.mark();
            if gc_stats_p() { incr_counter_by(gc_mark_root_iseq_count, root_iseqs.len() as u64); }
        });
        // Keep alive the callcaches megamorphic send sites dispatch through. Nothing
        // else roots them: the class's own callcache table drops one as soon as the
        // method is invalidated. See [`crate::send_cache`].
        time_gc_phase(gc_root_mark_send_cache_time_ns, crate::send_cache::mark_all);
        // Nothing else keeps an ISEQ sitting in the background compile queue alive:
        // the interpreter may stop calling it, or its defining module may be
        // discarded, between the enqueue and the compile. See [`crate::bgcompile`].
        time_gc_phase(gc_root_mark_bgcompile_time_ns, crate::bgcompile::mark);
    });
}
