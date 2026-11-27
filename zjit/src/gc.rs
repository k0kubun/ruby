//! This module is responsible for marking/moving objects on GC.

use std::ptr::null;
use std::{ffi::c_void, ops::Range};
use crate::{cruby::*, state::ZJITState, stats::with_time_stat, virtualmem::CodePtr};
use crate::payload::{IseqPayload, IseqVersionRef, get_iseq_payload_ptr};
use crate::stats::Counter::gc_time_ns;

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
        unsafe {
            rb_assert_holding_vm_lock();
            &*(payload as *const IseqPayload)
        }
    };
    with_time_stat(gc_time_ns, || iseq_mark(payload));
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
    with_time_stat(gc_time_ns, || iseq_update_references(payload));
}

/// GC callback for finalizing an ISEQ
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_iseq_free(iseq: IseqPtr) {
    if !ZJITState::has_instance() {
        return;
    }

    ZJITState::get_invariants().forget_iseq(iseq);

    // If ZJIT has never created a payload for this ISEQ, do nothing.
    let payload_ptr = get_iseq_payload_ptr(iseq);
    if payload_ptr.is_null() {
        return;
    }

    // Take ownership of the payload and unset it from the ISEQ.
    let payload = unsafe { Box::from_raw(payload_ptr) };
    unsafe { rb_iseq_clear_jit_payload(iseq) };

    // Clear IseqVersion references. Patch points may hold raw pointers to them, so
    // they have to outlive the ISEQ. They're dropped when the assumption is broken.
    for &version in payload.versions.iter() {
        unsafe { (*version.as_ptr()).iseq = null() };
    }

    // Free the IseqPayload.
    drop(payload);
}

/// GC callback for finalizing a CME
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_cme_free(cme: *const rb_callable_method_entry_struct) {
    if !ZJITState::has_instance() {
        return;
    }
    let invariants = ZJITState::get_invariants();
    invariants.forget_cme(cme);
}

/// GC callback for finalizing a class
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_klass_free(klass: VALUE) {
    if !ZJITState::has_instance() {
        return;
    }
    let invariants = ZJITState::get_invariants();
    invariants.forget_klass(klass);
}

/// GC callback for updating object references after all object moves
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_root_update_references() {
    if !ZJITState::has_instance() {
        return;
    }
    let invariants = ZJITState::get_invariants();
    invariants.update_references();

    // Update iseq pointers in JITFrames not owned by an IseqVersion for GC compaction.
    // rb_execution_context_update only updates JITFrames currently on the stack,
    // but JITFrames not on the stack also need their iseq pointers updated
    // because the JIT code will reuse them on the next call.
    // SAFETY: Only pointers to mutable memory from allocator(s) make it into the list.
    for &jit_frame in ZJITState::get_jit_frames().iter() {
        unsafe { &mut *jit_frame }.update_references();
    }
    ZJITState::get_jit_frame_iseqs().update_references();
}

fn iseq_mark(payload: &IseqPayload) {
    // Mark objects retained by profiling instructions
    payload.profile.each_object(|object| {
        unsafe { rb_gc_mark_movable(object); }
    });

    // Mark objects baked in JIT code
    let cb = ZJITState::get_code_block();
    for version in payload.versions.iter() {
        let version = unsafe { version.as_ref() };

        // Mark ISEQs of JITFrames, e.g. inlined callees, and callees of JIT-to-JIT calls
        for &iseq in version.jit_frame_iseqs.iter() {
            unsafe { rb_gc_mark_movable(VALUE::from(iseq)); }
        }
        for iseq_call in version.outgoing.iter() {
            unsafe { rb_gc_mark_movable(VALUE::from(iseq_call.iseq.get())); }
        }

        for &offset in version.gc_offsets.iter() {
            let value_ptr: *const u8 = offset.raw_ptr(cb);
            // Creating an unaligned pointer is well defined unlike in C.
            let value_ptr = value_ptr as *const VALUE;

            unsafe {
                let object = value_ptr.read_unaligned();
                rb_gc_mark_movable(object);
            }
        }
    }
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

    for &version in payload.versions.iter() {
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

    // Move ISEQ references in JITFrames
    for &jit_frame in unsafe { version.as_ref() }.jit_frames.iter() {
        unsafe { &mut *jit_frame }.update_references();
    }
    for iseq in unsafe { version.as_mut() }.jit_frame_iseqs.iter_mut() {
        *iseq = unsafe { rb_gc_location(VALUE::from(*iseq)) }.as_iseq();
    }

    // Move ISEQ references in outgoing IseqCalls
    for iseq_call in unsafe { version.as_mut() }.outgoing.iter_mut() {
        let old_iseq = iseq_call.iseq.get();
        let new_iseq = unsafe { rb_gc_location(VALUE(old_iseq as usize)) }.0 as IseqPtr;
        if old_iseq != new_iseq {
            iseq_call.iseq.set(new_iseq);
        }
    }

    // Move objects baked in JIT code.
    // The code region is already writable because rb_zjit_mark_all_writable() was called
    // before the GC update_references phase. We write directly to avoid per-page mprotect calls.
    let cb = ZJITState::get_code_block();
    for &offset in unsafe { version.as_ref() }.gc_offsets.iter() {
        let value_ptr: *const u8 = offset.raw_ptr(cb);
        // Creating an unaligned pointer is well defined unlike in C.
        let value_ptr = value_ptr as *const VALUE;

        let object = unsafe { value_ptr.read_unaligned() };
        let new_addr = unsafe { rb_gc_location(object) };

        // Only write when the VALUE moves, to be copy-on-write friendly.
        if new_addr != object {
            let value_ptr = value_ptr as *mut VALUE;
            unsafe { value_ptr.write_unaligned(new_addr) };
        }
    }
}

/// Append a set of gc_offsets to the iseq's payload
pub fn append_gc_offsets(iseq: IseqPtr, mut version: IseqVersionRef, offsets: &Vec<CodePtr>) {
    unsafe { version.as_mut() }.gc_offsets.extend(offsets);

    // Call writebarrier on each newly added value
    let cb = ZJITState::get_code_block();
    for &offset in offsets.iter() {
        let value_ptr: *const u8 = offset.raw_ptr(cb);
        let value_ptr = value_ptr as *const VALUE;
        unsafe {
            let object = value_ptr.read_unaligned();
            VALUE::from(iseq).write_barrier(object);
        }
    }
}

/// Remove GC offsets that overlap with a given removed_range.
/// We do this when invalidation rewrites some code with a jump instruction
/// and GC offsets are corrupted by the rewrite, assuming no on-stack code
/// will step into the instruction with the GC offsets after invalidation.
pub fn remove_gc_offsets(mut version: IseqVersionRef, removed_range: &Range<CodePtr>) {
    unsafe { version.as_mut() }.gc_offsets.retain(|&gc_offset| {
        let offset_range = gc_offset..(gc_offset.add_bytes(SIZEOF_VALUE));
        !ranges_overlap(&offset_range, removed_range)
    });
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
    // Mark iseq pointers in JITFrames not owned by an IseqVersion. JITFrames that are currently
    // on the stack are also marked via rb_execution_context_mark_ctx, but JITFrames not on
    // the stack still need their iseqs kept alive because JIT code will reuse them.
    if !ZJITState::has_instance() {
        return;
    }
    // There are far fewer distinct ISEQs than JITFrames, so mark the ISEQs without
    // loading every JITFrame.
    ZJITState::get_jit_frame_iseqs().mark();
}

#[cfg(test)]
mod tests {
    use crate::cruby::test_utils::{eval, with_rubyvm};
    use crate::options::set_call_threshold;

    /// Regression test that freeing an ISEQ must not leave an `IseqPayload` behind.
    #[test]
    fn test_freed_iseqs_do_not_retain_payloads() {
        with_rubyvm(|| {
            // A high call threshold to keep any ISEQs out of the JIT
            let old_call_threshold = unsafe { crate::options::rb_zjit_call_threshold };
            set_call_threshold(1_000_000);

            eval(r#"
                def zjit_eval_iseqs(n)
                  n.times do |i|
                    eval("def __zjit_churn#{i}(x) = x + 1", TOPLEVEL_BINDING, "zjit_churn#{i}.rb")
                    send(:"__zjit_churn#{i}", i)
                    Object.send(:remove_method, :"__zjit_churn#{i}")
                  end
                  3.times { GC.start }
                end
            "#);

            // Each iteration creates and frees several ISEQs. It should not grow the heap usage.
            let alloc_bytes_after_eval = |n: usize| {
                eval(&format!("zjit_eval_iseqs({n})"));
                crate::stats::zjit_alloc_bytes()
            };
            let small = alloc_bytes_after_eval(200);
            let large = alloc_bytes_after_eval(2000);

            set_call_threshold(old_call_threshold);

            let alloc_growth = large.saturating_sub(small);
            assert!(alloc_growth < 100_000, "zjit_alloc_bytes grew by {alloc_growth} bytes across freed ISEQs");
        });
    }

    /// Regression test that JITFrames of compiled ISEQs must not keep the ISEQs alive after
    /// nothing else references them.
    #[test]
    fn test_jit_frames_do_not_retain_freed_iseqs() {
        with_rubyvm(|| {
            let old_call_threshold = unsafe { crate::options::rb_zjit_call_threshold };
            set_call_threshold(2);

            eval(r#"
                def zjit_compiled_churn(n)
                  n.times do |i|
                    klass = Class.new
                    klass.class_eval("def churn(x) = x.to_s.size + x.abs", "zjit_compiled_churn#{i}.rb")
                    object = klass.new
                    2.times { object.churn(i) }
                  end
                  3.times { GC.start }
                  ObjectSpace.count_objects[:T_IMEMO]
                end
            "#);

            // Each iteration compiles an ISEQ whose class becomes garbage. It should not grow the
            // number of live ISEQs.
            let imemos_after_eval = |n: usize| eval(&format!("zjit_compiled_churn({n})")).as_fixnum();
            let small = imemos_after_eval(100);
            let large = imemos_after_eval(1000);

            set_call_threshold(old_call_threshold);

            let imemo_growth = large - small;
            assert!(imemo_growth < 100, "T_IMEMO objects grew by {imemo_growth} across freed compiled ISEQs");
        });
    }
}
