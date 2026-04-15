#![allow(dead_code)]
#![allow(static_mut_refs)]

#![allow(clippy::enum_variant_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_bool)]

// Add std docs to cargo doc.
#[doc(inline)]
pub use std;

mod state;
mod distribution;
mod cruby;
mod cruby_methods;
mod hir;
mod hir_type;
mod hir_effect;
#[cfg(not(target_arch = "wasm32"))]
mod codegen;
#[cfg(target_arch = "wasm32")]
mod codegen_stubs {
    use crate::cruby::*;

    /// On wasm32, the JIT entry point is a no-op that returns null.
    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_iseq_gen_entry_point(
        _iseq: IseqPtr, _ec: EcPtr, _jit_exception: bool
    ) -> *const u8 {
        std::ptr::null()
    }
}
mod stats;
mod cast;
#[cfg(not(target_arch = "wasm32"))]
mod virtualmem;
#[cfg(not(target_arch = "wasm32"))]
mod asm;
#[cfg(not(target_arch = "wasm32"))]
mod backend;
#[cfg(all(feature = "disasm", not(target_arch = "wasm32")))]
mod disasm;
mod options;
mod profile;
#[cfg(not(target_arch = "wasm32"))]
mod invariants;
#[cfg(target_arch = "wasm32")]
mod invariants {
    //! Stub invariants module for wasm32 (no code invalidation needed).
    use crate::cruby::*;

    pub fn iseq_escapes_ep(_iseq: IseqPtr) -> bool {
        false
    }

    pub fn non_root_box_created() -> bool {
        false
    }

    pub fn has_singleton_class_of(_klass: VALUE) -> bool {
        false
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_cme_invalidate(_cme: *const rb_callable_method_entry_t) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_constant_state_changed(_id: ID) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_before_ractor_spawn() {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_invalidate_root_box() {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_invalidate_no_singleton_class(_klass: VALUE) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_bop_redefined(_klass: VALUE, _bop: std::ffi::c_int) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_invalidate_no_ep_escape(_iseq: IseqPtr) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_tracing_invalidate_all() {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_cme_free(_cme: *const rb_callable_method_entry_t) {}
}
mod bitset;
#[cfg(not(target_arch = "wasm32"))]
mod gc;
#[cfg(target_arch = "wasm32")]
mod gc {
    //! Stub GC module for wasm32.
    use crate::cruby::*;
    use crate::payload::IseqPayload;

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_iseq_mark(_payload: *mut std::ffi::c_void) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_iseq_update_references(_payload: *mut std::ffi::c_void) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_iseq_free(_payload: *mut std::ffi::c_void) {
        if !_payload.is_null() {
            unsafe { drop(Box::from_raw(_payload as *mut IseqPayload)); }
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_root_mark() {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_klass_free(_klass: VALUE) {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_root_update_references() {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_mark_all_writable() {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_mark_all_executable() {}

    #[unsafe(no_mangle)]
    pub extern "C" fn rb_zjit_jit_frame_update_references(_payload: *mut std::ffi::c_void) {}
}
#[cfg(not(target_arch = "wasm32"))]
mod jit_frame;
mod payload;
mod json;
mod ttycolors;
mod iongraph_api;

/// Pull in YJIT's symbols for linking the test binary in `make zjit-test`. The test binary builds
/// ZJIT symbols and they should take precendence over the ones built for miniruby, so libminiruby
/// doesn't include any ZJIT code. But, in removing from libminiruby the object which contains all
/// rust code, including ZJIT code, we also remove all YJIT symbols which the rest of libminiruby
/// might request in YJIT+ZJIT configurations. We add back the YJIT symbols here.
///
/// Only relevant for YJIT+ZJIT configurations, but building YJIT is fast, so always do it for the
/// test binary for simplicity.
#[cfg(test)]
use yjit as _;
