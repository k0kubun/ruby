//! Compile-time-known interpreter state for side exits, kept off the code region.
//!
//! Every side exit has to hand the interpreter a `cfp->pc`, a `cfp->iseq` and a
//! `cfp->sp`, and a guard exit additionally has to call [`crate::codegen::exit_recompile`]
//! with the ISEQ/instruction index it is re-profiling. All of that is known when the
//! exit is compiled, and materializing it with immediates costs roughly 45 bytes of
//! executable memory per exit -- about half of a typical exit stub, and on a large
//! application half of that again of the whole 64MiB exec region.
//!
//! So the constants live here instead, one [`ExitMeta`] per distinct exit, and the
//! stub only loads its *index* into the scratch register before jumping to
//! `exit_meta_trampoline`. The trampoline calls
//! [`crate::codegen::rb_zjit_side_exit_from_meta`], which reads the record and does
//! the same work the inline code used to do. The table is indexed rather than
//! pointed into so that pushing to it may reallocate freely.

use crate::cruby::{IseqPtr, VALUE, rb_gc_location};
use crate::state::ZJITState;

/// Interpreter state a side exit restores, plus its optional recompile trigger.
///
/// `#[repr(C)]` is not required by the runtime -- only Rust reads these fields --
/// but it keeps the layout predictable for anyone matching this against a disasm
/// dump of the trampoline.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExitMeta {
    /// `cfp->pc` the interpreter resumes at.
    pub pc: *const VALUE,
    /// `cfp->iseq` of the exiting frame. Also the ISEQ that owns the profile entry
    /// `insn_idx` refers to, which is why `exit_recompile` needs no separate
    /// `frame_iseq`: the two are always the same ISEQ.
    pub iseq: IseqPtr,
    /// The compiled unit to invalidate to force a recompile. For an exit out of
    /// inlined code this is the outer ISEQ the callee was folded into, so it is not
    /// necessarily `iseq`. Null when this exit does not profile-and-recompile.
    pub compiled_iseq: IseqPtr,
    /// `cfp->sp` is the exiting frame's SP register plus this many `VALUE`s, i.e.
    /// the height of the Ruby stack the exit wrote out.
    pub sp_offset: u32,
    /// Instruction index within `iseq` whose re-profiling gates the recompile.
    /// Meaningless when `compiled_iseq` is null.
    pub insn_idx: u32,
}

impl ExitMeta {
    /// Update the ISEQ pointers after GC compaction, like `JITFrame` does. `pc`
    /// points into the ISEQ's malloc'd `iseq_encoded`, which compaction does not
    /// move, so it needs no fixup.
    pub fn update_references(&mut self) {
        if !self.iseq.is_null() {
            self.iseq = unsafe { rb_gc_location(VALUE::from(self.iseq)) }.as_iseq();
        }
        if !self.compiled_iseq.is_null() {
            self.compiled_iseq = unsafe { rb_gc_location(VALUE::from(self.compiled_iseq)) }.as_iseq();
        }
    }
}

/// Records to make room for each time the table fills up. The table only ever
/// grows, so `Vec`'s doubling would leave up to as many bytes unused as used --
/// three quarters of a megabyte on an application with a few hundred thousand
/// exits. Growing in fixed steps caps the slack at 128KiB instead, and the extra
/// copying only happens while compiling.
const GROWTH_STEP: usize = 4096;

/// Add `meta` to the process-wide table and return the index side-exit code loads.
/// Records are never freed, matching how `JITFrame`s and the code that references
/// them are retained for the lifetime of the process.
pub fn intern(meta: ExitMeta) -> u32 {
    // The record's ISEQs have to stay alive for as long as the record does, which is
    // forever; the mark phase reaches them through this set rather than by walking
    // the (much longer) table. See [`crate::gc::RootIseqs`].
    crate::gc::register_root_iseq(meta.iseq);
    crate::gc::register_root_iseq(meta.compiled_iseq);
    let metas = ZJITState::get_exit_metas();
    if metas.len() == metas.capacity() {
        metas.reserve_exact(GROWTH_STEP);
    }
    let idx = metas.len();
    metas.push(meta);
    u32::try_from(idx).expect("side-exit metadata index should fit in 32 bits")
}

/// Look up an interned record by the index baked into an exit stub.
pub fn get(idx: u32) -> ExitMeta {
    ZJITState::get_exit_metas()[idx as usize]
}
