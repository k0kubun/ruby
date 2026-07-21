//! A minimal, "as-is" High-level Intermediate Representation (HIR) for YJIT.
//!
//! This is the first step of an experiment to make YJIT mimic ZJIT's design.
//! Unlike ZJIT's SSA-based HIR, this representation is intentionally trivial: it
//! lifts each YARV instruction of a basic block verbatim (opcode + program
//! counter), preserving a 1:1 correspondence with the bytecode. There is no
//! stack abstraction, no SSA, and no optimization.
//!
//! The point is to introduce a *layer* between YARV decoding and code
//! generation. Instead of decoding YARV opcodes inline inside the codegen loop,
//! `gen_single_block` now first builds a [`BlockHir`] and then lowers each
//! [`HirInsn`] using the existing per-instruction codegen functions in
//! `codegen.rs`. Because each `HirInsn` keeps the real `pc`, those `gen_*`
//! functions (which read their operands from `jit.pc` via `jit.get_arg`) keep
//! working unchanged.
//!
//! Later steps can grow this HIR toward ZJIT's design (SSA values, explicit
//! stack modeling, optimization passes) without having to touch the codegen
//! entry point again.

// YARVINSN_* constants are not upper case; matching them in patterns trips this
// lint, same as in codegen.rs.
#![allow(non_upper_case_globals)]

use crate::core::IseqIdx;
use crate::cruby::*;
use std::fmt;

/// A single YARV instruction lifted into HIR, verbatim.
///
/// This carries just enough to (a) dispatch to the existing codegen function
/// and (b) let those functions read their operands from the real bytecode.
#[derive(Clone, Copy)]
pub struct HirInsn {
    /// Bytecode index of this instruction within the ISEQ.
    pub insn_idx: IseqIdx,

    /// YARV opcode (`ruby_vminsn_type`, stored as `usize` to match codegen).
    pub opcode: usize,

    /// Program counter for this instruction. Points into the ISEQ's encoded
    /// instruction array; operands live at `pc.offset(1..insn_len)`.
    pub pc: *mut VALUE,
}

impl HirInsn {
    /// Number of operands (encoded slots after the opcode) for this instruction.
    pub fn num_operands(&self) -> usize {
        (insn_len(self.opcode) as usize).saturating_sub(1)
    }

    /// Read the `idx`-th operand straight from the bytecode.
    pub fn operand(&self, idx: usize) -> VALUE {
        unsafe { *(self.pc.offset(idx as isize + 1)) }
    }
}

impl fmt::Display for HirInsn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04} {}", self.insn_idx, insn_name(self.opcode))?;
        for i in 0..self.num_operands() {
            write!(f, " {:#x}", self.operand(i).as_usize())?;
        }
        Ok(())
    }
}

/// HIR for a single YJIT basic block: a linear sequence of YARV instructions
/// lifted as-is. The sequence starts at the block's entry index and extends up
/// to and including the first block-terminating instruction (or the end of the
/// ISEQ). Note that the block may actually end earlier during lowering (for
/// example when an instruction defers compilation); trailing instructions are
/// simply never lowered, which is harmless.
pub struct BlockHir {
    pub insns: Vec<HirInsn>,
}

impl BlockHir {
    /// Lift the YARV instructions of one block into HIR.
    ///
    /// Scanning starts at `start_idx` and stops after the first
    /// block-terminating opcode (see [`opcode_ends_block`]) or when reaching
    /// `iseq_size`.
    pub fn from_iseq(iseq: IseqPtr, start_idx: IseqIdx, iseq_size: IseqIdx) -> BlockHir {
        let mut insns = Vec::new();
        let mut insn_idx = start_idx;

        while insn_idx < iseq_size {
            let pc = unsafe { rb_iseq_pc_at_idx(iseq, insn_idx.into()) };
            let opcode: usize = unsafe { rb_iseq_opcode_at_pc(iseq, pc) }
                .try_into()
                .unwrap();

            insns.push(HirInsn { insn_idx, opcode, pc });

            // Stop once we've included a terminator. This keeps the block's HIR
            // tight in the common case; over-approximating (scanning a few
            // extra instructions) would only be wasteful, never incorrect,
            // since lowering stops at the real terminator anyway.
            if opcode_ends_block(opcode) {
                break;
            }

            insn_idx += insn_len(opcode) as IseqIdx;
        }

        BlockHir { insns }
    }
}

impl fmt::Display for BlockHir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for insn in &self.insns {
            writeln!(f, "  {insn}")?;
        }
        Ok(())
    }
}

/// Returns `true` for opcodes that unconditionally terminate (or force a split
/// of) a YJIT basic block.
///
/// SAFETY OF THE APPROXIMATION: this must only return `true` for opcodes that
/// can never fall through into the same block. Returning `true` for a
/// fall-through opcode would truncate the block's HIR and drop instructions
/// (a bug). Returning `false` for a genuine terminator is safe — it only makes
/// `from_iseq` scan a bit further than necessary.
///
/// Note that many instructions (sends, and profiled ops that defer) also end a
/// block *dynamically* at codegen time; those are intentionally not listed here
/// because they can also continue the current block.
fn opcode_ends_block(opcode: usize) -> bool {
    matches!(
        opcode as u32,
        YARVINSN_leave
        | YARVINSN_jump
        | YARVINSN_branchif
        | YARVINSN_branchunless
        | YARVINSN_branchnil
        | YARVINSN_throw
        // opt_getconstant_path must live in a block by itself so it can be
        // invalidated by instruction index; gen_single_block splits before it.
        | YARVINSN_opt_getconstant_path
    )
}
