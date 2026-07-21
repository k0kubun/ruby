use crate::core::IseqIdx;
use crate::cruby::*;

/// A high-level instruction translated one-to-one from a YARV instruction.
///
/// This first HIR representation intentionally keeps YARV's opcode and operand
/// layout. In particular, operands stay in the encoded ISEQ and are accessed
/// through [`Insn::get_arg`]. This makes the HIR layer allocation-free and lets
/// the existing code generators consume it without changing their behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Insn {
    insn_idx: IseqIdx,
    opcode: usize,
    pc: *mut VALUE,
}

impl Insn {
    /// Translate the YARV instruction at `insn_idx` into HIR.
    pub fn from_iseq(iseq: IseqPtr, insn_idx: IseqIdx) -> Self {
        let pc = unsafe { rb_iseq_pc_at_idx(iseq, insn_idx.into()) };
        let opcode = unsafe { rb_iseq_opcode_at_pc(iseq, pc) }
            .try_into()
            .unwrap();

        Self::from_raw_parts(insn_idx, opcode, pc)
    }

    /// Build an instruction from an already-decoded YARV instruction.
    pub(crate) fn from_raw_parts(insn_idx: IseqIdx, opcode: usize, pc: *mut VALUE) -> Self {
        Self {
            insn_idx,
            opcode,
            pc,
        }
    }

    pub fn insn_idx(self) -> IseqIdx {
        self.insn_idx
    }

    pub fn opcode(self) -> usize {
        self.opcode
    }

    pub fn pc(self) -> *mut VALUE {
        self.pc
    }

    pub fn len(self) -> u32 {
        insn_len(self.opcode)
    }

    pub fn next_insn_idx(self) -> IseqIdx {
        self.insn_idx + self.len() as IseqIdx
    }

    pub fn get_arg(self, arg_idx: isize) -> VALUE {
        // insn_len requires non-test config.
        #[cfg(not(test))]
        assert!(self.len() > (arg_idx + 1).try_into().unwrap());

        unsafe { *self.pc.offset(arg_idx + 1) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insn_preserves_yarv_instruction() {
        let mut encoded = [VALUE(7), VALUE(8), VALUE(9)];
        let pc = encoded.as_mut_ptr();
        let insn = Insn::from_raw_parts(4, 7, pc);

        assert_eq!(4, insn.insn_idx());
        assert_eq!(7, insn.opcode());
        assert_eq!(pc, insn.pc());
        assert_eq!(VALUE(8), insn.get_arg(0));
        assert_eq!(VALUE(9), insn.get_arg(1));
    }
}
