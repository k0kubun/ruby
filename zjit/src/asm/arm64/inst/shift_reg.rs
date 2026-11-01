/// The operation to perform for this instruction.
enum Opcode {
    LSLV = 0b001000,
    LSRV = 0b001001,
    ASRV = 0b001010,
}

/// The struct that represents an A64 shift by a register amount (data-processing 2-source)
/// instruction. The shift amount is taken modulo the register size.
///
/// +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
/// | 31 30 29 28 | 27 26 25 24 | 23 22 21 20 | 19 18 17 16 | 15 14 13 12 | 11 10 09 08 | 07 06 05 04 | 03 02 01 00 |
/// |     0  0  1    1  0  1  0    1  1  0                                                                          |
/// | sf                                       rm..............  opcode...........  rn.............. rd.............. |
/// +-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
///
pub struct ShiftReg {
    /// The register number of the destination register.
    rd: u8,

    /// The register number of the value to be shifted.
    rn: u8,

    /// The operation to perform.
    opcode: Opcode,

    /// The register number of the shift amount.
    rm: u8,

    /// Whether or not this instruction is operating on 64-bit operands.
    sf: bool,
}

impl ShiftReg {
    /// LSLV
    /// <https://developer.arm.com/documentation/ddi0602/2023-06/Base-Instructions/LSLV--Logical-Shift-Left-Variable->
    pub fn lslv(rd: u8, rn: u8, rm: u8, num_bits: u8) -> Self {
        Self { rd, rn, opcode: Opcode::LSLV, rm, sf: num_bits == 64 }
    }

    /// LSRV
    /// <https://developer.arm.com/documentation/ddi0602/2023-06/Base-Instructions/LSRV--Logical-Shift-Right-Variable->
    pub fn lsrv(rd: u8, rn: u8, rm: u8, num_bits: u8) -> Self {
        Self { rd, rn, opcode: Opcode::LSRV, rm, sf: num_bits == 64 }
    }

    /// ASRV
    /// <https://developer.arm.com/documentation/ddi0602/2023-06/Base-Instructions/ASRV--Arithmetic-Shift-Right-Variable->
    pub fn asrv(rd: u8, rn: u8, rm: u8, num_bits: u8) -> Self {
        Self { rd, rn, opcode: Opcode::ASRV, rm, sf: num_bits == 64 }
    }
}

impl From<ShiftReg> for u32 {
    /// Convert an instruction into a 32-bit value.
    fn from(inst: ShiftReg) -> Self {
        0
        | ((inst.sf as u32) << 31)
        | (0b0011010110 << 21)
        | ((inst.rm as u32) << 16)
        | ((inst.opcode as u32) << 10)
        | ((inst.rn as u32) << 5)
        | (inst.rd as u32)
    }
}

impl From<ShiftReg> for [u8; 4] {
    /// Convert an instruction into a 4 byte array.
    fn from(inst: ShiftReg) -> [u8; 4] {
        let result: u32 = inst.into();
        result.to_le_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lslv() {
        let result: u32 = ShiftReg::lslv(0, 1, 2, 64).into();
        assert_eq!(0x9ac22020, result);
    }

    #[test]
    fn test_lsrv() {
        let result: u32 = ShiftReg::lsrv(0, 1, 2, 64).into();
        assert_eq!(0x9ac22420, result);
    }

    #[test]
    fn test_asrv() {
        let result: u32 = ShiftReg::asrv(0, 1, 2, 64).into();
        assert_eq!(0x9ac22820, result);
    }

    #[test]
    fn test_asrv_32() {
        let result: u32 = ShiftReg::asrv(0, 1, 2, 32).into();
        assert_eq!(0x1ac22820, result);
    }
}
