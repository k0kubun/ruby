//! Side-exit descriptors: the interpreter state of a side exit, kept off the code region.
//!
//! A side exit has to hand the interpreter a complete frame: `cfp->pc`, `cfp->sp` and
//! `cfp->iseq`, every Ruby stack slot and local wherever the register allocator put it,
//! the JITFrames of older inlined frames, and optionally a recompile or a traced exit
//! stack. Emitting all of that as machine code costs ~100 bytes per exit, interleaved
//! with the hot code it guards.
//!
//! Like V8's deopt exits, HotSpot's uncommon traps and Cinder's side-exit helper, ZJIT
//! instead emits each exit as a single `call`/`bl` into `exit_descriptor_trampoline`.
//! The trampoline saves every general-purpose register into a save area on the machine
//! stack and calls [`rb_zjit_side_exit_descriptor`] with that area and the call's return
//! address. The return address identifies the descriptor registered for that call site,
//! which says where each value lives ([`ExitLoc`]); the handler writes the frame out
//! exactly like the inline exit code used to, and the trampoline then jumps to
//! `materialize_exit_trampoline`.
//!
//! Descriptors are built as [`ExitDescriptor`]s at compile time and stored in a
//! compact byte encoding, like V8's translation arrays and HotSpot's compressed
//! debug info: a register is one byte, a spill slot or a small immediate is two or
//! three, and pointers the GC may move are stored as 8 raw bytes so that compaction
//! can update them in place.
//!
//! Interpreting a descriptor makes a taken exit slower than the inline code was. Like
//! JavaScriptCore's lazily compiled OSR exits, an exit that keeps being taken is
//! compiled from its descriptor into the same code the inline exit used to be, and its
//! call is patched into a jump to that code. See [`lazy_exit_hit`].

use std::ffi::c_char;
use std::ops::Range;

use crate::backend::lir::{CFP, NATIVE_BASE_PTR, SP};
use crate::codegen::exit_recompile;
use crate::cruby::*;
use crate::state::{rb_zjit_record_exit_stack, ZJITState};
use crate::stats::{incr_counter, incr_counter_by, Counter};
use crate::virtualmem::CodePtr;
use crate::asm::CodeBlock;
use crate::backend::lir::{Assembler, Opnd, Target, asm_comment, C_RET_OPND};
use crate::options::get_option;
use crate::payload::IseqVersionRef;

/// Where a side exit finds one 64-bit value when it runs.
///
/// Register numbers are machine register numbers (`reg_no`), which index the
/// trampoline's register save area. Displacements are in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExitLoc {
    /// A general-purpose register. `bits` < 64 zero-extends.
    Reg { reg: u8, bits: u8 },
    /// A native stack slot at `[NATIVE_BASE_PTR + disp]`.
    NativeSlot { disp: i32, bits: u8 },
    /// Memory at `[reg + disp]`.
    Mem { base: u8, bits: u8, disp: i32 },
    /// Memory at `[[NATIVE_BASE_PTR + slot] + disp]`: a spilled memory base.
    StackIndirect { slot: i32, disp: i32, bits: u8 },
    /// An immediate that is the sign extension of 32 bits. Never a heap object.
    SmallImm(i32),
    /// A raw 64-bit immediate, stored at this index of [`ExitDescriptor::imms`].
    Imm(u32),
    /// A Ruby VALUE, stored at this index of [`ExitDescriptor::imms`]. Marked and
    /// moved by the GC when it is a heap object.
    Value(u32),
}

/// Values the handler stores into native stack slots for a stack map, and the
/// JITFrame it installs, so that `rb_zjit_materialize_frames` can rebuild older
/// inlined frames. The JITFrame's own entries are written at compile time.
#[derive(Clone, Debug)]
pub struct ExitStackMap {
    /// (source, byte offset from NATIVE_BASE_PTR of the capture slot)
    pub captures: Box<[(ExitLoc, i32)]>,
    /// JITFrame to install
    pub jit_frame: *const zjit_jit_frame,
    /// Byte offset from NATIVE_BASE_PTR of the slot the JITFrame pointer goes to
    pub jit_frame_slot: i32,
}

/// Rarely used parts of a descriptor, boxed to keep [`ExitDescriptor`] small.
#[derive(Clone, Debug, Default)]
pub struct ExitDescriptorExtra {
    pub stack_map: Option<ExitStackMap>,
    /// Reason to pass to rb_zjit_record_exit_stack() for --zjit-trace-side-exits
    pub trace_reason: Option<*const c_char>,
}

/// Everything a side exit does, described for [`rb_zjit_side_exit_descriptor`].
/// This is the compile-time form; the table stores it encoded by [`Self::encode`].
#[derive(Clone, Debug)]
pub struct ExitDescriptor {
    /// `cfp->pc` the interpreter resumes at. Points into the ISEQ's malloc'd
    /// `iseq_encoded`, which does not move.
    pub pc: *const VALUE,
    /// `cfp->iseq` of the exiting frame
    pub iseq: IseqPtr,
    /// If not null, the compiled ISEQ to pass to exit_recompile(). For an exit out
    /// of inlined code, this is the outer ISEQ, not `iseq`.
    pub recompile: IseqPtr,
    /// Ruby stack slots (the first `num_stack`) followed by locals
    locs: Vec<ExitLoc>,
    num_stack: usize,
    /// 64-bit immediates referenced by [`ExitLoc::Imm`] and [`ExitLoc::Value`]
    imms: Vec<u64>,
    pub extra: Option<Box<ExitDescriptorExtra>>,
}

/// Collects the immediates of a descriptor while it is being built.
#[derive(Default)]
pub struct ExitImms(Vec<u64>);

impl ExitImms {
    /// Location of a 64-bit immediate. `is_value` is true when it's a Ruby VALUE.
    pub fn imm(&mut self, imm: u64, is_value: bool) -> ExitLoc {
        let small = imm as i64 as i32;
        if small as i64 as u64 == imm && (!is_value || VALUE(imm as usize).special_const_p()) {
            return ExitLoc::SmallImm(small);
        }
        let idx: u32 = self.0.len().try_into().unwrap();
        self.0.push(imm);
        if is_value { ExitLoc::Value(idx) } else { ExitLoc::Imm(idx) }
    }
}

// Descriptor encoding. All multi-byte integers are little-endian; "varint" is
// LEB128 and signed values are zigzag-encoded varints.
//
//   u64 pc, u64 iseq, u8 flags, u8 hits, [u64 recompile], varint num_stack, varint num_locals,
//   loc * (num_stack + num_locals),
//   [u64 jit_frame, svarint jit_frame_slot, varint num_captures, (loc, svarint slot) * num_captures],
//   [u64 trace_reason]
//
// Each loc starts with a byte whose low 3 bits are a tag and high 5 bits a payload.
// `hits` is updated in place by the handler, see [`lazy_exit_hit`].
const FLAG_RECOMPILE: u8 = 1 << 0; // a u64 recompile ISEQ follows
const FLAG_RECOMPILE_SELF: u8 = 1 << 1; // recompile the exiting ISEQ itself
const FLAG_STACK_MAP: u8 = 1 << 2;
const FLAG_TRACE: u8 = 1 << 3;

const TAG_REG: u8 = 0; // payload: reg_no. A 64-bit register.
const TAG_SLOT: u8 = 1; // svarint disp. A 64-bit native stack slot.
const TAG_SMALL_IMM: u8 = 2; // svarint imm
const TAG_VALUE: u8 = 3; // u64 VALUE, updated in place by compaction
const TAG_IMM: u8 = 4; // u64 imm
const TAG_MEM: u8 = 5; // payload: base reg_no, then u8 bits, svarint disp
const TAG_STACK_INDIRECT: u8 = 6; // u8 bits, svarint slot, svarint disp
const TAG_OTHER: u8 = 7; // payload 0: u8 reg_no, u8 bits (narrow register); payload 1: u8 bits, svarint disp (narrow slot)
const TAG_BITS: u32 = 3;
const TAG_MASK: u8 = (1 << TAG_BITS) - 1;

struct Encoder(Vec<u8>);

impl Encoder {
    fn u8(&mut self, byte: u8) {
        self.0.push(byte);
    }

    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn varint(&mut self, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                self.u8(byte);
                return;
            }
            self.u8(byte | 0x80);
        }
    }

    fn svarint(&mut self, value: i64) {
        self.varint(((value << 1) ^ (value >> 63)) as u64);
    }

    fn tag(&mut self, tag: u8, payload: u8) {
        assert!(payload < (1 << (8 - TAG_BITS)));
        self.u8(tag | (payload << TAG_BITS));
    }

    fn loc(&mut self, loc: ExitLoc, imms: &[u64]) {
        match loc {
            ExitLoc::Reg { reg, bits: 64 } => self.tag(TAG_REG, reg),
            ExitLoc::Reg { reg, bits } => { self.tag(TAG_OTHER, 0); self.u8(reg); self.u8(bits); }
            ExitLoc::NativeSlot { disp, bits: 64 } => { self.tag(TAG_SLOT, 0); self.svarint(disp as i64); }
            ExitLoc::NativeSlot { disp, bits } => { self.tag(TAG_OTHER, 1); self.u8(bits); self.svarint(disp as i64); }
            ExitLoc::Mem { base, bits, disp } => { self.tag(TAG_MEM, base); self.u8(bits); self.svarint(disp as i64); }
            ExitLoc::StackIndirect { slot, disp, bits } => {
                self.tag(TAG_STACK_INDIRECT, 0); self.u8(bits); self.svarint(slot as i64); self.svarint(disp as i64);
            }
            ExitLoc::SmallImm(imm) => { self.tag(TAG_SMALL_IMM, 0); self.svarint(imm as i64); }
            ExitLoc::Imm(idx) => { self.tag(TAG_IMM, 0); self.u64(imms[idx as usize]); }
            ExitLoc::Value(idx) => { self.tag(TAG_VALUE, 0); self.u64(imms[idx as usize]); }
        }
    }
}

/// Cursor over encoded descriptors. Descriptors are written by [`Encoder`] only,
/// so reading them does no bounds checks: this runs on every taken side exit.
struct Decoder {
    ptr: *mut u8,
}

impl Decoder {
    #[inline(always)]
    fn u8(&mut self) -> u8 {
        unsafe {
            let byte = self.ptr.read();
            self.ptr = self.ptr.add(1);
            byte
        }
    }

    /// Pointer to the next 8-byte field, for updating it in place
    #[inline(always)]
    fn u64_ptr(&mut self) -> *mut u64 {
        let ptr = self.ptr as *mut u64;
        self.ptr = unsafe { self.ptr.add(8) };
        ptr
    }

    #[inline(always)]
    fn u64(&mut self) -> u64 {
        unsafe { self.u64_ptr().read_unaligned() }
    }

    #[inline(always)]
    fn varint(&mut self) -> u64 {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let byte = self.u8();
            value |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return value;
            }
            shift += 7;
        }
    }

    #[inline(always)]
    fn svarint(&mut self) -> i64 {
        let value = self.varint();
        ((value >> 1) as i64) ^ -((value & 1) as i64)
    }

    /// Decode a loc. Immediates are returned with their 8-byte field so that the
    /// GC can update VALUEs in place.
    #[inline(always)]
    fn loc(&mut self) -> DecodedLoc {
        let byte = self.u8();
        let payload = byte >> TAG_BITS;
        match byte & TAG_MASK {
            TAG_REG => DecodedLoc::Loc(ExitLoc::Reg { reg: payload, bits: 64 }),
            TAG_SLOT => DecodedLoc::Loc(ExitLoc::NativeSlot { disp: self.svarint() as i32, bits: 64 }),
            TAG_SMALL_IMM => DecodedLoc::Loc(ExitLoc::SmallImm(self.svarint() as i32)),
            TAG_VALUE => DecodedLoc::Value(self.u64_ptr()),
            TAG_IMM => DecodedLoc::Imm(self.u64()),
            TAG_MEM => {
                let bits = self.u8();
                DecodedLoc::Loc(ExitLoc::Mem { base: payload, bits, disp: self.svarint() as i32 })
            }
            TAG_STACK_INDIRECT => {
                let bits = self.u8();
                let slot = self.svarint() as i32;
                DecodedLoc::Loc(ExitLoc::StackIndirect { slot, disp: self.svarint() as i32, bits })
            }
            _ => match payload {
                0 => {
                    let reg = self.u8();
                    DecodedLoc::Loc(ExitLoc::Reg { reg, bits: self.u8() })
                }
                _ => {
                    let bits = self.u8();
                    DecodedLoc::Loc(ExitLoc::NativeSlot { disp: self.svarint() as i32, bits })
                }
            },
        }
    }
}

enum DecodedLoc {
    Loc(ExitLoc),
    Imm(u64),
    Value(*mut u64),
}

/// Header of an encoded descriptor. `iseq` and `recompile` point at their 8-byte
/// fields so that the GC can update them in place.
struct DecodedHeader {
    pc: *const VALUE,
    iseq: *mut u64,
    /// Null if the exit doesn't recompile
    recompile: *mut u64,
    flags: u8,
    /// Hit counter and lazy compilation state, updated in place. See [`lazy_exit_hit`].
    hits: *mut u8,
    num_stack: usize,
    num_locals: usize,
}

impl Decoder {
    #[inline(always)]
    fn header(&mut self) -> DecodedHeader {
        let pc = self.u64() as *const VALUE;
        let iseq = self.u64_ptr();
        let flags = self.u8();
        let hits = self.ptr;
        self.u8();
        let recompile = if flags & FLAG_RECOMPILE != 0 {
            self.u64_ptr()
        } else if flags & FLAG_RECOMPILE_SELF != 0 {
            iseq
        } else {
            std::ptr::null_mut()
        };
        let num_stack = self.varint() as usize;
        let num_locals = self.varint() as usize;
        DecodedHeader { pc, iseq, recompile, flags, hits, num_stack, num_locals }
    }

    /// Skip or visit the rest of a descriptor after its header, calling `on_value`
    /// with each VALUE field. Leaves the cursor at the next descriptor.
    fn visit_rest(&mut self, header: &DecodedHeader, mut on_value: impl FnMut(*mut u64)) {
        let mut visit = |decoder: &mut Decoder| {
            if let DecodedLoc::Value(field) = decoder.loc() {
                on_value(field);
            }
        };
        for _ in 0..header.num_stack + header.num_locals {
            visit(self);
        }
        if header.flags & FLAG_STACK_MAP != 0 {
            self.u64();
            self.svarint();
            for _ in 0..self.varint() {
                visit(self);
                self.svarint();
            }
        }
        if header.flags & FLAG_TRACE != 0 {
            self.u64();
        }
    }
}

impl Decoder {
    /// Decode the rest of a descriptor after its header into its compile-time form
    fn descriptor(&mut self, header: &DecodedHeader) -> ExitDescriptor {
        let mut imms = ExitImms::default();
        let loc = |decoder: &mut Decoder, imms: &mut ExitImms| match decoder.loc() {
            DecodedLoc::Loc(loc) => loc,
            DecodedLoc::Imm(imm) => { imms.0.push(imm); ExitLoc::Imm(imms.0.len() as u32 - 1) }
            DecodedLoc::Value(field) => { imms.0.push(unsafe { field.read_unaligned() }); ExitLoc::Value(imms.0.len() as u32 - 1) }
        };
        let stack = (0..header.num_stack).map(|_| loc(self, &mut imms)).collect();
        let locals = (0..header.num_locals).map(|_| loc(self, &mut imms)).collect();
        let stack_map = (header.flags & FLAG_STACK_MAP != 0).then(|| {
            let jit_frame = self.u64() as *const zjit_jit_frame;
            let jit_frame_slot = self.svarint() as i32;
            let captures = (0..self.varint()).map(|_| (loc(self, &mut imms), self.svarint() as i32)).collect();
            ExitStackMap { captures, jit_frame, jit_frame_slot }
        });
        let trace_reason = (header.flags & FLAG_TRACE != 0).then(|| self.u64() as *const c_char);
        let extra = (stack_map.is_some() || trace_reason.is_some())
            .then(|| Box::new(ExitDescriptorExtra { stack_map, trace_reason }));
        let recompile = if header.recompile.is_null() { std::ptr::null() } else { unsafe { header.recompile.read_unaligned() as IseqPtr } };
        let iseq = unsafe { header.iseq.read_unaligned() as IseqPtr };
        ExitDescriptor::new(header.pc, iseq, stack, locals, imms, recompile, extra)
    }
}

impl ExitDescriptor {
    pub fn new(
        pc: *const VALUE,
        iseq: IseqPtr,
        stack: Vec<ExitLoc>,
        locals: Vec<ExitLoc>,
        imms: ExitImms,
        recompile: IseqPtr,
        extra: Option<Box<ExitDescriptorExtra>>,
    ) -> Self {
        let num_stack = stack.len();
        let mut locs = stack;
        locs.extend(locals);
        ExitDescriptor { pc, iseq, recompile, locs, num_stack, imms: imms.0, extra }
    }

    pub fn stack(&self) -> &[ExitLoc] {
        &self.locs[..self.num_stack]
    }

    pub fn locals(&self) -> &[ExitLoc] {
        &self.locs[self.num_stack..]
    }

    fn stack_map(&self) -> Option<&ExitStackMap> {
        self.extra.as_ref().and_then(|extra| extra.stack_map.as_ref())
    }

    fn trace_reason(&self) -> Option<*const c_char> {
        self.extra.as_ref().and_then(|extra| extra.trace_reason)
    }

    /// Encode this descriptor for the table
    pub fn encode(&self) -> Box<[u8]> {
        let mut enc = Encoder(Vec::with_capacity(32));
        enc.u64(self.pc as u64);
        enc.u64(self.iseq as u64);
        let mut flags = 0;
        if self.recompile == self.iseq {
            flags |= FLAG_RECOMPILE_SELF;
        } else if !self.recompile.is_null() {
            flags |= FLAG_RECOMPILE;
        }
        if self.stack_map().is_some() {
            flags |= FLAG_STACK_MAP;
        }
        if self.trace_reason().is_some() {
            flags |= FLAG_TRACE;
        }
        enc.u8(flags);
        enc.u8(0); // hits
        if flags & FLAG_RECOMPILE != 0 {
            enc.u64(self.recompile as u64);
        }
        enc.varint(self.num_stack as u64);
        enc.varint((self.locs.len() - self.num_stack) as u64);
        for &loc in &self.locs {
            enc.loc(loc, &self.imms);
        }
        if let Some(stack_map) = self.stack_map() {
            enc.u64(stack_map.jit_frame as u64);
            enc.svarint(stack_map.jit_frame_slot as i64);
            enc.varint(stack_map.captures.len() as u64);
            for &(loc, slot) in stack_map.captures.iter() {
                enc.loc(loc, &self.imms);
                enc.svarint(slot as i64);
            }
        }
        if let Some(reason) = self.trace_reason() {
            enc.u64(reason as u64);
        }
        enc.0.into_boxed_slice()
    }
}

/// Process-wide table of exit descriptors. Append-only: descriptors live as long
/// as the code that refers to them, which is never freed.
///
/// It is behind a lock because JIT code of other Ractors may take side exits
/// (reading it) while one Ractor compiles (appending to it). Compilation and GC
/// hold the VM lock, so the write lock is never contended with other writers.
#[derive(Default)]
pub struct ExitDescriptorTable {
    /// Encoded descriptors, back to back
    bytes: Vec<u8>,
    /// Sorted (return address as a CodePtr offset, offset into `bytes`). Code is
    /// written at monotonically increasing addresses, so this is appended to in
    /// order and looked up with a binary search.
    ret_addrs: Vec<(u32, u32)>,
    /// Address of CodePtr offset 0 in ZJIT's code block, to turn return addresses
    /// into offsets without borrowing the code block on every exit
    code_base_addr: usize,
    /// Address of materialize_exit_trampoline, where the handler sends exits it
    /// materialized from their descriptor
    materialize_exit_addr: usize,
    /// Sorted (byte range of descriptors, ISEQ version they were compiled for), to
    /// find the version that owns a lazily compiled exit's GC offsets
    versions: Vec<(Range<u32>, IseqVersionRef)>,
}

/// Reserve room for `additional` more elements in a table that only grows. Doubling
/// would leave up to as much unused as used, so grow by an eighth instead; the extra
/// copying only happens while compiling.
fn reserve_append_only<T>(vec: &mut Vec<T>, additional: usize) {
    if vec.capacity() - vec.len() < additional {
        let step = (vec.len() / 8).max(64 * 1024 / std::mem::size_of::<T>()).max(additional);
        vec.reserve_exact(step);
    }
}

impl ExitDescriptorTable {
    fn register(&mut self, ret_addr: u32, encoded: &[u8]) {
        let offset: u32 = self.bytes.len().try_into().expect("exit descriptor table should fit in 4GiB");
        incr_counter!(exit_descriptor_count);
        incr_counter_by(Counter::exit_descriptor_bytes, (encoded.len() + std::mem::size_of::<(u32, u32)>()) as u64);
        reserve_append_only(&mut self.bytes, encoded.len());
        self.bytes.extend_from_slice(encoded);
        reserve_append_only(&mut self.ret_addrs, 1);
        match self.ret_addrs.last() {
            Some(&(last, _)) if last >= ret_addr => {
                // Not expected for code in the main code block, but keep the table sorted.
                match self.ret_addrs.binary_search_by_key(&ret_addr, |&(addr, _)| addr) {
                    Ok(pos) => self.ret_addrs[pos].1 = offset,
                    Err(pos) => self.ret_addrs.insert(pos, (ret_addr, offset)),
                }
            }
            _ => self.ret_addrs.push((ret_addr, offset)),
        }
    }

    /// Return a decoder at the descriptor for a return address, and its table offset
    #[inline(always)]
    fn lookup(&self, ret_addr: usize) -> Option<(Decoder, u32)> {
        let ret_addr = u32::try_from(ret_addr.wrapping_sub(self.code_base_addr)).ok()?;
        let pos = self.ret_addrs.binary_search_by_key(&ret_addr, |&(addr, _)| addr).ok()?;
        let offset = self.ret_addrs[pos].1;
        Some((Decoder { ptr: unsafe { self.bytes.as_ptr().add(offset as usize) as *mut u8 } }, offset))
    }

    /// Return the ISEQ version whose descriptors include the one at `offset`
    fn version_of(&self, offset: u32) -> Option<IseqVersionRef> {
        let pos = self.versions.partition_point(|(range, _)| range.start <= offset).checked_sub(1)?;
        let (range, version) = &self.versions[pos];
        range.contains(&offset).then_some(*version)
    }

    /// Call `f` with the header and a decoder positioned after it, for each descriptor
    /// in `range` of the encoded descriptors starting at `base`.
    fn each_descriptor(base: *mut u8, range: &Range<u32>, mut f: impl FnMut(&DecodedHeader, &mut Decoder)) {
        let end = unsafe { base.add(range.end as usize) };
        let mut decoder = Decoder { ptr: unsafe { base.add(range.start as usize) } };
        while decoder.ptr < end {
            let header = decoder.header();
            f(&header, &mut decoder);
        }
        debug_assert_eq!(decoder.ptr, end);
    }

    /// Call `f` on every GC-managed object field in the descriptors in `range`.
    /// Pass `self.bytes.as_mut_ptr()` as `base` to write through the fields.
    fn each_object_field(base: *mut u8, range: &Range<u32>, mut f: impl FnMut(*mut u64)) {
        Self::each_descriptor(base, range, |header, decoder| {
            f(header.iseq);
            if header.flags & FLAG_RECOMPILE != 0 {
                f(header.recompile);
            }
            decoder.visit_rest(header, |field| {
                if !VALUE(unsafe { field.read_unaligned() } as usize).special_const_p() {
                    f(field);
                }
            });
        });
    }

    /// Byte offset the next descriptor gets. Descriptors registered while compiling
    /// one ISEQ version form a contiguous range of offsets.
    pub fn len(&self) -> u32 {
        self.bytes.len() as u32
    }
}

fn read_table() -> std::sync::RwLockReadGuard<'static, ExitDescriptorTable> {
    ZJITState::get_exit_descriptors().read().unwrap_or_else(|err| err.into_inner())
}

fn write_table() -> std::sync::RwLockWriteGuard<'static, ExitDescriptorTable> {
    ZJITState::get_exit_descriptors().write().unwrap_or_else(|err| err.into_inner())
}

/// The table offset the next descriptor gets
pub fn next_offset() -> u32 {
    read_table().len()
}

/// Register an encoded descriptor for the call whose return address is `ret_addr`.
/// Called by a PosMarker right after the call to exit_descriptor_trampoline is emitted.
pub fn register(ret_addr: CodePtr, cb: &CodeBlock, encoded: &[u8]) {
    // Only calls in ZJIT's code block can reach the handler. Code assembled into
    // another CodeBlock (unit tests) is never run, and its offsets would alias.
    let code_base_addr = |cb: &CodeBlock| {
        let start = cb.get_ptr(0);
        start.raw_addr(cb) - start.as_offset() as usize
    };
    if !ZJITState::has_instance() || code_base_addr(ZJITState::get_code_block()) != code_base_addr(cb) {
        return;
    }
    let ret_addr: u32 = ret_addr.as_offset().try_into().unwrap();
    let mut table = write_table();
    table.code_base_addr = code_base_addr(cb);
    table.materialize_exit_addr = ZJITState::get_materialize_exit_trampoline().raw_addr(cb);
    table.register(ret_addr, encoded);
}

/// Record the ISEQ version that the descriptors in `range` were registered for
pub fn register_version(range: &Range<u32>, version: IseqVersionRef) {
    if range.is_empty() {
        return;
    }
    let mut table = write_table();
    debug_assert!(table.versions.last().map_or(true, |(last, _)| last.end <= range.start));
    reserve_append_only(&mut table.versions, 1);
    table.versions.push((range.clone(), version));
}

/// Write barrier for objects referenced by the descriptors of a new ISEQ version
pub fn write_barrier(iseq: IseqPtr, range: &Range<u32>) {
    let table = read_table();
    ExitDescriptorTable::each_object_field(table.bytes.as_ptr().cast_mut(), range, |field| {
        VALUE::from(iseq).write_barrier(VALUE(unsafe { field.read_unaligned() } as usize));
    });
}

/// Mark objects referenced by the descriptors of an ISEQ version
pub fn mark(range: &Range<u32>) {
    let table = read_table();
    ExitDescriptorTable::each_object_field(table.bytes.as_ptr().cast_mut(), range, |field| {
        let object = VALUE(unsafe { field.read_unaligned() } as usize);
        if object != VALUE(0) {
            unsafe { rb_gc_mark_movable(object) };
        }
    });
}

/// Update objects referenced by the descriptors of an ISEQ version after compaction
pub fn update_references(range: &Range<u32>) {
    // Fields are updated in place, which doesn't change the table's layout
    let mut table = write_table();
    ExitDescriptorTable::each_object_field(table.bytes.as_mut_ptr(), range, |field| {
        let object = VALUE(unsafe { field.read_unaligned() } as usize);
        if object != VALUE(0) {
            let new_object = unsafe { rb_gc_location(object) };
            if new_object != object {
                unsafe { field.write_unaligned(new_object.0 as u64) };
            }
        }
    });
}

/// Read the value at `loc` given the register save area `regs`. This runs for
/// every value of every taken side exit, so it avoids overflow checks.
#[inline(always)]
unsafe fn read_loc(loc: DecodedLoc, regs: *const usize, native_base_ptr: usize) -> u64 {
    #[inline(always)]
    unsafe fn read_mem(addr: usize, bits: u8) -> u64 {
        unsafe {
            match bits {
                64 => (addr as *const u64).read_unaligned(),
                32 => (addr as *const u32).read_unaligned() as u64,
                16 => (addr as *const u16).read_unaligned() as u64,
                8 => (addr as *const u8).read() as u64,
                _ => unreachable!("unexpected memory operand size: {bits}"),
            }
        }
    }
    unsafe {
        let reg = |reg_no: u8| *regs.add(reg_no as usize);
        let loc = match loc {
            DecodedLoc::Loc(loc) => loc,
            DecodedLoc::Imm(imm) => return imm,
            DecodedLoc::Value(field) => return field.read_unaligned(),
        };
        match loc {
            ExitLoc::Reg { reg: reg_no, bits: 64 } => reg(reg_no) as u64,
            ExitLoc::Reg { reg: reg_no, bits } => reg(reg_no) as u64 & (u64::MAX >> (64 - bits as u32)),
            ExitLoc::NativeSlot { disp, bits } => read_mem(native_base_ptr.wrapping_add_signed(disp as isize), bits),
            ExitLoc::Mem { base, bits, disp } => read_mem(reg(base).wrapping_add_signed(disp as isize), bits),
            ExitLoc::StackIndirect { slot, disp, bits } => {
                let base = read_mem(native_base_ptr.wrapping_add_signed(slot as isize), 64) as usize;
                read_mem(base.wrapping_add_signed(disp as isize), bits)
            }
            ExitLoc::SmallImm(imm) => imm as i64 as u64,
            ExitLoc::Imm(_) | ExitLoc::Value(_) => unreachable!("decoded immediates are returned above"),
        }
    }
}

/// Called by exit_descriptor_trampoline when JIT code takes a side exit.
///
/// `regs` is the trampoline's save area, holding every general-purpose register
/// as of the exit, indexed by machine register number. `ret_addr` is the return
/// address of the exit's call, which identifies its descriptor. Returns the address
/// the trampoline goes to after restoring every register and popping the return
/// address: materialize_exit_trampoline, or the exit's lazily compiled code when
/// this hit compiled it (see [`lazy_exit_hit`]), which then does everything below.
///
/// This does what the inline exit code used to do, in the same order: restore
/// `cfp->pc`, `cfp->sp` and `cfp->iseq`, write out the Ruby stack and locals,
/// install the stack map for older inlined frames, then optionally record a
/// traced exit stack and count the exit towards recompilation. The trampoline
/// then jumps to materialize_exit_trampoline, which clears `cfp->jit_return`
/// and materializes the JIT frames below this one. Nothing here allocates Ruby
/// objects or can trigger GC before the optional calls at the end.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_side_exit_descriptor(regs: *const usize, ret_addr: usize) -> *const u8 {
    let reg = |opnd: crate::backend::lir::Opnd| unsafe { *regs.add(opnd.unwrap_reg().reg_no as usize) };
    let cfp = reg(CFP) as CfpPtr;

    let (trace_reason, recompile, materialize_exit_addr) = {
        // Another Ractor may be compiling, i.e. appending to the table, only when
        // there are multiple Ractors. Otherwise this thread holds the GVL, which
        // compilation needs too, so skip the two atomic operations of the lock.
        let multi_ractor = unsafe { rb_jit_multi_ractor_p() };
        let guard;
        let table = if multi_ractor {
            guard = read_table();
            &*guard
        } else {
            ZJITState::get_exit_descriptors_unlocked()
        };
        let (mut decoder, offset) = table.lookup(ret_addr)
            .unwrap_or_else(|| panic!("no side-exit descriptor for return address {ret_addr:#x}"));
        let desc_start = decoder.ptr;
        let header = decoder.header();

        // Patching code while other Ractors may run it would need a barrier, so
        // exits stay descriptor-only once there are multiple Ractors.
        if !multi_ractor {
            if let Some(code) = lazy_exit_hit(table, &header, desc_start, offset, ret_addr) {
                return code;
            }
        }
        let sp = reg(SP) as *mut VALUE;
        let native_base_ptr = reg(NATIVE_BASE_PTR);

        unsafe {
            (*cfp).pc = header.pc;
            (*cfp).sp = sp.add(header.num_stack);
            (*cfp)._iseq = header.iseq.read_unaligned() as IseqPtr;
            // cfp->block_code and cfp->jit_return are cleared by materialize_exit_trampoline

            // Stack slots are written from SP upwards
            for idx in 0..header.num_stack {
                sp.add(idx).write(VALUE(read_loc(decoder.loc(), regs, native_base_ptr) as usize));
            }

            // Local i lives at EP[-local_size_and_idx_to_ep_offset(n, i)] with EP = SP - 1,
            // i.e. SP[i - n - VM_ENV_DATA_SIZE], so locals are contiguous too.
            let num_locals = header.num_locals;
            let locals_base = sp.sub(num_locals + VM_ENV_DATA_SIZE as usize);
            debug_assert!(num_locals == 0 || locals_base == sp.offset(-local_size_and_idx_to_ep_offset(num_locals, 0) as isize - 1));
            for idx in 0..num_locals {
                locals_base.add(idx).write(VALUE(read_loc(decoder.loc(), regs, native_base_ptr) as usize));
            }

            if header.flags & FLAG_STACK_MAP != 0 {
                let jit_frame = decoder.u64() as *const zjit_jit_frame;
                let jit_frame_slot = decoder.svarint() as isize;
                for _ in 0..decoder.varint() {
                    let value = read_loc(decoder.loc(), regs, native_base_ptr);
                    let slot = decoder.svarint() as isize;
                    (native_base_ptr.wrapping_add_signed(slot) as *mut u64).write(value);
                }
                (native_base_ptr.wrapping_add_signed(jit_frame_slot) as *mut *const zjit_jit_frame).write(jit_frame);
            }
        }

        let trace_reason = (header.flags & FLAG_TRACE != 0).then(|| decoder.u64() as *const c_char);
        let recompile = if header.recompile.is_null() {
            std::ptr::null()
        } else {
            unsafe { header.recompile.read_unaligned() as IseqPtr }
        };
        (trace_reason, recompile, table.materialize_exit_addr)
    };

    if trace_reason.is_some() || !recompile.is_null() {
        // Clear cfp->jit_return to prepare for a C call. Normally, cfp->jit_return
        // is cleared by the materialize_exit trampoline, but if we're about to
        // make a C call, we need to clear any stale JITFrame.
        unsafe { (*cfp).jit_return = std::ptr::null_mut(); }
    }
    if let Some(reason) = trace_reason {
        rb_zjit_record_exit_stack(reason);
    }
    if !recompile.is_null() {
        exit_recompile(VALUE::from(recompile));
    }

    materialize_exit_addr as *const u8
}

/// `hits` value of a descriptor that is no longer counted: its exit has been
/// compiled, or compiling it failed and it stays descriptor-only.
const LAZY_EXIT_DONE: u8 = u8::MAX;
/// Largest hit count. `--zjit-lazy-exit-threshold` above this never compiles.
pub const LAZY_EXIT_MAX_HITS: u8 = LAZY_EXIT_DONE - 1;

/// Count a hit of the exit whose descriptor starts at `desc_start` (table offset
/// `offset`), and compile the exit into machine code once it has been materialized
/// from its descriptor `--zjit-lazy-exit-threshold` times.
///
/// This is how JavaScriptCore compiles DFG/FTL OSR exits: the exit site initially
/// calls a generic handler, and the first time an exit is taken, `compileOSRExit`
/// generates specialized exit code and repatches the exit to jump to it. Here the
/// descriptor handler is the generic path, which ZJIT keeps for exits that are
/// taken only a few times, so that exits that never fire (most of them) cost no
/// code memory and exits that fire in a loop run the same code the inline exit
/// stubs used to be.
///
/// Returns the compiled code if this hit compiled it. The trampoline then jumps
/// there with every register restored, so the compiled code does the whole exit.
fn lazy_exit_hit(table: &ExitDescriptorTable, header: &DecodedHeader, desc_start: *mut u8, offset: u32, ret_addr: usize) -> Option<*const u8> {
    let hits = unsafe { header.hits.read() };
    if hits == LAZY_EXIT_DONE {
        return None;
    }
    if hits == 0 {
        incr_counter!(exit_descriptor_taken_count);
    }
    let threshold = get_option!(lazy_exit_threshold);
    if threshold > LAZY_EXIT_MAX_HITS as usize || (hits as usize) < threshold {
        // Saturate so that a threshold above LAZY_EXIT_MAX_HITS keeps counting distinct exits
        unsafe { header.hits.write(hits.saturating_add(1).min(LAZY_EXIT_MAX_HITS)) };
        return None;
    }

    // Whether or not this works, stop counting this exit
    unsafe { header.hits.write(LAZY_EXIT_DONE) };
    let mut decoder = Decoder { ptr: desc_start };
    let header = decoder.header();
    let desc = decoder.descriptor(&header);
    let version = table.version_of(offset);
    // There is only one Ractor, whose thread holds the GVL, so no VM lock is needed
    let code = version.and_then(|version| compile_lazy_exit(&desc, version, ret_addr - table.code_base_addr));
    if code.is_none() {
        incr_counter!(lazy_exit_compile_failure_count);
    }
    code
}

/// Convert where a descriptor finds a value into an operand of lazily compiled
/// exit code. A spilled memory base is loaded into LAZY_EXIT_BASE_OPND first.
fn exit_loc_opnd(asm: &mut Assembler, desc: &ExitDescriptor, loc: ExitLoc) -> Opnd {
    use crate::backend::lir::{gp_reg, LAZY_EXIT_BASE_OPND};
    match loc {
        ExitLoc::Reg { reg, bits } => Opnd::Reg(gp_reg(reg, bits)),
        ExitLoc::NativeSlot { disp, bits } => Opnd::mem(bits, NATIVE_BASE_PTR, disp),
        ExitLoc::Mem { base, bits, disp } => Opnd::mem(bits, Opnd::Reg(gp_reg(base, 64)), disp),
        ExitLoc::StackIndirect { slot, disp, bits } => {
            asm.load_into(LAZY_EXIT_BASE_OPND, Opnd::mem(64, NATIVE_BASE_PTR, slot));
            Opnd::mem(bits, LAZY_EXIT_BASE_OPND, disp)
        }
        ExitLoc::SmallImm(imm) => Opnd::Imm(imm as i64),
        ExitLoc::Imm(idx) => Opnd::UImm(desc.imms[idx as usize]),
        ExitLoc::Value(idx) => Opnd::Value(VALUE(desc.imms[idx as usize] as usize)),
    }
}

/// Generate the code of a side exit from its descriptor: what the inline exit stub
/// emitted by compile_exits() did before descriptors, in the same order as
/// [`rb_zjit_side_exit_descriptor`]. It runs with every register as of the exit
/// and ends by jumping to materialize_exit_trampoline.
fn gen_lazy_exit(desc: &ExitDescriptor) -> Assembler {
    use crate::cruby::{RUBY_OFFSET_CFP_ISEQ, RUBY_OFFSET_CFP_JIT_RETURN, RUBY_OFFSET_CFP_PC, RUBY_OFFSET_CFP_SP, SIZEOF_VALUE_I32};
    let mut asm = Assembler::new();
    asm.new_block_without_id("lazy_side_exit");
    asm_comment!(asm, "lazily compiled side exit: {:?}", desc.pc);

    asm_comment!(asm, "save cfp->pc");
    asm.store(Opnd::mem(64, CFP, RUBY_OFFSET_CFP_PC), Opnd::const_ptr(desc.pc));
    asm_comment!(asm, "save cfp->sp");
    asm.lea_into(Opnd::mem(64, CFP, RUBY_OFFSET_CFP_SP), Opnd::mem(64, SP, desc.num_stack as i32 * SIZEOF_VALUE_I32));
    asm_comment!(asm, "save cfp->iseq");
    asm.store(Opnd::mem(64, CFP, RUBY_OFFSET_CFP_ISEQ), VALUE::from(desc.iseq).into());
    // cfp->block_code and cfp->jit_return are cleared by materialize_exit_trampoline

    if !desc.stack().is_empty() {
        asm_comment!(asm, "write stack slots");
        for (idx, &loc) in desc.stack().iter().enumerate() {
            let opnd = exit_loc_opnd(&mut asm, desc, loc);
            asm.store(Opnd::mem(64, SP, idx as i32 * SIZEOF_VALUE_I32), opnd);
        }
    }
    let num_locals = desc.locals().len();
    if num_locals > 0 {
        asm_comment!(asm, "write locals");
        for (idx, &loc) in desc.locals().iter().enumerate() {
            let opnd = exit_loc_opnd(&mut asm, desc, loc);
            asm.store(Opnd::mem(64, SP, (-local_size_and_idx_to_ep_offset(num_locals, idx) - 1) * SIZEOF_VALUE_I32), opnd);
        }
    }
    if let Some(stack_map) = desc.stack_map() {
        asm_comment!(asm, "install side-exit JITFrame");
        for &(loc, slot) in stack_map.captures.iter() {
            let opnd = exit_loc_opnd(&mut asm, desc, loc);
            asm.store(Opnd::mem(64, NATIVE_BASE_PTR, slot), opnd);
        }
        asm.store(Opnd::mem(64, NATIVE_BASE_PTR, stack_map.jit_frame_slot), Opnd::const_ptr(stack_map.jit_frame));
    }

    let trace_reason = desc.trace_reason();
    if trace_reason.is_some() || !desc.recompile.is_null() {
        asm_comment!(asm, "clear cfp->jit_return");
        asm.store(Opnd::mem(64, CFP, RUBY_OFFSET_CFP_JIT_RETURN), 0.into());
    }
    if let Some(reason) = trace_reason {
        asm_comment!(asm, "call rb_zjit_record_exit_stack");
        asm.ccall_into(C_RET_OPND, rb_zjit_record_exit_stack as *const u8, vec![Opnd::const_ptr(reason)]);
    }
    if !desc.recompile.is_null() {
        asm_comment!(asm, "call exit_recompile");
        asm.ccall_into(C_RET_OPND, exit_recompile as *const u8, vec![VALUE::from(desc.recompile).into()]);
    }

    asm_comment!(asm, "exit to the interpreter");
    asm.jmp(Target::CodePtr(ZJITState::get_materialize_exit_trampoline()));
    asm
}

/// Compile a side exit from its descriptor at the code block's write position, and
/// patch the exit's call to exit_descriptor_trampoline, which ends at code offset
/// `ret_offset`, into a jump to it. Returns the compiled code, or None if the call
/// can't be patched or the code block is out of memory.
fn compile_lazy_exit(desc: &ExitDescriptor, version: IseqVersionRef, ret_offset: usize) -> Option<*const u8> {
    let cb = ZJITState::get_code_block();
    let call_end = cb.get_ptr(ret_offset);
    if !Assembler::exit_call_patchable(cb, call_end) || cb.has_dropped_bytes() {
        return None;
    }

    let asm = gen_lazy_exit(desc);
    let result = asm.compile_exit_code(cb);
    let patched = match result {
        Ok((start_ptr, gc_offsets)) => {
            let size = cb.get_write_ptr().as_offset() - start_ptr.as_offset();
            incr_counter!(lazy_exit_compiled_count);
            incr_counter_by(Counter::lazy_exit_compiled_bytes, size as u64);
            crate::perf::register_current_code_range(cb, "lazy side exit", start_ptr);
            #[cfg(feature = "disasm")]
            if let Some(dump_disasm) = crate::options::get_option_ref!(dump_disasm) {
                crate::disasm::dump_disasm_addr_range(cb, start_ptr, cb.get_write_ptr(), dump_disasm);
            }
            crate::gc::append_gc_offsets(unsafe { version.as_ref() }.iseq, version, &gc_offsets);
            Assembler::patch_exit_call(cb, call_end, start_ptr).then(|| start_ptr.raw_ptr(cb))
        }
        Err(err) => {
            // Like Assembler::compile(), let later compilation retry if memory is freed
            if err == crate::stats::CompileError::OutOfMemory {
                cb.update_dropped_bytes();
            }
            None
        }
    };
    cb.mark_all_executable();
    patched
}

/// Decode the descriptors registered for every compiled version of `iseq`
#[cfg(test)]
pub fn descriptors_of(iseq: IseqPtr) -> Vec<ExitDescriptor> {
    let payload = crate::payload::get_or_create_iseq_payload(iseq);
    let table = read_table();
    let mut descs = vec![];
    for version in payload.versions.iter() {
        let range = unsafe { version.as_ref() }.exit_descs.clone();
        ExitDescriptorTable::each_descriptor(table.bytes.as_ptr().cast_mut(), &range, |header, decoder| {
            descs.push(decoder.descriptor(header));
        });
    }
    descs
}

#[cfg(test)]
impl ExitDescriptor {
    /// Return all locations: stack, locals and stack-map captures
    pub fn all_locs(&self) -> Vec<ExitLoc> {
        let captures = self.stack_map().into_iter().flat_map(|stack_map| stack_map.captures.iter().map(|&(loc, _)| loc));
        self.locs.iter().copied().chain(captures).collect()
    }

    pub fn has_stack_map(&self) -> bool {
        self.stack_map().is_some()
    }

    pub fn has_trace_reason(&self) -> bool {
        self.trace_reason().is_some()
    }

    /// Resolve an immediate location
    pub fn imm_value(&self, loc: ExitLoc) -> Option<u64> {
        match loc {
            ExitLoc::SmallImm(imm) => Some(imm as i64 as u64),
            ExitLoc::Imm(idx) | ExitLoc::Value(idx) => Some(self.imms[idx as usize]),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut imms = ExitImms::default();
        let locs = vec![
            ExitLoc::Reg { reg: 7, bits: 64 },
            ExitLoc::Reg { reg: 3, bits: 32 },
            ExitLoc::NativeSlot { disp: -48, bits: 64 },
            ExitLoc::NativeSlot { disp: -8000, bits: 32 },
            ExitLoc::Mem { base: 13, bits: 64, disp: 24 },
            ExitLoc::StackIndirect { slot: -16, disp: 8, bits: 64 },
            imms.imm(Qnil.as_u64(), true),
            imms.imm(-5i64 as u64, false),
            imms.imm(1 << 40, false),
            imms.imm(0x1234_5678_9abc_def0, true),
        ];
        assert!(matches!(locs[6], ExitLoc::SmallImm(_)));
        assert!(matches!(locs[8], ExitLoc::Imm(_)));
        assert!(matches!(locs[9], ExitLoc::Value(_)));
        let stack_map = ExitStackMap { captures: vec![(ExitLoc::Reg { reg: 1, bits: 64 }, -40)].into_boxed_slice(), jit_frame: 0x1000 as *const _, jit_frame_slot: -16 };
        let desc = ExitDescriptor::new(0x5000 as *const VALUE, 0x6000 as IseqPtr, locs[..4].to_vec(), locs[4..].to_vec(), imms, 0x7000 as IseqPtr,
            Some(Box::new(ExitDescriptorExtra { stack_map: Some(stack_map), trace_reason: Some(0x8000 as *const c_char) })));
        let mut encoded = desc.encode().into_vec();
        encoded.extend_from_slice(&desc.encode());

        let bytes = encoded.as_mut_ptr_range();
        let mut decoder = Decoder { ptr: bytes.start };
        for _ in 0..2 {
            let header = decoder.header();
            assert_eq!(header.pc, desc.pc);
            assert_eq!(unsafe { header.iseq.read_unaligned() }, 0x6000);
            assert_eq!(unsafe { header.recompile.read_unaligned() }, 0x7000);
            assert_eq!((header.num_stack, header.num_locals), (4, 6));
            for &expected in &locs {
                let actual = match decoder.loc() {
                    DecodedLoc::Loc(loc) => loc,
                    DecodedLoc::Imm(imm) => { assert_eq!(Some(imm), desc.imm_value(expected)); expected }
                    DecodedLoc::Value(field) => { assert_eq!(Some(unsafe { field.read_unaligned() }), desc.imm_value(expected)); expected }
                };
                assert_eq!(actual, expected);
            }
            assert_eq!(decoder.u64(), 0x1000);
            assert_eq!(decoder.svarint(), -16);
            assert_eq!(decoder.varint(), 1);
            assert!(matches!(decoder.loc(), DecodedLoc::Loc(ExitLoc::Reg { reg: 1, bits: 64 })));
            assert_eq!(decoder.svarint(), -40);
            assert_eq!(decoder.u64(), 0x8000);
        }
        assert_eq!(decoder.ptr, bytes.end);
    }
}
