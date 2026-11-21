//! Side-exit descriptors: what a side exit does, kept off the code region.
//!
//! The Ruby stack and locals of a side exit are described by the stack map of its
//! JITFrame, which `rb_zjit_materialize_frames()` decodes when the exit is taken.
//! The stack map refers to values in native stack slots, so values the register
//! allocator keeps in registers have to be stored ("captured") into reserved slots
//! first. The rest of an exit is saving `cfp->sp`, installing the JITFrame pointer,
//! and optionally recording a traced exit stack and counting the exit towards
//! recompilation. Emitting all of that as machine code costs ~35 bytes per exit,
//! interleaved with the hot code it guards.
//!
//! Like V8's deopt exits, HotSpot's uncommon traps and Cinder's side-exit helper, ZJIT
//! instead emits each exit as a single `call`/`bl` into `exit_descriptor_trampoline`.
//! The trampoline saves every general-purpose register into a save area on the machine
//! stack and calls [`rb_zjit_side_exit_descriptor`] with that area and the call's return
//! address. The return address identifies the descriptor registered for that call site,
//! which the handler executes exactly like the inline exit code used to, and the
//! trampoline then returns into `materialize_exit_trampoline`.
//!
//! Descriptors are built as [`ExitDescriptor`]s at compile time and stored in a
//! compact byte encoding: a captured register is one byte plus one for its slot, and
//! pointers the GC may move are stored as 8 raw bytes so that compaction can update
//! them in place.

use std::ffi::c_char;
use std::ops::Range;

use crate::backend::lir::{CFP, NATIVE_BASE_PTR, SP};
use crate::codegen::exit_recompile;
use crate::cruby::*;
use crate::state::{rb_zjit_record_exit_stack, ZJITState};
use crate::stats::{incr_counter, incr_counter_by, Counter};
use crate::virtualmem::{CodePtr, CodePtrBase};
use crate::asm::CodeBlock;

/// Where a side exit finds a value to capture into a native stack slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExitLoc {
    /// A general-purpose register, by machine register number (`reg_no`), which
    /// indexes the trampoline's register save area.
    Reg(u8),
    /// A raw 64-bit immediate
    Imm(u64),
    /// A Ruby VALUE. Marked and moved by the GC when it is a heap object.
    Value(VALUE),
}

/// Everything a side exit does, described for [`rb_zjit_side_exit_descriptor`].
/// This is the compile-time form; the table stores it encoded by [`Self::encode`].
#[derive(Clone, Debug)]
pub struct ExitDescriptor {
    /// Number of Ruby stack slots, for `cfp->sp`
    pub vm_stack_size: usize,
    /// JITFrame to install. Its stack map is written at compile time.
    pub jit_frame: *const zjit_jit_frame,
    /// Inlining depth of the exiting frame, which selects the JITFrame slot
    pub frame_depth: usize,
    /// (source, byte offset from NATIVE_BASE_PTR of the capture slot)
    pub captures: Vec<(ExitLoc, i32)>,
    /// If not null, the compiled ISEQ to pass to exit_recompile(). For an exit out
    /// of inlined code, this is the outer ISEQ.
    pub recompile: IseqPtr,
    /// Reason to pass to rb_zjit_record_exit_stack() for --zjit-trace-side-exits
    pub trace_reason: Option<*const c_char>,
}

// Descriptor encoding. All multi-byte integers are little-endian; "varint" is LEB128.
//
//   u8 flags, varint vm_stack_size, varint jit_frame, varint frame_depth,
//   varint num_captures, (loc, varint slot) * num_captures,
//   [u64 recompile], [u64 trace_reason]
//
// A capture slot is stored as -disp / SIZEOF_VALUE, which is positive. JITFrames
// are usually allocated below INT32_MAX, so `jit_frame` usually takes 4-5 bytes.
// A loc starts with a byte whose low 2 bits are a tag and high 6 bits a payload.
const FLAG_RECOMPILE: u8 = 1 << 0; // a u64 recompile ISEQ follows
const FLAG_TRACE: u8 = 1 << 1; // a u64 trace reason follows

const TAG_REG: u8 = 0; // payload: reg_no
const TAG_VALUE: u8 = 1; // u64 VALUE, updated in place by compaction
const TAG_IMM: u8 = 2; // u64 imm
const TAG_BITS: u32 = 2;
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

    fn tag(&mut self, tag: u8, payload: u8) {
        assert!(payload < (1 << (8 - TAG_BITS)));
        self.u8(tag | (payload << TAG_BITS));
    }

    fn loc(&mut self, loc: ExitLoc) {
        match loc {
            ExitLoc::Reg(reg) => self.tag(TAG_REG, reg),
            ExitLoc::Value(value) => { self.tag(TAG_VALUE, 0); self.u64(value.0 as u64); }
            ExitLoc::Imm(imm) => { self.tag(TAG_IMM, 0); self.u64(imm); }
        }
    }

    fn slot(&mut self, disp: i32) {
        assert!(disp < 0 && disp % SIZEOF_VALUE_I32 == 0, "unexpected capture slot: {disp}");
        self.varint((-disp / SIZEOF_VALUE_I32) as u64);
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
    fn slot(&mut self) -> isize {
        -(self.varint() as isize) * SIZEOF_VALUE as isize
    }

    #[inline(always)]
    fn loc(&mut self) -> DecodedLoc {
        let byte = self.u8();
        match byte & TAG_MASK {
            TAG_REG => DecodedLoc::Reg(byte >> TAG_BITS),
            TAG_VALUE => DecodedLoc::Value(self.u64_ptr()),
            _ => DecodedLoc::Imm(self.u64()),
        }
    }
}

enum DecodedLoc {
    Reg(u8),
    Imm(u64),
    /// The 8-byte field of a VALUE, so that the GC can update it in place
    Value(*mut u64),
}

/// Header of an encoded descriptor, up to the captures
struct DecodedHeader {
    flags: u8,
    vm_stack_size: usize,
    jit_frame: *const zjit_jit_frame,
    frame_depth: usize,
    num_captures: usize,
}

impl Decoder {
    #[inline(always)]
    fn header(&mut self) -> DecodedHeader {
        let flags = self.u8();
        let vm_stack_size = self.varint() as usize;
        let jit_frame = self.varint() as *const zjit_jit_frame;
        let frame_depth = self.varint() as usize;
        let num_captures = self.varint() as usize;
        DecodedHeader { flags, vm_stack_size, jit_frame, frame_depth, num_captures }
    }

    /// Visit the rest of a descriptor after its header, calling `on_object` with each
    /// field that may hold a GC-managed object. Leaves the cursor at the next descriptor.
    fn visit_rest(&mut self, header: &DecodedHeader, mut on_object: impl FnMut(*mut u64)) {
        for _ in 0..header.num_captures {
            if let DecodedLoc::Value(field) = self.loc() {
                on_object(field);
            }
            self.slot();
        }
        if header.flags & FLAG_RECOMPILE != 0 {
            on_object(self.u64_ptr());
        }
        if header.flags & FLAG_TRACE != 0 {
            self.u64();
        }
    }
}

impl ExitDescriptor {
    /// Encode this descriptor for the table
    pub fn encode(&self) -> Box<[u8]> {
        let mut enc = Encoder(Vec::with_capacity(16));
        let mut flags = 0;
        if !self.recompile.is_null() {
            flags |= FLAG_RECOMPILE;
        }
        if self.trace_reason.is_some() {
            flags |= FLAG_TRACE;
        }
        enc.u8(flags);
        enc.varint(self.vm_stack_size as u64);
        enc.varint(self.jit_frame as u64);
        enc.varint(self.frame_depth as u64);
        enc.varint(self.captures.len() as u64);
        for &(loc, slot) in &self.captures {
            enc.loc(loc);
            enc.slot(slot);
        }
        if !self.recompile.is_null() {
            enc.u64(self.recompile as u64);
        }
        if let Some(reason) = self.trace_reason {
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
        incr_counter!(side_exit_descriptor_count);
        incr_counter_by(Counter::side_exit_descriptor_bytes, (encoded.len() + std::mem::size_of::<(u32, u32)>()) as u64);
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

    /// Return a decoder at the descriptor for a return address
    #[inline(always)]
    fn lookup(&self, ret_addr: usize) -> Option<Decoder> {
        let code_base_addr = ZJITState::get_code_region_base().base_ptr().as_ptr() as usize;
        let ret_addr = u32::try_from(ret_addr.wrapping_sub(code_base_addr)).ok()?;
        let pos = self.ret_addrs.binary_search_by_key(&ret_addr, |&(addr, _)| addr).ok()?;
        let offset = self.ret_addrs[pos].1 as usize;
        Some(Decoder { ptr: unsafe { self.bytes.as_ptr().add(offset) as *mut u8 } })
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
    if !ZJITState::has_instance() {
        return 0;
    }
    read_table().len()
}

/// Register an encoded descriptor for the call whose return address is `ret_addr`.
/// Called by a PosMarker right after the call to exit_descriptor_trampoline is emitted.
pub fn register(ret_addr: CodePtr, cb: &CodeBlock, encoded: &[u8]) {
    // Only calls in ZJIT's code block can reach the handler. Code assembled into
    // another CodeBlock (unit tests) is never run, and its offsets would alias.
    if !ZJITState::has_instance() || ZJITState::get_code_region_base().base_ptr() != cb.base_ptr() {
        return;
    }
    let ret_addr: u32 = ret_addr.as_offset().try_into().unwrap();
    write_table().register(ret_addr, encoded);
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
    if range.is_empty() {
        return;
    }
    let table = read_table();
    ExitDescriptorTable::each_object_field(table.bytes.as_ptr().cast_mut(), range, |field| {
        unsafe { rb_gc_mark_movable(VALUE(field.read_unaligned() as usize)) };
    });
}

/// Update objects referenced by the descriptors of an ISEQ version after compaction
pub fn update_references(range: &Range<u32>) {
    if range.is_empty() {
        return;
    }
    // Fields are updated in place, which doesn't change the table's layout
    let mut table = write_table();
    ExitDescriptorTable::each_object_field(table.bytes.as_mut_ptr(), range, |field| {
        let object = VALUE(unsafe { field.read_unaligned() } as usize);
        let new_object = unsafe { rb_gc_location(object) };
        if new_object != object {
            unsafe { field.write_unaligned(new_object.0 as u64) };
        }
    });
}

/// Called by exit_descriptor_trampoline when JIT code takes a side exit.
///
/// `regs` is the trampoline's save area, holding every general-purpose register
/// as of the exit, indexed by machine register number. `ret_addr` is the return
/// address of the exit's call, which identifies its descriptor. Returns `regs`.
///
/// This does what the inline exit code used to do, in the same order: save
/// `cfp->sp`, capture register values the JITFrame's stack map refers to into
/// their native stack slots, install the JITFrame, then optionally record a
/// traced exit stack and count the exit towards recompilation. The trampoline
/// then returns into materialize_exit_trampoline, which decodes the stack maps.
/// Nothing here allocates Ruby objects or can trigger GC before the optional
/// calls at the end.
#[unsafe(no_mangle)]
pub extern "C" fn rb_zjit_side_exit_descriptor(regs: *const usize, ret_addr: usize) -> *const usize {
    let reg = |reg_no: u8| unsafe { *regs.add(reg_no as usize) };
    let reg_opnd = |opnd: crate::backend::lir::Opnd| reg(opnd.unwrap_reg().reg_no);
    let cfp = reg_opnd(CFP) as CfpPtr;

    let (trace_reason, recompile) = {
        // Another Ractor may be compiling, i.e. appending to the table, only when
        // there are multiple Ractors. Otherwise this thread holds the GVL, which
        // compilation needs too, so skip the two atomic operations of the lock.
        let guard;
        let table = if unsafe { rb_jit_multi_ractor_p() } {
            guard = read_table();
            &*guard
        } else {
            ZJITState::get_exit_descriptors_unlocked()
        };
        let mut decoder = table.lookup(ret_addr)
            .unwrap_or_else(|| panic!("no side-exit descriptor for return address {ret_addr:#x}"));
        let header = decoder.header();
        let sp = reg_opnd(SP) as *mut VALUE;
        let native_base_ptr = reg_opnd(NATIVE_BASE_PTR);

        unsafe {
            (*cfp).sp = sp.add(header.vm_stack_size);
            for _ in 0..header.num_captures {
                let value = match decoder.loc() {
                    DecodedLoc::Reg(reg_no) => reg(reg_no) as u64,
                    DecodedLoc::Imm(imm) => imm,
                    DecodedLoc::Value(field) => field.read_unaligned(),
                };
                (native_base_ptr.wrapping_add_signed(decoder.slot()) as *mut u64).write(value);
            }
            let jit_frame_slot = -((header.frame_depth as isize + 1) * SIZEOF_VALUE as isize);
            (native_base_ptr.wrapping_add_signed(jit_frame_slot) as *mut *const zjit_jit_frame).write(header.jit_frame);
        }

        let recompile = if header.flags & FLAG_RECOMPILE != 0 { decoder.u64() as IseqPtr } else { std::ptr::null() };
        let trace_reason = (header.flags & FLAG_TRACE != 0).then(|| decoder.u64() as *const c_char);
        (trace_reason, recompile)
    };

    if let Some(reason) = trace_reason {
        rb_zjit_record_exit_stack(reason);
    }
    if !recompile.is_null() {
        exit_recompile(VALUE::from(recompile));
    }

    // Hand the save area back so that the trampoline can find the return address slot
    regs
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
            descs.push(decode_rest(header, decoder));
        });
    }
    descs
}

#[cfg(test)]
fn decode_rest(header: &DecodedHeader, decoder: &mut Decoder) -> ExitDescriptor {
    let captures = (0..header.num_captures).map(|_| {
        let loc = match decoder.loc() {
            DecodedLoc::Reg(reg) => ExitLoc::Reg(reg),
            DecodedLoc::Imm(imm) => ExitLoc::Imm(imm),
            DecodedLoc::Value(field) => ExitLoc::Value(VALUE(unsafe { field.read_unaligned() } as usize)),
        };
        (loc, decoder.slot() as i32)
    }).collect();
    let recompile = if header.flags & FLAG_RECOMPILE != 0 { decoder.u64() as IseqPtr } else { std::ptr::null() };
    let trace_reason = (header.flags & FLAG_TRACE != 0).then(|| decoder.u64() as *const c_char);
    ExitDescriptor {
        vm_stack_size: header.vm_stack_size,
        jit_frame: header.jit_frame,
        frame_depth: header.frame_depth,
        captures,
        recompile,
        trace_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let desc = ExitDescriptor {
            vm_stack_size: 300,
            jit_frame: 0x7fff_0000 as *const _,
            frame_depth: 2,
            captures: vec![
                (ExitLoc::Reg(7), -8),
                (ExitLoc::Reg(31), -8000),
                (ExitLoc::Imm(1 << 40), -16),
                (ExitLoc::Value(VALUE(0x1234_5678_9abc_def0)), -24),
            ],
            recompile: 0x7000 as IseqPtr,
            trace_reason: Some(0x8000 as *const c_char),
        };
        let mut encoded = desc.encode().into_vec();
        encoded.extend_from_slice(&desc.encode());

        let bytes = encoded.as_mut_ptr_range();
        let mut decoder = Decoder { ptr: bytes.start };
        for _ in 0..2 {
            let header = decoder.header();
            let decoded = decode_rest(&header, &mut decoder);
            assert_eq!(decoded.vm_stack_size, desc.vm_stack_size);
            assert_eq!(decoded.jit_frame, desc.jit_frame);
            assert_eq!(decoded.frame_depth, desc.frame_depth);
            assert_eq!(decoded.captures, desc.captures);
            assert_eq!(decoded.recompile, desc.recompile);
            assert_eq!(decoded.trace_reason, desc.trace_reason);
        }
        assert_eq!(decoder.ptr, bytes.end);
    }

    #[test]
    fn test_encoded_size() {
        // A typical exit: two captured registers and a low JITFrame
        let desc = ExitDescriptor {
            vm_stack_size: 3,
            jit_frame: 0x0800_0000 as *const _,
            frame_depth: 0,
            captures: vec![(ExitLoc::Reg(0), -48), (ExitLoc::Reg(1), -56)],
            recompile: std::ptr::null(),
            trace_reason: None,
        };
        assert_eq!(desc.encode().len(), 1 + 1 + 4 + 1 + 1 + 2 * 2);
    }
}
