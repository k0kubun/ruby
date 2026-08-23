//! High-level intermediary representation (IR) in static single-assignment (SSA) form.

// We use the YARV bytecode constants which have a CRuby-style name
#![allow(non_upper_case_globals)]

#![allow(clippy::if_same_then_else)]
#![allow(clippy::match_like_matches_macro)]
use crate::{
    backend::lir::C_ARG_OPNDS, cast::IntoUsize, cruby::*, invariants::{self, iseq_seen_ep_escape}, json::Json, options::{DumpHIR, InlineDepth, debug, get_option}, payload::get_or_create_iseq_payload, profile::reset_profiles_remaining, state::{self, ZJITState},
};
use std::{
    cell::{Cell, RefCell}, collections::VecDeque, ffi::{c_void, c_uint, c_int, CStr}, fmt::Display, ptr, slice::Iter,
    sync::atomic::Ordering,
};
use crate::hir_type::{Type, types};
use crate::hir_effect::{Effect, abstract_heaps, effects};
use crate::bitset::{BitMatrix, BitSet};
use crate::fasthash::{FastHashMap as HashMap, FastHashSet as HashSet};
use crate::profile::{TypeDistributionSummary, ProfiledType};
use crate::stats::{Counter, incr_counter};
use SendFallbackReason::*;

pub(crate) mod tests;
mod opt_tests;

#[allow(unused_macros)]
macro_rules! hir_comment {
    ($func:expr, $block:expr, $($arg:tt)*) => {
        // If a diagnostic dump is requested, enrich it with HIR comments. Otherwise, avoid
        // allocating comment strings or adding comment instructions that nobody can observe.
        let enable_comment = $crate::options::get_option_ref!(dump_hir_init).is_some() ||
            $crate::options::get_option_ref!(dump_hir_opt).is_some() ||
            $crate::options::get_option_ref!(dump_hir_graphviz).is_some() ||
            $crate::options::get_option!(dump_hir_iongraph) ||
            $crate::options::get_option_ref!(dump_lir).is_some() ||
            $crate::options::get_option_ref!(dump_disasm).is_some();
        if enable_comment {
            $func.push_comment($block, format!($($arg)*));
        }
    };
}

#[allow(unused_imports)]
pub(crate) use hir_comment;
use crate::options::INLINE_BUDGET_UNLIMITED;

/// An index of an [`Insn`] in a [`Function`]. This is a popular
/// type since this effectively acts as a pointer to an [`Insn`].
/// See also: [`Function::find`].
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug)]
pub struct InsnId(pub u32);

impl IntoUsize for InsnId {
    fn to_usize(self) -> usize {
        self.0.to_usize()
    }
}

impl From<InsnId> for usize {
    fn from(val: InsnId) -> Self {
        val.to_usize()
    }
}

impl From<usize> for InsnId {
    fn from(val: usize) -> Self {
        InsnId(val.try_into().expect("InsnId should fit in u32"))
    }
}

impl std::fmt::Display for InsnId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// The index of a [`Block`], which effectively acts like a pointer.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl IntoUsize for BlockId {
    fn to_usize(self) -> usize {
        self.0.to_usize()
    }
}

impl From<BlockId> for usize {
    fn from(val: BlockId) -> Self {
        val.to_usize()
    }
}

impl From<usize> for BlockId {
    fn from(val: usize) -> Self {
        BlockId(val.try_into().expect("BlockId should fit in u32"))
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

type InsnSet = BitSet<InsnId>;
type BlockSet = BitSet<BlockId>;

fn write_vec<T: std::fmt::Display>(f: &mut std::fmt::Formatter, objs: &Vec<T>) -> std::fmt::Result {
    write!(f, "[")?;
    let mut prefix = "";
    for obj in objs {
        write!(f, "{prefix}{obj}")?;
        prefix = ", ";
    }
    write!(f, "]")
}

impl std::fmt::Display for VALUE {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.print(&PtrPrintMap::identity()).fmt(f)
    }
}

impl VALUE {
    pub fn print(self, ptr_map: &PtrPrintMap) -> VALUEPrinter<'_> {
        VALUEPrinter { inner: self, ptr_map }
    }
}

/// Print adaptor for [`VALUE`]. See [`PtrPrintMap`].
pub struct VALUEPrinter<'a> {
    inner: VALUE,
    ptr_map: &'a PtrPrintMap,
}

impl<'a> std::fmt::Display for VALUEPrinter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.inner {
            val if val.fixnum_p() => write!(f, "{}", val.as_fixnum()),
            Qnil => write!(f, "nil"),
            Qtrue => write!(f, "true"),
            Qfalse => write!(f, "false"),
            val => write!(f, "VALUE({:p})", self.ptr_map.map_ptr(val.as_ptr::<VALUE>())),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct BranchEdge {
    pub target: BlockId,
    pub args: Vec<InsnId>,
}

impl std::fmt::Display for BranchEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}(", self.target)?;
        let mut prefix = "";
        for arg in &self.args {
            write!(f, "{prefix}{arg}")?;
            prefix = ", ";
        }
        write!(f, ")")
    }
}

/// Invalidation reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Invariant {
    /// Basic operation is redefined
    BOPRedefined {
        /// {klass}_REDEFINED_OP_FLAG
        klass: RedefinitionFlag,
        /// BOP_{bop}
        bop: ruby_basic_operators,
    },
    MethodRedefined {
        /// The class object whose method we want to assume unchanged
        klass: VALUE,
        /// The method ID of the method we want to assume unchanged
        method: ID,
        /// The callable method entry that we want to track
        cme: *const rb_callable_method_entry_t,
    },
    /// No class below `klass` overrides `method`, so `method` resolves to the same
    /// callable method entry for every instance of `klass` and of its subclasses.
    /// Invalidated by any method table change for `method`, anywhere.
    NoMethodOverride {
        /// The class rooting the hierarchy we assume nothing overrides `method` in
        klass: VALUE,
        /// The method ID whose lookup we want to assume unchanged below `klass`
        method: ID,
        /// The callable method entry every receiver below `klass` resolves to
        cme: *const rb_callable_method_entry_t,
    },
    /// A list of constant expression path segments that must have not been written to for the
    /// following code to be valid.
    StableConstantNames {
        idlist: *const ID,
    },
    /// TracePoint is not enabled. If TracePoint is enabled, this is invalidated.
    NoTracePoint,
    /// No NEWOBJ internal event hook is active. The inline allocation fast path
    /// bypasses rb_newobj, so it can't fire NEWOBJ; this is invalidated when such
    /// a hook is enabled.
    NoNewObjHook,
    /// cfp->ep is not escaped to the heap on the ISEQ
    NoEPEscape(IseqPtr),
    /// There is one ractor running. If a non-root ractor gets spawned, this is invalidated.
    SingleRactorMode,
    /// Objects of this class have no singleton class.
    /// When a singleton class is created for an object of this class, this is invalidated.
    NoSingletonClass {
        klass: VALUE,
    },
    /// Only the root box is active, so we can safely read from the prime classext.
    /// Invalidated if a non-root box duplicates any classext.
    RootBoxOnly,
}

impl Invariant {
    pub fn print(self, ptr_map: &PtrPrintMap) -> InvariantPrinter<'_> {
        InvariantPrinter { inner: self, ptr_map }
    }
}

impl Display for Invariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.print(&PtrPrintMap::identity()).fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecialObjectType {
    VMCore = 1,
    CBase = 2,
    ConstBase = 3,
}

impl From<u32> for SpecialObjectType {
    fn from(value: u32) -> Self {
        match value {
            VM_SPECIAL_OBJECT_VMCORE => SpecialObjectType::VMCore,
            VM_SPECIAL_OBJECT_CBASE => SpecialObjectType::CBase,
            VM_SPECIAL_OBJECT_CONST_BASE => SpecialObjectType::ConstBase,
            _ => panic!("Invalid special object type: {value}"),
        }
    }
}

impl From<SpecialObjectType> for u64 {
    fn from(special_type: SpecialObjectType) -> Self {
        special_type as u64
    }
}

impl std::fmt::Display for SpecialObjectType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SpecialObjectType::VMCore => write!(f, "VMCore"),
            SpecialObjectType::CBase => write!(f, "CBase"),
            SpecialObjectType::ConstBase => write!(f, "ConstBase"),
        }
    }
}

/// Print adaptor for [`Invariant`]. See [`PtrPrintMap`].
pub struct InvariantPrinter<'a> {
    inner: Invariant,
    ptr_map: &'a PtrPrintMap,
}

impl<'a> std::fmt::Display for InvariantPrinter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.inner {
            Invariant::BOPRedefined { klass, bop } => {
                write!(f, "BOPRedefined(")?;
                match klass {
                    INTEGER_REDEFINED_OP_FLAG => write!(f, "INTEGER_REDEFINED_OP_FLAG")?,
                    STRING_REDEFINED_OP_FLAG => write!(f, "STRING_REDEFINED_OP_FLAG")?,
                    ARRAY_REDEFINED_OP_FLAG => write!(f, "ARRAY_REDEFINED_OP_FLAG")?,
                    HASH_REDEFINED_OP_FLAG => write!(f, "HASH_REDEFINED_OP_FLAG")?,
                    _ => write!(f, "{klass}")?,
                }
                write!(f, ", ")?;
                match bop {
                    BOP_PLUS     => write!(f, "BOP_PLUS")?,
                    BOP_MINUS    => write!(f, "BOP_MINUS")?,
                    BOP_MULT     => write!(f, "BOP_MULT")?,
                    BOP_DIV      => write!(f, "BOP_DIV")?,
                    BOP_MOD      => write!(f, "BOP_MOD")?,
                    BOP_EQ       => write!(f, "BOP_EQ")?,
                    BOP_EQQ      => write!(f, "BOP_EQQ")?,
                    BOP_LT       => write!(f, "BOP_LT")?,
                    BOP_LE       => write!(f, "BOP_LE")?,
                    BOP_LTLT     => write!(f, "BOP_LTLT")?,
                    BOP_AREF     => write!(f, "BOP_AREF")?,
                    BOP_ASET     => write!(f, "BOP_ASET")?,
                    BOP_LENGTH   => write!(f, "BOP_LENGTH")?,
                    BOP_SIZE     => write!(f, "BOP_SIZE")?,
                    BOP_EMPTY_P  => write!(f, "BOP_EMPTY_P")?,
                    BOP_NIL_P    => write!(f, "BOP_NIL_P")?,
                    BOP_SUCC     => write!(f, "BOP_SUCC")?,
                    BOP_GT       => write!(f, "BOP_GT")?,
                    BOP_GE       => write!(f, "BOP_GE")?,
                    BOP_NOT      => write!(f, "BOP_NOT")?,
                    BOP_NEQ      => write!(f, "BOP_NEQ")?,
                    BOP_MATCH    => write!(f, "BOP_MATCH")?,
                    BOP_FREEZE   => write!(f, "BOP_FREEZE")?,
                    BOP_UMINUS   => write!(f, "BOP_UMINUS")?,
                    BOP_MAX      => write!(f, "BOP_MAX")?,
                    BOP_MIN      => write!(f, "BOP_MIN")?,
                    BOP_HASH     => write!(f, "BOP_HASH")?,
                    BOP_CALL     => write!(f, "BOP_CALL")?,
                    BOP_AND      => write!(f, "BOP_AND")?,
                    BOP_OR       => write!(f, "BOP_OR")?,
                    BOP_CMP      => write!(f, "BOP_CMP")?,
                    BOP_DEFAULT  => write!(f, "BOP_DEFAULT")?,
                    BOP_PACK     => write!(f, "BOP_PACK")?,
                    BOP_INCLUDE_P => write!(f, "BOP_INCLUDE_P")?,
                    _ => write!(f, "{bop}")?,
                }
                write!(f, ")")
            }
            Invariant::MethodRedefined { klass, method, cme } => {
                let class_name = get_class_name(klass);
                write!(f, "MethodRedefined({class_name}@{:p}, {}@{:p}, cme:{:p})",
                    self.ptr_map.map_ptr(klass.as_ptr::<VALUE>()),
                    method.contents_lossy(),
                    self.ptr_map.map_id(method.0),
                    self.ptr_map.map_ptr(cme)
                )
            }
            Invariant::NoMethodOverride { klass, method, cme } => {
                let class_name = get_class_name(klass);
                write!(f, "NoMethodOverride({class_name}@{:p}, {}@{:p}, cme:{:p})",
                    self.ptr_map.map_ptr(klass.as_ptr::<VALUE>()),
                    method.contents_lossy(),
                    self.ptr_map.map_id(method.0),
                    self.ptr_map.map_ptr(cme)
                )
            }
            Invariant::StableConstantNames { idlist } => {
                write!(f, "StableConstantNames({:p}, ", self.ptr_map.map_ptr(idlist))?;
                let mut idx = 0;
                let mut sep = "";
                loop {
                    let id = unsafe { *idlist.wrapping_add(idx) };
                    if id.0 == 0 {
                        break;
                    }
                    write!(f, "{sep}{}", id.contents_lossy())?;
                    sep = "::";
                    idx += 1;
                }
                write!(f, ")")
            }
            Invariant::NoTracePoint => write!(f, "NoTracePoint"),
            Invariant::NoNewObjHook => write!(f, "NoNewObjHook"),
            Invariant::NoEPEscape(iseq) => write!(f, "NoEPEscape({})", &iseq_name(iseq)),
            Invariant::SingleRactorMode => write!(f, "SingleRactorMode"),
            Invariant::NoSingletonClass { klass } => {
                let class_name = get_class_name(klass);
                write!(f, "NoSingletonClass({}@{:p})",
                    class_name,
                    self.ptr_map.map_ptr(klass.as_ptr::<VALUE>()))
            }
            Invariant::RootBoxOnly => write!(f, "RootBoxOnly"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Const {
    Value(VALUE),
    CBool(bool),
    CInt8(i8),
    CInt16(i16),
    CInt32(i32),
    CInt64(i64),
    CUInt8(u8),
    CUInt16(u16),
    CUInt32(u32),
    CAttrIndex(attr_index_t),
    CShape(ShapeId),
    CUInt64(u64),
    CPtr(*const u8),
    CDouble(f64),
}

impl std::fmt::Display for Const {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.print(&PtrPrintMap::identity()).fmt(f)
    }
}

impl Const {
    pub fn print<'a>(&'a self, ptr_map: &'a PtrPrintMap) -> ConstPrinter<'a> {
        ConstPrinter { inner: self, ptr_map }
    }
}

#[derive(Clone, Copy)]
pub enum RangeType {
    Inclusive = 0, // include the end value
    Exclusive = 1, // exclude the end value
}

impl std::fmt::Display for RangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", match self {
            RangeType::Inclusive => "NewRangeInclusive",
            RangeType::Exclusive => "NewRangeExclusive",
        })
    }
}

impl std::fmt::Debug for RangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl From<u32> for RangeType {
    fn from(flag: u32) -> Self {
        match flag {
            0 => RangeType::Inclusive,
            1 => RangeType::Exclusive,
            _ => panic!("Invalid range flag: {flag}"),
        }
    }
}

/// Special regex backref symbol types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecialBackrefSymbol {
    LastMatch,     // $&
    PreMatch,      // $`
    PostMatch,     // $'
    LastGroup,     // $+
}

impl TryFrom<u8> for SpecialBackrefSymbol {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value as char {
            '&' => Ok(SpecialBackrefSymbol::LastMatch),
            '`' => Ok(SpecialBackrefSymbol::PreMatch),
            '\'' => Ok(SpecialBackrefSymbol::PostMatch),
            '+' => Ok(SpecialBackrefSymbol::LastGroup),
            c => Err(format!("invalid backref symbol: '{c}'")),
        }
    }
}

/// Print adaptor for [`Const`]. See [`PtrPrintMap`].
pub struct ConstPrinter<'a> {
    inner: &'a Const,
    ptr_map: &'a PtrPrintMap,
}

impl<'a> std::fmt::Display for ConstPrinter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.inner {
            Const::Value(val) => write!(f, "Value({})", val.print(self.ptr_map)),
            // Since `&` coerces to a raw pointer, be careful to get `val` and not `&val` here.
            &Const::CPtr(val) => write!(f, "CPtr({:p})", self.ptr_map.map_ptr(val)),
            &Const::CShape(shape_id) => write!(f, "CShape({:p})", self.ptr_map.map_shape(shape_id)),
            &Const::CUInt64(int) => {
                // Print in hex if signed bit is set
                if 0 != int & (1 << (u64::BITS - 1)) {
                    write!(f, "CUInt64(0x{int:x})")
                } else {
                    write!(f, "CUInt64({int})")
                }
            }
            _ => write!(f, "{:?}", self.inner),
        }
    }
}

/// For output stability in tests, we assign each pointer with a stable
/// address the first time we see it. This mapping is off by default;
/// set [`PtrPrintMap::map_ptrs`] to switch it on.
///
/// Because this is extra state external to any pointer being printed, a
/// printing adapter struct that wraps the pointer along with this map is
/// required to make use of this effectively. The [`std::fmt::Display`]
/// implementation on the adapter struct can then be reused to implement
/// `Display` on the inner type with a default [`PtrPrintMap`], which
/// does not perform any mapping.
pub struct PtrPrintMap {
    inner: RefCell<PtrPrintMapInner>,
    map_ptrs: bool,
}

struct PtrPrintMapInner {
    map: HashMap<*const c_void, *const c_void>,
    next_ptr: *const c_void,
}

impl PtrPrintMap {
    /// Return a mapper that maps the pointer to itself.
    pub fn identity() -> Self {
        Self {
            map_ptrs: false,
            inner: RefCell::new(PtrPrintMapInner {
                map: HashMap::default(), next_ptr:
                ptr::without_provenance(0x1000) // Simulate 4 KiB zero page
            })
        }
    }
}

struct Offset(i32);

impl std::fmt::LowerHex for Offset {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let prefix = if f.alternate() { "0x" } else { "" };
        let bare_hex = format!("{:x}", self.0.abs());
        f.pad_integral(self.0 >= 0, prefix, &bare_hex)
    }
}

/// A trait tailored for [`PtrPrintMap`] to disable coercion of `&*const T` into `*const *const T`.
/// This is implemented for `*const/mut T`, but rules for coercing into `impl Trait` don't consider the
/// underlying type, so we avoid the undesirable coercion. (It would be weird for the treatment of a
/// trait to change based on the set of types that implements it since Rust has a nominal type system.)
pub trait OneLevelPtr: Copy {
    /// Get the address component of the pointer.
    fn addr(self) -> usize;
    /// The layout of the pointed-to type.
    fn pointee_layout(self) -> std::alloc::Layout;
}

impl<T> OneLevelPtr for *const T {
    fn addr(self) -> usize {
        <*const T>::addr(self)
    }

    fn pointee_layout(self) -> std::alloc::Layout {
        std::alloc::Layout::new::<T>()
    }
}

impl<T> OneLevelPtr for *mut T {
    fn addr(self) -> usize {
        <*mut T>::addr(self)
    }

    fn pointee_layout(self) -> std::alloc::Layout {
        std::alloc::Layout::new::<T>()
    }
}

impl PtrPrintMap {
    /// Map a pointer for printing.
    ///
    /// The type bound on this function rejects `&*const T`, which we commonly get through matching:
    ///
    /// ```compile_fail
    /// let value = 0;
    /// let ptr: *const usize = &value;
    /// let ref_to_ptr: &*const usize = &ptr;
    ///
    /// let map = zjit::hir::PtrPrintMap::identity();
    /// // error[E0277]: the trait bound `&*const usize: OneLevelPtr` is not satisfied
    /// map.map_ptr(ref_to_ptr);
    /// ```
    pub fn map_ptr(&self, ptr: impl OneLevelPtr) -> *const c_void {
        let raw = ptr::without_provenance(ptr.addr());
        if !self.map_ptrs {
            return raw
        }

        use std::collections::hash_map::Entry::*;
        let inner = &mut *self.inner.borrow_mut();
        match inner.map.entry(raw) {
            Occupied(entry) => *entry.get(),
            Vacant(entry) => {
                // Pick a fake address that is suitably aligned for the pointee and
                // remember it in the map
                let layout = ptr.pointee_layout();
                let mapped = inner.next_ptr.wrapping_add(inner.next_ptr.align_offset(layout.align()));
                entry.insert(mapped);

                // Bump for the next pointer
                inner.next_ptr = mapped.wrapping_add(layout.size());
                mapped
            }
        }
    }

    /// Map a Ruby ID (index into intern table) for printing
    fn map_id(&self, id: u64) -> *const c_void {
        self.map_ptr(id as *const c_void)
    }

    /// Map an index into a Ruby object (e.g. for an ivar) for printing
    fn map_index(&self, id: u64) -> *const c_void {
        self.map_ptr(id as *const c_void)
    }

    fn map_offset(&self, id: i32) -> Offset {
        Offset(self.map_ptr(id as *const c_void) as i32)
    }

    /// Map shape ID into a pointer for printing
    pub fn map_shape(&self, id: ShapeId) -> *const c_void {
        self.map_ptr(id.0 as *const c_void)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SideExitReason {
    UnhandledNewarraySend(vm_opt_newarray_send_type),
    UnhandledDuparraySend(u64),
    UnknownSpecialVariable(u64),
    UnhandledHIRThrow,
    UnhandledHIRInvokeBuiltin,
    UnhandledHIRUnknown(InsnId),
    UnhandledYARVInsn(u32),
    UnhandledCallType(CallType),
    UnhandledBlockArg,
    BlockArgNotNil,
    TooManyKeywordParameters,
    FixnumAddOverflow,
    FixnumSubOverflow,
    FixnumMultOverflow,
    FixnumLShiftOverflow,
    GuardType(Type),
    GuardShape(ShapeId),
    ExpandArray,
    /// `expandarray` was never profiled, so we don't know what shape to compile it for.
    NoProfileExpandArray,
    GuardNotFrozen,
    GuardNotShared,
    GuardNotDependant,
    GuardLess,
    GuardGreaterEq,
    GuardSuperMethodEntry,
    PatchPoint(Invariant),
    CalleeSideExit,
    Interrupt,
    BlockParamProxyNotIseqOrIfunc,
    BlockParamProxyNotNil,
    BlockParamProxyNotProc,
    BlockParamProxyFallbackMiss,
    BlockParamProxyProfileNotCovered,
    InvokeBlockHandlerNotIseq,
    InvokeBlockIseqChanged,
    BlockParamWbRequired,
    StackOverflow,
    FixnumModByZero,
    FixnumDivByZero,
    BoxFixnumOverflow,
    SplatKwNotNilOrHash,
    SplatKwPolymorphic,
    SplatKwNotProfiled,
    SplatLengthChanged,
    SplatLastRuby2Keywords,
    DirectiveInduced,
    SendWhileTracing,
    NoProfileSend,
    NoProfileGetIvar,
    NoProfileSetIvar,
    InvokeBlockNotIfunc,
}

/// Marks a side exit as triggering profiling and recompilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Recompile;

#[derive(Debug, Clone, Copy)]
pub enum MethodType {
    Iseq,
    Cfunc,
    Attrset,
    Ivar,
    Bmethod,
    Zsuper,
    Alias,
    Undefined,
    NotImplemented,
    Optimized,
    Missing,
    Refined,
    Null,
}

impl From<u32> for MethodType {
    fn from(value: u32) -> Self {
        match value {
            VM_METHOD_TYPE_ISEQ => MethodType::Iseq,
            VM_METHOD_TYPE_CFUNC => MethodType::Cfunc,
            VM_METHOD_TYPE_ATTRSET => MethodType::Attrset,
            VM_METHOD_TYPE_IVAR => MethodType::Ivar,
            VM_METHOD_TYPE_BMETHOD => MethodType::Bmethod,
            VM_METHOD_TYPE_ZSUPER => MethodType::Zsuper,
            VM_METHOD_TYPE_ALIAS => MethodType::Alias,
            VM_METHOD_TYPE_UNDEF => MethodType::Undefined,
            VM_METHOD_TYPE_NOTIMPLEMENTED => MethodType::NotImplemented,
            VM_METHOD_TYPE_OPTIMIZED => MethodType::Optimized,
            VM_METHOD_TYPE_MISSING => MethodType::Missing,
            VM_METHOD_TYPE_REFINED => MethodType::Refined,
            _ => unreachable!("unknown send_without_block def_type: {}", value),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizedMethodType {
    Send,
    Call,
    BlockCall,
    StructAref,
    StructAset,
}

impl From<u32> for OptimizedMethodType {
    fn from(value: u32) -> Self {
        match value {
            OPTIMIZED_METHOD_TYPE_SEND => OptimizedMethodType::Send,
            OPTIMIZED_METHOD_TYPE_CALL => OptimizedMethodType::Call,
            OPTIMIZED_METHOD_TYPE_BLOCK_CALL => OptimizedMethodType::BlockCall,
            OPTIMIZED_METHOD_TYPE_STRUCT_AREF => OptimizedMethodType::StructAref,
            OPTIMIZED_METHOD_TYPE_STRUCT_ASET => OptimizedMethodType::StructAset,
            _ => unreachable!("unknown send_without_block optimized method type: {}", value),
        }
    }
}

impl std::fmt::Display for SideExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SideExitReason::UnhandledYARVInsn(opcode) => write!(f, "UnhandledYARVInsn({})", insn_name(*opcode as usize)),
            SideExitReason::UnhandledNewarraySend(VM_OPT_NEWARRAY_SEND_MAX) => write!(f, "UnhandledNewarraySend(MAX)"),
            SideExitReason::UnhandledNewarraySend(VM_OPT_NEWARRAY_SEND_MIN) => write!(f, "UnhandledNewarraySend(MIN)"),
            SideExitReason::UnhandledNewarraySend(VM_OPT_NEWARRAY_SEND_HASH) => write!(f, "UnhandledNewarraySend(HASH)"),
            SideExitReason::UnhandledNewarraySend(VM_OPT_NEWARRAY_SEND_PACK) => write!(f, "UnhandledNewarraySend(PACK)"),
            SideExitReason::UnhandledNewarraySend(VM_OPT_NEWARRAY_SEND_PACK_BUFFER) => write!(f, "UnhandledNewarraySend(PACK_BUFFER)"),
            SideExitReason::UnhandledNewarraySend(VM_OPT_NEWARRAY_SEND_INCLUDE_P) => write!(f, "UnhandledNewarraySend(INCLUDE_P)"),
            SideExitReason::UnhandledDuparraySend(method_id) => write!(f, "UnhandledDuparraySend({method_id})"),
            SideExitReason::GuardType(guard_type) => write!(f, "GuardType({guard_type})"),
            SideExitReason::GuardNotShared => write!(f, "GuardNotShared"),
            SideExitReason::PatchPoint(invariant) => write!(f, "PatchPoint({invariant})"),
            _ => write!(f, "{self:?}"),
        }
    }
}

/// Result of resolving the receiver type for method dispatch optimization.
/// Represents whether we know the receiver's class statically at compile-time,
/// have profiled type information, or know nothing about it.
pub enum ReceiverTypeResolution {
    /// No profile information available for the receiver
    NoProfile,
    /// The receiver has a monomorphic profile (single type observed, guard needed)
    Monomorphic { profiled_type: ProfiledType },
    /// The receiver is polymorphic (multiple types, none dominant)
    Polymorphic,
    /// The receiver has a skewed polymorphic profile (dominant type with some other types, guard needed)
    SkewedPolymorphic { profiled_type: ProfiledType },
    /// More than N types seen with no clear winner
    Megamorphic,
    /// Megamorphic, but with a significant skew towards one type
    SkewedMegamorphic { profiled_type: ProfiledType },
    /// The receiver's class is statically known at JIT compile-time (no guard needed)
    StaticallyKnown { class: VALUE },
}

/// Reason why a send-ish instruction cannot be optimized from a fallback instruction
#[derive(Debug, Clone, Copy)]
pub enum SendFallbackReason {
    SendCfuncNotVariadic,
    SendNotOptimizedMethodTypeOptimized(OptimizedMethodType),
    SendBopRedefined,
    SendOperandsNotFixnum,
    SendPolymorphicFallback,
    SendAncestorGuardFallback,
    SendDirectKeywordMismatch,
    SendDirectKeywordCountMismatch,
    SendDirectMissingKeyword,
    SendDirectTooManyKeywords,
    SendPolymorphic,
    SendMegamorphic,
    SendNoProfiles,
    SendCfuncVariadic,
    SendCfuncArrayVariadic,
    SendNotOptimizedMethodType(MethodType),
    SendNotOptimizedNeedPermission,
    /// The block argument is not nil, so we can't optimize to SendWithoutBlockDirect
    SendBlockArgNotNil,
    CCallWithFrameTooManyArgs,
    ObjToStringNotString,
    /// Too many arguments in a C call to fit in C ABI registers.
    TooManyArgsForLir,
    /// An operand doesn't fit in the integer type that encodes it,
    /// e.g. an argument count that overflows IseqCall's u16.
    OperandTooLarge,
    /// The Proc object for a BMETHOD is not defined by an ISEQ. (See `enum rb_block_type`.)
    BmethodNonIseqProc,
    /// Caller supplies too few or too many arguments than what the callee's parameters expects.
    ArgcParamMismatch,
    /// The call has at least one feature on the caller or callee side that the optimizer does not
    /// support.
    ComplexArgPass,
    /// Caller has keyword arguments but callee doesn't expect them; need to convert to hash.
    UnexpectedKeywordArgs,
    /// A singleton class has been seen for the receiver class, so we skip the optimization
    /// to avoid an invalidation loop.
    SingletonClassSeen,
    /// The super call is passed a block that the optimizer does not support.
    SuperCallWithBlock,
    /// When the `super` is in a block, finding the running CME for guarding requires a loop. Not
    /// supported for now.
    SuperFromBlock,
    /// The profiled super class cannot be found.
    SuperClassNotFound,
    /// The `super` call uses a complex argument pattern that the optimizer does not support.
    SuperComplexArgsPass,
    /// The cached target of a `super` call could not be found.
    SuperTargetNotFound,
    /// Attempted to specialize a `super` call that doesn't have profile data.
    SuperNoProfiles,
    /// Cannot optimize the `super` call due to the target method.
    SuperNotOptimizedMethodType(MethodType),
    /// The `super` call is polymorpic.
    SuperPolymorphic,
    /// A previous version of this ISEQ guarded the frame's method entry at this `super` and the
    /// guard kept missing, so this version dispatches `super` dynamically instead of exiting.
    SuperMethodEntryUnstable,
    /// The `invokeblock` instruction is not yet optimized in `type_specialize`.
    InvokeBlockNotSpecialized,
    /// The `invokeblock` call site passes a splat, keyword, or block argument.
    InvokeBlockComplexArgs,
    /// The `invokeblock` site has no block-handler profile to specialize on.
    InvokeBlockNoProfile,
    /// The profiled block handler is not an ISEQ block (Proc, IFUNC, or symbol).
    InvokeBlockHandlerNotIseqProfile,
    /// The `invokeblock` site saw too many distinct block handlers to build a chain over.
    InvokeBlockMegamorphicProfile,
    /// The profiled block ISEQs that can dispatch directly do not cover enough of the site's
    /// profile to pay for the comparison chain.
    InvokeBlockChainCoverage,
    /// The profiled block ISEQ takes optional, rest, post, keyword, or block parameters, so it
    /// cannot use the simple callee setup the direct dispatch relies on.
    InvokeBlockNotSimpleIseq,
    /// A one-argument `yield` to a `|x,|` block, which auto-splats and then truncates.
    InvokeBlockAmbiguousParam0,
    /// The profiled block ISEQ contains a `throw` that is not a plain non-local `return`
    /// (`break`, `redo`, or `next` out of a rescue).
    InvokeBlockMayThrow,
    /// The runtime block handler at a polymorphic `invokeblock` site did not match any
    /// profiled ISEQ candidate, so the site dispatched through the generic fallback.
    InvokeBlockPolymorphicMiss,
    /// A one-argument `yield` to a block that takes several parameters auto-splats. The
    /// yielded value was not an Array of exactly that many elements, so the site dispatched
    /// through the generic fallback instead of the expanded direct call.
    InvokeBlockAutosplatMiss,
    /// The `sendforward` instruction (argument forwarding `...`) is not yet optimized in
    /// `type_specialize`.
    SendForwardNotSpecialized,
    /// The `invokesuperforward` instruction (super with forwarding `...`) is not yet optimized in
    /// `type_specialize`.
    InvokeSuperForwardNotSpecialized,
    /// The single-ractor-mode assumption could not be made.
    SingleRactorModeRequired,
    /// A `send`/`__send__` call site whose method-name argument did not match any of the
    /// names the profiler observed there.
    SendUnprofiledMethodName,
    /// Initial fallback reason for every instruction, which should be mutated to
    /// a more actionable reason when an attempt to specialize the instruction fails.
    Uncategorized(VmInsnType),
}

impl Display for SendFallbackReason {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SendCfuncNotVariadic => write!(f, "Send: C function is not variadic"),
            SendNotOptimizedMethodTypeOptimized(opt_type) => write!(f, "Send: unsupported optimized method type {:?}", opt_type),
            SendNotOptimizedNeedPermission => write!(f, "Send: method private or protected and no FCALL"),
            SendBopRedefined => write!(f, "Send: basic operation was redefined"),
            SendOperandsNotFixnum => write!(f, "Send: operands are not fixnums"),
            SendPolymorphicFallback => write!(f, "Send: polymorphic fallback"),
            SendAncestorGuardFallback => write!(f, "Send: ancestor guard fallback"),
            SendDirectKeywordMismatch => write!(f, "SendDirect: keyword mismatch"),
            SendDirectKeywordCountMismatch => write!(f, "SendDirect: keyword count mismatch"),
            SendDirectMissingKeyword => write!(f, "SendDirect: missing keyword"),
            SendDirectTooManyKeywords => write!(f, "SendDirect: too many keywords for fixnum bitmask"),
            SendPolymorphic => write!(f, "Send: polymorphic call site"),
            SendMegamorphic => write!(f, "Send: megamorphic call site"),
            SendNoProfiles => write!(f, "Send: no profile data available"),
            SendCfuncVariadic => write!(f, "Send: C function is variadic"),
            SendCfuncArrayVariadic => write!(f, "Send: C function expects array variadic"),
            SendNotOptimizedMethodType(method_type) => write!(f, "Send: unsupported method type {:?}", method_type),
            SendBlockArgNotNil => write!(f, "Send: block argument is not nil"),
            CCallWithFrameTooManyArgs => write!(f, "CCallWithFrame: too many arguments"),
            ObjToStringNotString => write!(f, "ObjToString: result is not a string"),
            TooManyArgsForLir => write!(f, "Too many arguments for LIR"),
            OperandTooLarge => write!(f, "Operand doesn't fit in its encoding"),
            BmethodNonIseqProc => write!(f, "Bmethod: Proc object is not defined by an ISEQ"),
            ArgcParamMismatch => write!(f, "Argument count does not match parameter count"),
            ComplexArgPass => write!(f, "Complex argument passing"),
            UnexpectedKeywordArgs => write!(f, "Unexpected Keyword Args"),
            SingletonClassSeen => write!(f, "Singleton class previously created for receiver class"),
            SuperFromBlock => write!(f, "super: call from within a block"),
            SuperCallWithBlock => write!(f, "super: call made with a block"),
            SuperClassNotFound => write!(f, "super: profiled class cannot be found"),
            SuperComplexArgsPass => write!(f, "super: complex argument passing to `super` call"),
            SuperNoProfiles => write!(f, "super: no profile data available"),
            SuperNotOptimizedMethodType(method_type) => write!(f, "super: unsupported target method type {:?}", method_type),
            SuperPolymorphic => write!(f, "super: polymorphic call site"),
            SuperMethodEntryUnstable => write!(f, "super: frame method entry guard kept missing"),
            SuperTargetNotFound => write!(f, "super: profiled target method cannot be found"),
            InvokeBlockNotSpecialized => write!(f, "InvokeBlock: not yet specialized"),
            InvokeBlockComplexArgs => write!(f, "InvokeBlock: splat, keyword, or block argument"),
            InvokeBlockNoProfile => write!(f, "InvokeBlock: no block handler profile"),
            InvokeBlockHandlerNotIseqProfile => write!(f, "InvokeBlock: profiled handler is not an ISEQ block"),
            InvokeBlockMegamorphicProfile => write!(f, "InvokeBlock: megamorphic block handler profile"),
            InvokeBlockChainCoverage => write!(f, "InvokeBlock: dispatchable ISEQs cover too little of the profile"),
            InvokeBlockNotSimpleIseq => write!(f, "InvokeBlock: block takes non-lead parameters"),
            InvokeBlockAmbiguousParam0 => write!(f, "InvokeBlock: |x,| block truncates an auto-splat"),
            InvokeBlockMayThrow => write!(f, "InvokeBlock: block contains a non-return throw"),
            InvokeBlockPolymorphicMiss => write!(f, "InvokeBlock: polymorphic dispatch miss"),
            InvokeBlockAutosplatMiss => write!(f, "InvokeBlock: auto-splat expansion miss"),
            SendForwardNotSpecialized => write!(f, "SendForward: not yet specialized"),
            InvokeSuperForwardNotSpecialized => write!(f, "InvokeSuperForward: not yet specialized"),
            SingleRactorModeRequired => write!(f, "Single-ractor mode required"),
            SendUnprofiledMethodName => write!(f, "send: method name not seen while profiling"),
            Uncategorized(insn) => write!(f, "Uncategorized({})", insn_name(insn.to_usize())),
        }
    }
}

/// How a block is passed to a send-like instruction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockHandler {
    /// Literal block ISEQ (e.g. `foo { ... }`)
    BlockIseq(IseqPtr),
    /// Block arg passed via &proc (e.g. `foo(&block)`)
    BlockArg,
}

/// Identifier used by LoadField/StoreField/LoadArg for HIR dumps. Variants
/// without an associated value name internal VM fields that we used to intern
/// as CRuby IDs just to print them; the `Id` variant carries a real CRuby ID
/// (e.g. local variable, ivar, struct field name).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldName {
    VM_ENV_DATA_INDEX_ME_CREF,
    VM_ENV_DATA_INDEX_SPECVAL,
    VM_ENV_DATA_INDEX_FLAGS,
    RBASIC_FLAGS,
    code_iseq,
    shape_id,
    as_heap,
    fields_obj,
    thread_ptr,
    len,
    SelfParam,
    /// A VM stack slot, read back at an exception handler entry
    StackSlot,
    Id(ID),
}

impl std::fmt::Display for FieldName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use FieldName::*;
        match self {
            Id(id) if id_is_empty(*id) => f.write_str("<empty>"),
            Id(id) => f.write_str(&id.contents_lossy()),
            SelfParam => f.write_str("self"),
            _ => write!(f, "{self:?}"),
        }
    }
}

impl From<ID> for FieldName {
    fn from(id: ID) -> Self {
        FieldName::Id(id)
    }
}

/// Payload of [`Insn::CCallWithFrame`]. Boxed in the enum to keep `Insn` small.
#[derive(Debug, Clone)]
pub struct CCallWithFrameData {
    pub cd: *const rb_call_data, // cd for falling back to Send
    pub cfunc: *const u8,
    pub recv: InsnId,
    pub args: Vec<InsnId>,
    pub cme: *const rb_callable_method_entry_t,
    pub name: ID,
    pub state: InsnId,
    pub return_type: Type,
    pub elidable: bool,
    pub block: Option<BlockHandler>,
    /// See [`SendDirectData::block_arg`]. Unlike the ISEQ case this is *not* taken off the VM
    /// stack: the frame the C method runs in is pushed over the argument slots, so the frame
    /// setup counts the block argument's slot even though the C function never sees it.
    pub block_arg: Option<InsnId>,
}

/// Payload of [`Insn::SendDirect`]. Boxed in the enum to keep `Insn` small.
#[derive(Debug, Clone)]
pub struct SendDirectData {
    pub recv: InsnId,
    pub cd: *const rb_call_data,
    pub cme: *const rb_callable_method_entry_t,
    pub iseq: IseqPtr,
    pub args: Vec<InsnId>,
    pub kw_bits: u32,
    pub jit_entry_idx: u16,
    pub block: Option<BlockHandler>,
    /// `foo(&blk)` reduced to the block handler `vm_caller_setup_arg_block` would have
    /// produced: the Proc itself when `blk` is one, or this frame's own block handler when
    /// `blk` is the block param proxy. Mutually exclusive with a literal block in `block`,
    /// and kept out of `args` because it is not one of the callee's parameters.
    pub block_arg: Option<InsnId>,
    pub state: InsnId,
    /// The interpreter's own state at the call site, which is what code inlined in place of
    /// this call must side-exit to. It differs from `state` -- the state that describes the
    /// callee frame -- whenever the caller's stack was rewritten for the frame setup: a
    /// stripped nil block arg, an expanded splat, or a `send`/`__send__` name argument
    /// dropped by [`Function::send_mid_overrides`]. Code that runs without pushing the
    /// callee frame re-runs the original call instruction when it exits, so it needs the
    /// original stack.
    pub guard_state: InsnId,
}

/// Payload of [`Insn::CCallVariadic`]. Boxed in the enum to keep `Insn` small.
#[derive(Debug, Clone)]
pub struct CCallVariadicData {
    pub cfunc: *const u8,
    pub recv: InsnId,
    pub args: Vec<InsnId>,
    pub cme: *const rb_callable_method_entry_t,
    pub name: ID,
    pub state: InsnId,
    pub return_type: Type,
    pub elidable: bool,
    pub block: Option<BlockHandler>,
    /// See [`CCallWithFrameData::block_arg`].
    pub block_arg: Option<InsnId>,
}

/// An instruction in the SSA IR. The output of an instruction is referred to by the index of
/// the instruction ([`InsnId`]). SSA form enables this, and [`UnionFind`] ([`Function::find`])
/// helps with editing.
#[derive(Debug, Clone)]
pub enum Insn {
    /// Comment that can be inserted into HIR for diagnostics.
    Comment { message: String },

    Const { val: Const },
    /// SSA block parameter. Also used for function parameters in the function's entry block.
    Param,
    /// Load a function argument from the calling convention.
    /// Used in JIT entry blocks. idx is the calling convention index, id is for display.
    LoadArg { idx: u32, id: FieldName, val_type: Type },

    /// Synthetic terminator for the entries superblock. Targets all entry blocks
    /// so that CFG analyses see a single root. Not lowered to machine code.
    Entries { targets: Vec<BlockId> },

    StringCopy { val: InsnId, chilled: bool, state: InsnId },
    StringIntern { val: InsnId, state: InsnId },
    StringConcat { strings: Vec<InsnId>, state: InsnId },
    /// Call rb_str_getbyte with known-Fixnum index
    StringGetbyte { string: InsnId, index: InsnId },
    /// Return the coderange of `string`, scanning the string to compute and cache it when the
    /// cached value is [`RUBY_ENC_CODERANGE_UNKNOWN`]. `cached` is the coderange bits already
    /// loaded out of RBASIC flags; only the UNKNOWN case reaches the scan, which is what
    /// `String#ascii_only?` and friends do, so there is no reason to leave the JIT for it.
    StringCoderangeOrScan { string: InsnId, cached: InsnId, state: InsnId },
    /// Write the low byte of the Fixnum `value` into `string`'s buffer at the already
    /// bounds-checked C integer `index`, clearing the cached coderange. The string must be
    /// guarded with `guard_string_not_dependant` so that writing in place is possible.
    StringSetbyteFixnum { string: InsnId, index: InsnId, value: InsnId },
    StringAppend { recv: InsnId, other: InsnId, state: InsnId },
    StringAppendCodepoint { recv: InsnId, other: InsnId, state: InsnId },
    StringEqual { left: InsnId, right: InsnId },

    /// Combine count stack values into a regexp
    ToRegexp { opt: usize, values: Vec<InsnId>, state: InsnId },

    /// Put special object (VMCORE, CBASE, etc.) based on value_type
    PutSpecialObject { value_type: SpecialObjectType, state: InsnId },

    /// The generic form of the `splatkw` YARV instruction: pass `nil` through, and otherwise
    /// convert `val` to a Hash with `to_hash`. Used when the profile does not let us pick one
    /// of the two shapes up front.
    ToHash { val: InsnId, state: InsnId },
    /// Call `to_a` on `val` if the method is defined, or make a new array `[val]` otherwise.
    ToArray { val: InsnId, state: InsnId },
    /// Convert `val` to an Array by calling `#to_ary` on it, or return `nil` if it cannot be
    /// converted. Returns `val` itself if it is already an `Array`. Mirrors
    /// `rb_check_array_type()`; can run arbitrary Ruby code because of `#to_ary`.
    CheckArrayType { val: InsnId, state: InsnId },
    /// Convert `val` the way `vm_expandarray()` does: `#to_ary` if that is defined, and the
    /// one-element array `[val]` otherwise. Always returns an `Array`. Mirrors `rb_ary_to_ary()`;
    /// can run arbitrary Ruby code because of `#to_ary`.
    ToAryForExpand { val: InsnId, state: InsnId },
    /// Call `to_a` on `val` if the method is defined, or make a new array `[val]` otherwise. If we
    /// called `to_a`, duplicate the returned array.
    ToNewArray { val: InsnId, state: InsnId },
    NewArray { elements: Vec<InsnId>, state: InsnId },
    /// NewHash contains a vec of (key, value) pairs
    NewHash { elements: Vec<InsnId>, state: InsnId },
    NewRange { low: InsnId, high: InsnId, flag: RangeType, state: InsnId },
    NewRangeFixnum { low: InsnId, high: InsnId, flag: RangeType, state: InsnId },
    ArrayDup { val: InsnId, state: InsnId },
    ArrayHash { elements: Vec<InsnId>, state: InsnId },
    ArrayMax { elements: Vec<InsnId>, state: InsnId },
    ArrayMin { elements: Vec<InsnId>, state: InsnId },
    ArrayInclude { elements: Vec<InsnId>, target: InsnId, state: InsnId },
    ArrayPackBuffer { elements: Vec<InsnId>, fmt: InsnId, buffer: Option<InsnId>, state: InsnId },
    DupArrayInclude { ary: VALUE, target: InsnId, state: InsnId },
    /// Extend `left` with the elements from `right`. `left` and `right` must both be `Array`.
    ArrayExtend { left: InsnId, right: InsnId, state: InsnId },
    /// Push `val` onto `array`, where `array` is already `Array`.
    ArrayPush { array: InsnId, val: InsnId, state: InsnId },
    ArrayAref { array: InsnId, index: InsnId },
    /// Like [`Insn::ArrayAref`], but `index` (already adjusted by [`Insn::AdjustBounds`]) may be
    /// out of `0...length`, in which case the result is `nil`. Lets `ary[i]` compile without a
    /// bounds guard, so the ordinary Ruby idiom of walking an array until it reads past the end
    /// does not have to leave JIT code.
    ArrayArefOrNil { array: InsnId, index: InsnId, length: InsnId },
    ArrayAset { array: InsnId, index: InsnId, val: InsnId },
    /// Store `val` into `array[index]`, growing `array` when `index` is past the end. Unlike
    /// [`Insn::ArrayAset`], which needs an index that was already bounds-checked with side-exiting
    /// guards, this takes the raw (possibly negative, possibly out-of-range) index plus the array
    /// `length` and calls `rb_ary_store` for out-of-range indices. That matches `Array#[]=`, which
    /// grows the array instead of raising, so there is no reason to leave the JIT for it. A
    /// negative index that is still negative after adjustment raises IndexError, which
    /// `rb_ary_store` does for us.
    ///
    /// `array` must already be known unfrozen and unshared, like [`Insn::ArrayAset`].
    ArrayAsetOrStore { array: InsnId, index: InsnId, length: InsnId, val: InsnId, state: InsnId },
    ArrayPop { array: InsnId, state: InsnId },
    /// Return the length of the array as a C `long` ([`types::CInt64`])
    ArrayLength { array: InsnId },
    /// Adjust potentially-negative index by the given length, returning the adjusted index. If
    /// still negative, return a negative number, which indicates the index is still out-of-bounds.
    AdjustBounds { index: InsnId, length: InsnId },

    HashAref { hash: InsnId, key: InsnId, state: InsnId },
    HashAset { hash: InsnId, key: InsnId, val: InsnId, state: InsnId },
    HashDup { val: InsnId, state: InsnId },

    /// Allocate an instance of the `val` object without calling `#initialize` on it.
    /// This can:
    /// * raise an exception if `val` is not a class
    /// * run arbitrary code if `val` is a class with a custom allocator
    ObjectAlloc { val: InsnId, state: InsnId },
    /// Allocate an instance of the `val` class without calling `#initialize` on it.
    /// This requires that `class` has the default allocator (for example via `IsMethodCfunc`).
    /// This won't raise or run arbitrary code because `class` has the default allocator.
    ObjectAllocClass { class: VALUE, state: InsnId },

    /// Check if the value is truthy and "return" a C boolean. In reality, we will likely fuse this
    /// with IfTrue/IfFalse in the backend to generate jcc.
    Test { val: InsnId },
    /// Return C `true` if `val`'s method on cd resolves to the cfunc.
    IsMethodCfunc { val: InsnId, cd: *const rb_call_data, cfunc: *const u8, state: InsnId },
    /// Return C `true` if left == right
    IsBitEqual { left: InsnId, right: InsnId },
    /// Return C `true` if left != right
    IsBitNotEqual { left: InsnId, right: InsnId },
    /// Convert a C `bool` to a Ruby `Qtrue`/`Qfalse`. Same as `RBOOL` macro.
    BoxBool { val: InsnId },
    /// Convert a C `long` to a Ruby `Fixnum`. Side exit on overflow.
    BoxFixnum { val: InsnId, state: InsnId },
    UnboxFixnum { val: InsnId },
    FixnumAref { recv: InsnId, index: InsnId },
    // TODO(max): In iseq body types that are not ISEQ_TYPE_METHOD, rewrite to Constant false.
    // `lep_level` is the lexical distance from this insn's iseq up to its local_iseq, used only
    // for the DEFINED_YIELD op_type to materialize the local EP inline. Zero for other op_types.
    Defined { op_type: defined_type, obj: VALUE, pushval: VALUE, v: InsnId, lep_level: u32, state: InsnId },
    GetConstant { klass: InsnId, id: ID, allow_nil: InsnId, state: InsnId },
    GetConstantPath { ic: *const iseq_inline_constant_cache, state: InsnId },
    /// Kernel#block_given? but without pushing a frame. Similar to [`Insn::Defined`] with
    /// `DEFINED_YIELD`
    IsBlockGiven { block_handler: InsnId },
    /// Test the bit at index of val, a Fixnum.
    /// Return Qtrue if the bit is set, else Qfalse.
    FixnumBitCheck { val: InsnId, index: u8 },
    /// Return Qtrue if `val` is an instance of `class`, else Qfalse.
    /// Equivalent to `class_search_ancestor(CLASS_OF(val), class)`.
    IsA { val: InsnId, class: InsnId },
    /// `case`/`when`/`rescue` match check for `pattern` against `target`.
    CheckMatch { target: InsnId, pattern: InsnId, flag: u32, state: InsnId },

    /// Get a global variable named `id`
    GetGlobal { id: ID, state: InsnId },
    /// Set a global variable named `id` to `val`
    SetGlobal { id: ID, val: InsnId, state: InsnId },

    //NewObject?
    /// Get an instance variable `id` from `self_val`, using the inline cache `ic` if present
    GetIvar { self_val: InsnId, id: ID, ic: *const iseq_inline_iv_cache_entry, state: InsnId },
    /// Record the shape of a receiver that reached a frozen ivar dispatch's fallback path, so
    /// the site can earn a recompile that specializes it. See
    /// [`crate::profile::rb_zjit_ivar_reprofile`].
    IvarReprofile { self_val: InsnId, state: InsnId },
    /// Set `self_val`'s instance variable `id` to `val`, using the inline cache `ic` if present
    SetIvar { self_val: InsnId, id: ID, val: InsnId, ic: *const iseq_inline_iv_cache_entry, state: InsnId },
    /// Check whether an instance variable exists on `self_val`
    DefinedIvar { self_val: InsnId, id: ID, pushval: VALUE, state: InsnId },

    /// Load cfp->pc
    LoadPC,
    /// Load EC
    LoadEC,
    /// Load SP
    LoadSP,
    /// Load cfp->self
    LoadSelf,
    LoadField { recv: InsnId, id: FieldName, offset: i32, return_type: Type, num_bits: u8 },
    /// Read the method entry (or cref) that `val` designates, where `val` is the value of a
    /// frame's `ep[VM_ENV_DATA_INDEX_ME_CREF]`. That slot normally holds the frame's method
    /// entry, but the VM overwrites it with an `imemo_svar` wrapping the entry the first time
    /// the frame touches a special variable (`$~`, `$_`, a back reference, ...), so read
    /// through the svar when there is one. This mirrors `rb_vm_frame_method_entry`, which is
    /// what the profiler records.
    UnwrapSvar { val: InsnId },
    /// Write `val` at an offset of `recv`.
    /// When writing a Ruby object to a Ruby object, one must use GuardNotFrozen (or equivalent) before and WriteBarrier after.
    StoreField { recv: InsnId, id: FieldName, offset: i32, val: InsnId, num_bits: u8 },
    WriteBarrier { recv: InsnId, val: InsnId },

    /// Check whether VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM is set in the (already loaded) environment flags.
    /// Returns CBool (0/1).
    IsBlockParamModified { flags: InsnId },
    /// Get the block parameter as a Proc.
    GetBlockParam { level: u32, ep_offset: u32, state: InsnId },
    /// Set a local variable in a higher scope or the heap
    SetLocal { level: u32, ep_offset: u32, val: InsnId, state: InsnId },
    GetSpecialSymbol { symbol_type: SpecialBackrefSymbol, state: InsnId },
    GetSpecialNumber { nth: u64, state: InsnId },

    /// `once`: run `body_iseq` at most once and return the value cached in `ise`.
    ///
    /// `ise` points into the `is_entries` of the ISEQ that contains the `once`
    /// instruction, so it stays valid for as long as that ISEQ is alive. It is not a
    /// `VALUE`, so it needs no GC offset; `ise->once.value` is marked by ISEQ marking.
    Once { body_iseq: IseqPtr, ise: *const iseq_inline_storage_entry, state: InsnId },

    /// Get a class variable `id`
    GetClassVar { id: ID, ic: *const iseq_inline_cvar_cache_entry, state: InsnId },
    /// Set a class variable `id` to `val`
    SetClassVar { id: ID, val: InsnId, ic: *const iseq_inline_cvar_cache_entry, state: InsnId },

    /// Get the EP at the given level from the current CFP.
    GetEP { level: u32 },

    /// Own a FrameState so that instructions can look up their dominating FrameState when
    /// generating deopt side-exits and frame reconstruction metadata. Does not directly generate
    /// any code.
    Snapshot { state: Box<FrameState> },

    /// Unconditional jump
    Jump(BranchEdge),

    /// Conditional branch
    CondBranch { val: InsnId, if_true: BranchEdge, if_false: BranchEdge },

    /// Call a C function without pushing a frame
    /// `name` and `owner` are for printing purposes only
    CCall { cfunc: *const u8, recv: InsnId, args: Vec<InsnId>, name: ID, owner: VALUE, return_type: Type, elidable: bool },

    /// Call a C function that pushes a frame
    CCallWithFrame(Box<CCallWithFrameData>),

    /// Call a variadic C function with signature: func(int argc, VALUE *argv, VALUE recv)
    /// This handles frame setup, argv creation, and frame teardown all in one
    CCallVariadic(Box<CCallVariadicData>),

    /// Un-optimized fallback implementation (dynamic dispatch) for send-ish instructions
    /// Ignoring keyword arguments etc for now
    Send {
        recv: InsnId,
        cd: *const rb_call_data,
        block: Option<BlockHandler>,
        args: Vec<InsnId>,
        state: InsnId,
        reason: SendFallbackReason,
    },
    SendForward {
        recv: InsnId,
        cd: *const rb_call_data,
        blockiseq: IseqPtr,
        args: Vec<InsnId>,
        state: InsnId,
        reason: SendFallbackReason,
    },
    InvokeSuper {
        recv: InsnId,
        cd: *const rb_call_data,
        blockiseq: IseqPtr,
        args: Vec<InsnId>,
        state: InsnId,
        reason: SendFallbackReason,
    },
    InvokeSuperForward {
        recv: InsnId,
        cd: *const rb_call_data,
        blockiseq: IseqPtr,
        args: Vec<InsnId>,
        state: InsnId,
        reason: SendFallbackReason,
    },
    InvokeBlock {
        cd: *const rb_call_data,
        args: Vec<InsnId>,
        state: InsnId,
        reason: SendFallbackReason,
    },
    /// Optimized invokeblock for IFUNC block handlers.
    /// Calls rb_vm_yield_with_cfunc directly instead of going through rb_vm_invokeblock.
    InvokeBlockIfunc {
        cd: *const rb_call_data,
        block_handler: InsnId,
        args: Vec<InsnId>,
        state: InsnId,
    },
    /// Call Proc#call optimized method type.
    InvokeProc {
        recv: InsnId,
        args: Vec<InsnId>,
        state: InsnId,
        kw_splat: bool,
    },
    /// Fast-path `yield`: the enclosing frame's block is a known ISEQ block, so push its frame
    /// inline and jump straight to its JIT entry. `iseq` is the comptime-known block iseq; the
    /// tag + iseq-identity guards live in preceding HIR instructions, and `captured` is the
    /// guarded, untagged `struct rb_captured_block *` codegen reads self/ep from. Args are the
    /// positional arguments (lead-only, exact arity).
    InvokeBlockIseqDirect {
        iseq: IseqPtr,
        /// Guarded `struct rb_captured_block *` (block handler with the ISEQ tag masked off).
        captured: InsnId,
        args: Vec<InsnId>,
        state: InsnId,
    },

    /// Optimized ISEQ call
    SendDirect(Box<SendDirectData>),

    /// Push a lighter weight frame used for inlined methods.
    ///
    /// When `captured` is `Some`, this pushes a *block* frame for an inlined block ISEQ
    /// instead of a method frame: `cme` is null, the frame type is `VM_FRAME_MAGIC_BLOCK`,
    /// and the specval is `VM_GUARDED_PREV_EP(captured->ep)`. `recv` is then the block's
    /// self, which is `captured->self`.
    PushInlineFrame {
        iseq: IseqPtr,
        cme: *const rb_callable_method_entry_t,
        recv: InsnId,
        num_args: u16,
        blockiseq: Option<IseqPtr>,
        /// Guarded `struct rb_captured_block *` when this frame is an inlined block.
        captured: Option<InsnId>,
        state: InsnId,
        /// The interpreter's own state at the call site that this frame was pushed for.
        /// Instructions that [`Function::eliminate_empty_inline_frames`] leaves behind after
        /// eliding this frame run against the caller's frame, so they must side-exit here
        /// rather than to `state`, which describes the callee frame this push sets up. The
        /// two differ whenever the caller's stack was rewritten for the frame setup; see
        /// [`SendDirectData::guard_state`].
        guard_state: InsnId,
    },

    /// Pop a lighter weight frame used for inlined methods.
    PopInlineFrame {
        iseq: IseqPtr,
        argc: usize,
        state: InsnId,
    },

    // Invoke a builtin function
    InvokeBuiltin {
        bf: *const rb_builtin_function,
        recv: InsnId,
        args: Vec<InsnId>,
        state: InsnId,
        leaf: bool,
        return_type: Type,  // BasicObject for unannotated builtins
    },

    /// Set up frame. Remember the address as the JIT entry for the insn_idx in `jit_entry_insns()[jit_entry_idx]`.
    EntryPoint { jit_entry_idx: Option<usize> },
    /// Control flow instructions.
    ///
    /// `pop_inlined_frames` is the number of inlined callee frames that are still on
    /// the CFP stack when this returns, on top of the compiled function's own frame.
    /// It is zero for an ordinary `leave` and non-zero only for a non-local `return`
    /// out of an inlined block, which returns from the compiled function while the
    /// inlined frames between it and the block are still pushed.
    Return { val: InsnId, pop_inlined_frames: u32 },
    /// Non-local control flow. See the throw YARV instruction
    Throw { throw_state: u32, val: InsnId, state: InsnId },

    /// Fixnum +, -, *, /, %, ==, !=, <, <=, >, >=, &, |, ^, <<
    FixnumAdd  { left: InsnId, right: InsnId, state: InsnId },
    FixnumSub  { left: InsnId, right: InsnId, state: InsnId },
    FixnumMult { left: InsnId, right: InsnId, state: InsnId },
    FixnumDiv  { left: InsnId, right: InsnId, state: InsnId },
    FixnumMod  { left: InsnId, right: InsnId, state: InsnId },
    FixnumEq   { left: InsnId, right: InsnId },
    FixnumNeq  { left: InsnId, right: InsnId },
    FixnumLt   { left: InsnId, right: InsnId },
    FixnumLe   { left: InsnId, right: InsnId },
    FixnumGt   { left: InsnId, right: InsnId },
    FixnumGe   { left: InsnId, right: InsnId },
    FixnumAnd  { left: InsnId, right: InsnId },
    FixnumOr   { left: InsnId, right: InsnId },
    FixnumXor  { left: InsnId, right: InsnId },
    IntAnd     { left: InsnId, right: InsnId },
    IntOr      { left: InsnId, right: InsnId },
    FixnumLShift { left: InsnId, right: InsnId, state: InsnId },
    FixnumRShift { left: InsnId, right: InsnId },

    /// Float arithmetic: delegates to rb_float_plus/minus/mul/div with GC preparation
    FloatAdd  { recv: InsnId, other: InsnId, state: InsnId },
    FloatSub  { recv: InsnId, other: InsnId, state: InsnId },
    FloatMul  { recv: InsnId, other: InsnId, state: InsnId },
    FloatDiv  { recv: InsnId, other: InsnId, state: InsnId },
    /// Float comparison: delegates to rb_float_lt/le/gt/ge. Unlike the arithmetic
    /// instructions above these are leaf and allocation-free (they return Qtrue/Qfalse),
    /// so they need neither a FrameState nor GC preparation.
    FloatLt   { left: InsnId, right: InsnId },
    FloatLe   { left: InsnId, right: InsnId },
    FloatGt   { left: InsnId, right: InsnId },
    FloatGe   { left: InsnId, right: InsnId },
    /// Float#to_i: truncate float to integer via rb_jit_flo_to_i
    FloatToInt { recv: InsnId, state: InsnId },

    AnyToString { val: InsnId, state: InsnId },

    /// Refine the known type information of with additional type information.
    /// Computes the intersection of the existing type and the new type.
    RefineType { val: InsnId, new_type: Type },
    /// Return CBool[true] if val has type Type and CBool[false] otherwise.
    HasType { val: InsnId, expected: Type },
    /// Return CBool[true] if val is a heap object whose class is `class` or a
    /// subclass of it, and CBool[false] otherwise. Conservatively false for
    /// immediates and for objects with a singleton class, both of which the
    /// generated code rejects without looking at the ancestry.
    HasAncestor { val: InsnId, class: VALUE },

    /// Side-exit if val doesn't have the expected type.
    GuardType { val: InsnId, guard_type: Type, state: InsnId, recompile: Option<Recompile> },
    /// Side-exit if val is not the expected Const.
    GuardBitEquals { val: InsnId, expected: Const, reason: Box<SideExitReason>, state: InsnId, recompile: Option<Recompile> },
    /// Side-exit if (val & mask) == 0
    GuardAnyBitSet { val: InsnId, mask: Const, mask_name: Option<ID>, reason: Box<SideExitReason>, state: InsnId, recompile: Option<Recompile> },
    /// Side-exit if (val & mask) != 0
    GuardNoBitsSet { val: InsnId, mask: Const, mask_name: Option<ID>, reason: Box<SideExitReason>, state: InsnId },
    /// Side-exit if val is a Hash flagged with RHASH_PASS_AS_KEYWORDS. Such a hash makes the
    /// interpreter reinterpret the last splatted argument as keywords, which the expanded
    /// positional argument list can't reproduce. See CALLER_SETUP_ARG in vm_insnhelper.c.
    GuardNotRuby2KeywordsHash { val: InsnId, state: InsnId, recompile: Option<Recompile> },
    /// Side-exit if left is not greater than or equal to right (both operands are C long).
    /// If recompile is not None, the side exit will profile and invalidate the ISEQ so that it
    /// gets recompiled with the new profile data.
    GuardGreaterEq { left: InsnId, right: InsnId, reason: Box<SideExitReason>, state: InsnId, recompile: Option<Recompile> },
    /// Side-exit if left is not less than right (both operands are C long).
    GuardLess { left: InsnId, right: InsnId, reason: Box<SideExitReason>, state: InsnId },

    /// Generate no code (or padding if necessary) and insert a patch point
    /// that can be rewritten to a side exit when the Invariant is broken.
    PatchPoint { invariant: Invariant, state: InsnId },

    /// Side-exit into the interpreter.
    /// If recompile is not None, the side exit will profile and invalidate the ISEQ
    /// so that it gets recompiled with the new profile data.
    SideExit { state: InsnId, reason: Box<SideExitReason>, recompile: Option<Recompile> },

    /// Increment a counter in ZJIT stats
    IncrCounter(Counter),

    /// Increment a counter in ZJIT stats for the given counter pointer
    IncrCounterPtr { counter_ptr: *mut u64 },

    /// Equivalent of RUBY_VM_CHECK_INTS. Automatically inserted by the compiler before jumps and
    /// return instructions.
    CheckInterrupts { state: InsnId },

    BreakPoint,

    /// Only use this instruction in tests where you need to end a block with
    /// a terminator, but don't ever expect the code to be executed.  This
    /// instruction should never be generated from iseq_to_hir
    Unreachable,
}

/// Macro that enumerates all operands of an Insn, dispatching to caller-provided
/// `$visit_one` macro for a single InsnId field and `$visit_many` macro for a
/// slice/Vec of InsnIds. Used by both `for_each_operand` and `for_each_operand_mut`.
macro_rules! for_each_operand_impl {
    ($self:expr, $visit_one:ident, $visit_many:ident) => {
        match $self {
            Insn::Comment { .. }
            | Insn::Const { .. }
            | Insn::Param
            | Insn::LoadArg { .. }
            | Insn::Entries { .. }
            | Insn::EntryPoint { .. }
            | Insn::LoadPC
            | Insn::LoadSP
            | Insn::LoadEC
            | Insn::GetEP { .. }
            | Insn::LoadSelf
            | Insn::BreakPoint | Insn::Unreachable
            | Insn::IncrCounter(_)
            | Insn::IncrCounterPtr { .. } => {}

            Insn::IsBlockGiven { block_handler } => {
                $visit_one!(*block_handler);
            }
            Insn::IsBlockParamModified { flags } => {
                $visit_one!(*flags);
            }
            Insn::CheckMatch { target, pattern, state, .. } => {
                $visit_one!(*target);
                $visit_one!(*pattern);
                $visit_one!(*state);
            }
            Insn::PatchPoint { state, .. }
            | Insn::CheckInterrupts { state }
            | Insn::PutSpecialObject { state, .. }
            | Insn::GetBlockParam { state, .. }
            | Insn::GetConstantPath { state, .. } => {
                $visit_one!(*state);
            }
            Insn::FixnumBitCheck { val, .. } => {
                $visit_one!(*val);
            }
            Insn::ArrayMax { elements, state, .. }
            | Insn::ArrayMin { elements, state, .. }
            | Insn::ArrayHash { elements, state, .. }
            | Insn::NewHash { elements, state, .. }
            | Insn::NewArray { elements, state, .. } => {
                $visit_many!(elements);
                $visit_one!(*state);
            }
            Insn::ArrayInclude { elements, target, state, .. } => {
                $visit_many!(elements);
                $visit_one!(*target);
                $visit_one!(*state);
            }
            Insn::ArrayPackBuffer { elements, fmt, buffer, state, .. } => {
                $visit_many!(elements);
                $visit_one!(*fmt);
                if let Some(buffer) = buffer {
                    $visit_one!(*buffer);
                }
                $visit_one!(*state);
            }
            Insn::DupArrayInclude { target, state, .. } => {
                $visit_one!(*target);
                $visit_one!(*state);
            }
            Insn::NewRange { low, high, state, .. }
            | Insn::NewRangeFixnum { low, high, state, .. } => {
                $visit_one!(*low);
                $visit_one!(*high);
                $visit_one!(*state);
            }
            Insn::StringConcat { strings, state, .. } => {
                $visit_many!(strings);
                $visit_one!(*state);
            }
            Insn::StringGetbyte { string, index } => {
                $visit_one!(*string);
                $visit_one!(*index);
            }
            Insn::StringCoderangeOrScan { string, cached, state } => {
                $visit_one!(*string);
                $visit_one!(*cached);
                $visit_one!(*state);
            }
            Insn::StringSetbyteFixnum { string, index, value } => {
                $visit_one!(*string);
                $visit_one!(*index);
                $visit_one!(*value);
            }
            Insn::StringAppend { recv, other, state }
            | Insn::StringAppendCodepoint { recv, other, state } => {
                $visit_one!(*recv);
                $visit_one!(*other);
                $visit_one!(*state);
            }
            Insn::StringEqual { left, right } => {
                $visit_one!(*left);
                $visit_one!(*right);
            }
            Insn::ToRegexp { values, state, .. } => {
                $visit_many!(values);
                $visit_one!(*state);
            }
            Insn::RefineType { val, .. }
            | Insn::HasType { val, .. }
            | Insn::HasAncestor { val, .. }
            | Insn::Return { val, .. }
            | Insn::Test { val }
            | Insn::BoxBool { val } => {
                $visit_one!(*val);
            }
            Insn::SetGlobal { val, state, .. }
            | Insn::Defined { v: val, state, .. }
            | Insn::StringIntern { val, state }
            | Insn::StringCopy { val, state, .. }
            | Insn::ObjectAlloc { val, state }
            | Insn::GuardType { val, state, .. }
            | Insn::GuardBitEquals { val, state, .. }
            | Insn::GuardAnyBitSet { val, state, .. }
            | Insn::GuardNoBitsSet { val, state, .. }
            | Insn::GuardNotRuby2KeywordsHash { val, state, .. }
            | Insn::ToArray { val, state }
            | Insn::ToHash { val, state }
            | Insn::CheckArrayType { val, state }
            | Insn::ToAryForExpand { val, state }
            | Insn::IsMethodCfunc { val, state, .. }
            | Insn::ToNewArray { val, state }
            | Insn::SetLocal { val, state, .. }
            | Insn::BoxFixnum { val, state } => {
                $visit_one!(*val);
                $visit_one!(*state);
            }
            Insn::GuardGreaterEq { left, right, state, .. }
            | Insn::GuardLess { left, right, state, .. } => {
                $visit_one!(*left);
                $visit_one!(*right);
                $visit_one!(*state);
            }
            Insn::Snapshot { state } => {
                $visit_many!(state.stack);
                $visit_many!(state.locals);
                // Option iterates like a 0/1-element slice, so visit_many works here.
                $visit_many!(state.caller);
            }
            Insn::FixnumAdd { left, right, state }
            | Insn::FixnumSub { left, right, state }
            | Insn::FixnumMult { left, right, state }
            | Insn::FixnumDiv { left, right, state }
            | Insn::FixnumMod { left, right, state }
            | Insn::ArrayExtend { left, right, state }
            | Insn::FixnumLShift { left, right, state } => {
                $visit_one!(*left);
                $visit_one!(*right);
                $visit_one!(*state);
            }
            Insn::FloatAdd { recv, other, state }
            | Insn::FloatSub { recv, other, state }
            | Insn::FloatMul { recv, other, state }
            | Insn::FloatDiv { recv, other, state } => {
                $visit_one!(*recv);
                $visit_one!(*other);
                $visit_one!(*state);
            }
            Insn::FloatToInt { recv, state } => {
                $visit_one!(*recv);
                $visit_one!(*state);
            }
            Insn::FixnumLt { left, right }
            | Insn::FixnumLe { left, right }
            | Insn::FixnumGt { left, right }
            | Insn::FixnumGe { left, right }
            | Insn::FloatLt { left, right }
            | Insn::FloatLe { left, right }
            | Insn::FloatGt { left, right }
            | Insn::FloatGe { left, right }
            | Insn::FixnumEq { left, right }
            | Insn::FixnumNeq { left, right }
            | Insn::FixnumAnd { left, right }
            | Insn::FixnumOr { left, right }
            | Insn::FixnumXor { left, right }
            | Insn::IntAnd { left, right }
            | Insn::IntOr { left, right }
            | Insn::FixnumRShift { left, right }
            | Insn::IsBitEqual { left, right }
            | Insn::IsBitNotEqual { left, right } => {
                $visit_one!(*left);
                $visit_one!(*right);
            }
            Insn::Jump(BranchEdge { args, .. }) => {
                $visit_many!(args);
            }
            Insn::CondBranch { val, if_true: BranchEdge { args: true_args, .. }, if_false: BranchEdge { args: false_args, .. } } => {
                $visit_one!(*val);
                $visit_many!(true_args);
                $visit_many!(false_args);
            }
            Insn::ArrayDup { val, state }
            | Insn::Throw { val, state, .. }
            | Insn::HashDup { val, state } => {
                $visit_one!(*val);
                $visit_one!(*state);
            }
            Insn::ArrayAref { array, index } => {
                $visit_one!(*array);
                $visit_one!(*index);
            }
            Insn::ArrayArefOrNil { array, index, length } => {
                $visit_one!(*array);
                $visit_one!(*index);
                $visit_one!(*length);
            }
            Insn::ArrayAset { array, index, val } => {
                $visit_one!(*array);
                $visit_one!(*index);
                $visit_one!(*val);
            }
            Insn::ArrayAsetOrStore { array, index, length, val, state } => {
                $visit_one!(*array);
                $visit_one!(*index);
                $visit_one!(*length);
                $visit_one!(*val);
                $visit_one!(*state);
            }
            Insn::ArrayPop { array, state } => {
                $visit_one!(*array);
                $visit_one!(*state);
            }
            Insn::ArrayLength { array } => {
                $visit_one!(*array);
            }
            Insn::AdjustBounds { index, length } => {
                $visit_one!(*index);
                $visit_one!(*length);
            }
            Insn::HashAref { hash, key, state } => {
                $visit_one!(*hash);
                $visit_one!(*key);
                $visit_one!(*state);
            }
            Insn::HashAset { hash, key, val, state } => {
                $visit_one!(*hash);
                $visit_one!(*key);
                $visit_one!(*val);
                $visit_one!(*state);
            }
            Insn::Send { recv, args, state, .. }
            | Insn::SendForward { recv, args, state, .. }
            | Insn::InvokeBuiltin { recv, args, state, .. }
            | Insn::InvokeSuper { recv, args, state, .. }
            | Insn::InvokeSuperForward { recv, args, state, .. }
            | Insn::InvokeProc { recv, args, state, .. } => {
                $visit_one!(*recv);
                $visit_many!(args);
                $visit_one!(*state);
            }
            Insn::PushInlineFrame { recv, captured, state, guard_state, .. } => {
                $visit_one!(*recv);
                if let Some(captured) = captured {
                    $visit_one!(*captured);
                }
                $visit_one!(*state);
                $visit_one!(*guard_state);
            }
            // SendDirect/CCallWithFrame/CCallVariadic carry their operands behind a Box,
            // which stable Rust can't destructure in a pattern. visit_one takes a place, so
            // a box field works the same as the deref'd bindings used by other arms.
            Insn::SendDirect(insn) => {
                $visit_one!(insn.recv);
                $visit_many!(insn.args);
                $visit_many!(insn.block_arg);
                $visit_one!(insn.state);
                $visit_one!(insn.guard_state);
            }
            Insn::CCallWithFrame(insn) => {
                $visit_one!(insn.recv);
                $visit_many!(insn.args);
                $visit_many!(insn.block_arg);
                $visit_one!(insn.state);
            }
            Insn::CCallVariadic(insn) => {
                $visit_one!(insn.recv);
                $visit_many!(insn.args);
                $visit_many!(insn.block_arg);
                $visit_one!(insn.state);
            }
            Insn::InvokeBlock { args, state, .. } => {
                $visit_many!(args);
                $visit_one!(*state);
            }
            Insn::InvokeBlockIseqDirect { captured, args, state, .. } => {
                $visit_one!(*captured);
                $visit_many!(args);
                $visit_one!(*state);
            }
            Insn::InvokeBlockIfunc { block_handler, args, state, .. } => {
                $visit_one!(*block_handler);
                $visit_many!(args);
                $visit_one!(*state);
            }
            Insn::CCall { recv, args, .. } => {
                $visit_one!(*recv);
                $visit_many!(args);
            }
            Insn::IvarReprofile { self_val, state }
            | Insn::GetIvar { self_val, state, .. }
            | Insn::DefinedIvar { self_val, state, .. } => {
                $visit_one!(*self_val);
                $visit_one!(*state);
            }
            Insn::GetConstant { klass, allow_nil, state, .. } => {
                $visit_one!(*klass);
                $visit_one!(*allow_nil);
                $visit_one!(*state);
            }
            Insn::SetIvar { self_val, val, state, .. } => {
                $visit_one!(*self_val);
                $visit_one!(*val);
                $visit_one!(*state);
            }
            Insn::GetClassVar { state, .. }
            | Insn::PopInlineFrame { state, .. } => {
                $visit_one!(*state);
            }
            Insn::SetClassVar { val, state, .. } => {
                $visit_one!(*val);
                $visit_one!(*state);
            }
            Insn::ArrayPush { array, val, state } => {
                $visit_one!(*array);
                $visit_one!(*val);
                $visit_one!(*state);
            }
            Insn::AnyToString { val, state, .. } => {
                $visit_one!(*val);
                $visit_one!(*state);
            }
            Insn::LoadField { recv, .. } => {
                $visit_one!(*recv);
            }
            Insn::UnwrapSvar { val } => {
                $visit_one!(*val);
            }
            Insn::StoreField { recv, val, .. }
            | Insn::WriteBarrier { recv, val } => {
                $visit_one!(*recv);
                $visit_one!(*val);
            }
            Insn::GetGlobal { state, .. }
            | Insn::GetSpecialSymbol { state, .. }
            | Insn::GetSpecialNumber { state, .. }
            | Insn::Once { state, .. }
            | Insn::ObjectAllocClass { state, .. }
            | Insn::SideExit { state, .. } => {
                $visit_one!(*state);
            }
            Insn::UnboxFixnum { val } => {
                $visit_one!(*val);
            }
            Insn::FixnumAref { recv, index } => {
                $visit_one!(*recv);
                $visit_one!(*index);
            }
            Insn::IsA { val, class } => {
                $visit_one!(*val);
                $visit_one!(*class);
            }
        }
    };
}

impl Insn {
    /// Not every instruction returns a value. Return true if the instruction does and false otherwise.
    pub fn has_output(&self) -> bool {
        match self {
            Insn::Comment { .. }
            | Insn::Jump(_)
            | Insn::Entries { .. }
            | Insn::CondBranch { .. } | Insn::EntryPoint { .. } | Insn::Return { .. }
            | Insn::PatchPoint { .. } | Insn::SetIvar { .. } | Insn::SetClassVar { .. } | Insn::ArrayExtend { .. }
            | Insn::ArrayPush { .. } | Insn::SideExit { .. } | Insn::SetGlobal { .. }
            | Insn::SetLocal { .. } | Insn::Throw { .. } | Insn::IncrCounter(_) | Insn::IncrCounterPtr { .. }
            | Insn::CheckInterrupts { .. } | Insn::BreakPoint | Insn::Unreachable
            | Insn::StoreField { .. } | Insn::WriteBarrier { .. } | Insn::HashAset { .. }
            | Insn::ArrayAset { .. } | Insn::ArrayAsetOrStore { .. } | Insn::IvarReprofile { .. }
            | Insn::PushInlineFrame { .. } | Insn::PopInlineFrame { .. } => false,
            _ => true,
        }
    }

    /// Return true if the instruction ends a basic block and false otherwise.
    pub fn is_terminator(&self) -> bool {
        match self {
            Insn::Unreachable | Insn::CondBranch { .. } | Insn::Jump(_) | Insn::Entries { .. } | Insn::Return { .. } | Insn::SideExit { .. } | Insn::Throw { .. } => true,
            _ => false,
        }
    }

    /// Return true if the instruction is a jump (has successor blocks in the CFG).
    pub fn is_jump(&self) -> bool {
        match self {
            Insn::CondBranch { .. } | Insn::Jump(_) | Insn::Entries { .. } => true,
            _ => false,
        }
    }

    /// Call `f` on each operand (InsnId) of this instruction.
    pub fn for_each_operand(&self, mut f: impl FnMut(InsnId)) {
        macro_rules! visit_one { ($p:expr) => { f($p) }; }
        macro_rules! visit_many { ($s:expr) => { for id in ($s).iter() { f(*id) } }; }
        for_each_operand_impl!(self, visit_one, visit_many);
    }

    /// Call `f` on a mutable reference to each operand (InsnId) of this instruction.
    pub fn for_each_operand_mut(&mut self, mut f: impl FnMut(&mut InsnId)) {
        macro_rules! visit_one { ($p:expr) => { f(&mut $p) }; }
        macro_rules! visit_many { ($s:expr) => { for id in ($s).iter_mut() { f(id) } }; }
        for_each_operand_impl!(self, visit_one, visit_many);
    }

    /// Call `f` on each operand, short-circuiting on the first error.
    pub fn try_for_each_operand<E>(&self, mut f: impl FnMut(InsnId) -> Result<(), E>) -> Result<(), E> {
        macro_rules! visit_one { ($p:expr) => { f($p)? }; }
        macro_rules! visit_many { ($s:expr) => { for id in ($s).iter() { f(*id)? } }; }
        for_each_operand_impl!(self, visit_one, visit_many);
        Ok(())
    }

    pub fn print<'a>(&self, ptr_map: &'a PtrPrintMap, fun: Option<&'a Function>) -> InsnPrinter<'a> {
        InsnPrinter { inner: self.clone(), ptr_map, fun }
    }

    // TODO(Jacob): Model SP. ie, all allocations modify stack size but using the effect for stack modification feels excessive
    // TODO(Jacob): Add sideeffect failure bit
    fn effects_of(&self) -> Effect {
        const allocates: Effect = Effect::read_write(abstract_heaps::PC.union(abstract_heaps::Allocator), abstract_heaps::Allocator);
        match &self {
            Insn::Comment { .. } => effects::Empty,
            Insn::Const { .. } => effects::Empty,
            Insn::Param { .. } => effects::Empty,
            Insn::LoadArg { .. } => effects::Empty,
            Insn::StringCopy { .. } => allocates,
            Insn::StringIntern { .. } => effects::Any,
            Insn::StringConcat { .. } => effects::Any,
            Insn::StringGetbyte { .. } => Effect::read_write(abstract_heaps::Other, abstract_heaps::Empty),
            // Scanning caches the computed coderange in the string's RBASIC flags, so later loads
            // of those flags must not be forwarded from ones taken before this instruction.
            Insn::StringCoderangeOrScan { .. } => effects::Any,
            // Writes both the string contents and the flags (to clear the coderange)
            Insn::StringSetbyteFixnum { .. } => Effect::read_write(abstract_heaps::Other, abstract_heaps::Other),
            Insn::StringAppend { .. } => effects::Any,
            Insn::StringAppendCodepoint { .. } => effects::Any,
            Insn::StringEqual { .. } => Effect::write(abstract_heaps::Allocator),
            Insn::ToRegexp { .. } => effects::Any,
            Insn::PutSpecialObject { .. } => effects::Any,
            Insn::ToArray { .. } => effects::Any,
            Insn::ToHash { .. } => effects::Any,
            Insn::CheckArrayType { .. } => effects::Any,
            Insn::ToAryForExpand { .. } => effects::Any,
            Insn::ToNewArray { .. } => effects::Any,
            Insn::NewArray { .. } => allocates,
            Insn::NewHash { elements, .. } => {
                // NewHash's operands may be hashed and compared for equality, which could have
                // side-effects. Empty hashes are definitely elidable.
                if elements.is_empty() {
                    Effect::write(abstract_heaps::Allocator)
                }
                else {
                    effects::Any
                }
            },
            Insn::NewRange { .. } => effects::Any,
            Insn::NewRangeFixnum { .. } => allocates,
            Insn::ArrayDup { .. } => allocates,
            Insn::ArrayHash { .. } => effects::Any,
            Insn::ArrayMax { .. } => effects::Any,
            Insn::ArrayMin { .. } => effects::Any,
            Insn::ArrayInclude { .. } => effects::Any,
            Insn::ArrayPackBuffer { .. } => effects::Any,
            Insn::DupArrayInclude { .. } => effects::Any,
            Insn::ArrayExtend { .. } => effects::Any,
            Insn::ArrayPush { .. } => effects::Any,
            Insn::ArrayAref { ..  } => effects::Any,
            Insn::ArrayArefOrNil { ..  } => effects::Any,
            Insn::ArrayAset { .. } => effects::Any,
            Insn::ArrayAsetOrStore { .. } => effects::Any,
            Insn::ArrayPop { ..  } => effects::Any,
            Insn::ArrayLength { .. } => Effect::write(abstract_heaps::Empty),
            Insn::AdjustBounds { .. } => effects::Empty,
            Insn::HashAref { .. } => effects::Any,
            Insn::HashAset { .. } => effects::Any,
            Insn::HashDup { .. } => allocates,
            Insn::ObjectAlloc { .. } => effects::Any,
            Insn::ObjectAllocClass { .. } => allocates,
            Insn::Test { .. } => effects::Empty,
            Insn::IsMethodCfunc { .. } => effects::Any,
            Insn::IsBitEqual { .. } => effects::Empty,
            Insn::IsBitNotEqual { .. } => effects::Empty,
            Insn::BoxBool { .. } => effects::Empty,
            Insn::BoxFixnum { .. } => effects::Empty,
            Insn::UnboxFixnum { .. } => effects::Empty,
            Insn::FixnumAref { .. } => effects::Empty,
            Insn::Defined { .. } => effects::Any,
            Insn::GetConstant { .. } => effects::Any,
            Insn::GetConstantPath { .. } => effects::Any,
            Insn::IsBlockGiven { .. } => effects::Empty,
            Insn::FixnumBitCheck { .. } => effects::Empty,
            // IsA needs to read the class of the value and traverse the class hierarchy, which we model as reading from Memory.
            Insn::IsA { .. } => Effect::read_write(abstract_heaps::Memory, abstract_heaps::Empty),
            Insn::GetGlobal { .. } => effects::Any,
            Insn::SetGlobal { .. } => effects::Any,
            Insn::GetIvar { .. } => effects::Any,
            Insn::IvarReprofile { .. } => effects::Any,
            Insn::SetIvar { .. } => effects::Any,
            Insn::DefinedIvar { .. } => effects::Any,
            Insn::LoadPC { .. } => Effect::read_write(abstract_heaps::PC, abstract_heaps::Empty),
            Insn::LoadEC { .. } => effects::Empty,
            Insn::LoadSP { .. } => effects::Empty,
            // GetEP reads from the current frame pointer (abstract_heaps::Frame) and also traverses previous frames too.
            Insn::GetEP { .. } => Effect::read_write(abstract_heaps::Memory, abstract_heaps::Empty),
            Insn::LoadSelf { .. } => Effect::read_write(abstract_heaps::Frame, abstract_heaps::Empty),
            Insn::LoadField { .. } => Effect::read_write(abstract_heaps::Memory, abstract_heaps::Empty),
            Insn::UnwrapSvar { .. } => Effect::read_write(abstract_heaps::Memory, abstract_heaps::Empty),
            Insn::StoreField { .. } => effects::Any,
            // TODO: Refine CheckMatch effects by flag.
            Insn::CheckMatch { .. } => effects::Any,
            // WriteBarrier can write to object flags and mark bits in Allocator memory.
            // This is why WriteBarrier writes to the "Memory" effect. We do not yet have a more granular specialization for flags
            Insn::WriteBarrier { .. } => Effect::read_write(abstract_heaps::Allocator, abstract_heaps::Allocator.union(abstract_heaps::Memory)),
            Insn::SetLocal { .. } => effects::Any,
            Insn::GetSpecialSymbol { .. } => effects::Any,
            Insn::GetSpecialNumber { .. } => effects::Any,
            // The `once` body can run arbitrary Ruby code on its first execution.
            Insn::Once { .. } => effects::Any,
            Insn::GetClassVar { .. } => effects::Any,
            Insn::SetClassVar { .. } => effects::Any,
            Insn::IsBlockParamModified { .. } => effects::Empty,
            Insn::GetBlockParam { .. } => effects::Any,
            Insn::Snapshot { .. } => effects::Empty,
            Insn::Jump(_) => effects::Any,
            Insn::CondBranch { .. } => effects::Any,
            Insn::CCall { elidable, .. } => {
                if *elidable {
                    Effect::write(abstract_heaps::Allocator)
                }
                else {
                    effects::Any
                }
            },
            Insn::CCallWithFrame(insn) => {
                if insn.elidable {
                    Effect::write(abstract_heaps::Allocator)
                }
                else {
                    effects::Any
                }
            },
            Insn::CCallVariadic(_) => effects::Any,
            Insn::Send { .. } => effects::Any,
            Insn::SendForward { .. } => effects::Any,
            Insn::InvokeSuper { .. } => effects::Any,
            Insn::InvokeSuperForward { .. } => effects::Any,
            Insn::InvokeBlock { .. } => effects::Any,
            Insn::InvokeBlockIfunc { .. } => effects::Any,
            Insn::SendDirect(_) => effects::Any,
            // TODO (nirvdrum 2026-05-28): Revisit when PushInlineFrame is
            // actually lightweight. The frame writes here pay for the spill
            // ceremony in the current full frame-push codegen. A lightweight
            // push that keeps caller state in SSA should drop Frame (and likely
            // most of Other) from the write set, and PopLightweightFrame could
            // collapse to Empty.
            //
            // Currently, spills caller PC, SP, locals, and stack and installs a
            // new CFP. It may side-exit on stack overflow. It does not allocate,
            // invalidate PatchPoint invariants, or set the interrupt flag, so
            // optimizations tracking those heaps can flow across an inlined call.
            Insn::PushInlineFrame { .. } => Effect::read_write(
                abstract_heaps::Memory,
                abstract_heaps::Frame
                    .union(abstract_heaps::Other)
                    .union(abstract_heaps::Control),
            ),
            // Restores SP/CFP and updates ec->cfp.
            // No side exit, no allocation, no PatchPoint invalidation, no interrupt flag write.
            Insn::PopInlineFrame { .. } => Effect::read_write(
                abstract_heaps::Empty,
                abstract_heaps::Other,
            ),
            Insn::InvokeBuiltin { .. } => effects::Any,
            Insn::EntryPoint { .. } => effects::Any,
            Insn::Return { .. } => effects::Any,
            Insn::Throw { .. } => effects::Any,
            Insn::FixnumAdd { .. } => effects::Empty,
            Insn::FixnumSub { .. } => effects::Empty,
            Insn::FixnumMult { .. } => effects::Empty,
            Insn::FixnumDiv { .. } => effects::Any,
            Insn::FixnumMod { .. } => effects::Any,
            Insn::FloatAdd { .. } => effects::Any,
            Insn::FloatSub { .. } => effects::Any,
            Insn::FloatMul { .. } => effects::Any,
            Insn::FloatDiv { .. } => effects::Any,
            Insn::FloatToInt { .. } => effects::Any,
            Insn::FloatLt { .. } => effects::Empty,
            Insn::FloatLe { .. } => effects::Empty,
            Insn::FloatGt { .. } => effects::Empty,
            Insn::FloatGe { .. } => effects::Empty,
            Insn::FixnumEq { .. } => effects::Empty,
            Insn::FixnumNeq { .. } => effects::Empty,
            Insn::FixnumLt { .. } => effects::Empty,
            Insn::FixnumLe { .. } => effects::Empty,
            Insn::FixnumGt { .. } => effects::Empty,
            Insn::FixnumGe { .. } => effects::Empty,
            Insn::FixnumAnd { .. } => effects::Empty,
            Insn::FixnumOr { .. } => effects::Empty,
            Insn::FixnumXor { .. } => effects::Empty,
            Insn::IntAnd { .. } => effects::Empty,
            Insn::IntOr { .. } => effects::Empty,
            Insn::FixnumLShift { .. } => effects::Empty,
            Insn::FixnumRShift { .. } => effects::Empty,
            Insn::AnyToString { .. } => effects::Any,
            Insn::GuardType { guard_type, .. }
                => Effect::read_write(
                    if guard_type.is_subtype(types::Immediate) { abstract_heaps::Empty } else { abstract_heaps::Memory },
                    abstract_heaps::Control
                ),
            Insn::GuardBitEquals { .. } => Effect::read_write(abstract_heaps::Empty, abstract_heaps::Control),
            Insn::GuardAnyBitSet { .. } => Effect::read_write(abstract_heaps::Empty, abstract_heaps::Control),
            Insn::GuardNoBitsSet { .. } => Effect::read_write(abstract_heaps::Empty, abstract_heaps::Control),
            // Reads the object header of a heap object to check its type and flags.
            Insn::GuardNotRuby2KeywordsHash { .. } => Effect::read_write(abstract_heaps::Memory, abstract_heaps::Control),
            Insn::GuardGreaterEq { .. } => Effect::read_write(abstract_heaps::Empty, abstract_heaps::Control),
            Insn::GuardLess { .. } => Effect::read_write(abstract_heaps::Empty, abstract_heaps::Control),
            Insn::PatchPoint { .. } => Effect::read_write(abstract_heaps::PatchPoint, abstract_heaps::Control),
            Insn::SideExit { .. } => effects::Any,
            Insn::IncrCounter(_) => Effect::read_write(abstract_heaps::Empty, abstract_heaps::Stats),
            Insn::IncrCounterPtr { .. } => Effect::read_write(abstract_heaps::Empty, abstract_heaps::Stats),
            Insn::CheckInterrupts { .. } => Effect::read_write(abstract_heaps::InterruptFlag, abstract_heaps::Control),
            Insn::InvokeProc { .. } => effects::Any,
            Insn::InvokeBlockIseqDirect { .. } => effects::Any,
            Insn::RefineType { .. } => effects::Empty,
            Insn::HasType { expected, .. }
                => Effect::read_write(
                    if expected.is_subtype(types::Immediate) { abstract_heaps::Empty } else { abstract_heaps::Memory },
                    abstract_heaps::Empty
                ),
            Insn::HasAncestor { .. } => Effect::read_write(abstract_heaps::Memory, abstract_heaps::Empty),
            Insn::Entries { .. } => effects::Any,
            Insn::BreakPoint | Insn::Unreachable => Effect::read_write(abstract_heaps::Empty, abstract_heaps::Control),
        }
    }

    /// Return true if we can safely omit the instruction. This occurs when one of the following
    /// conditions are met.
    /// 1. The instruction does not write anything.
    /// 2. The instruction only allocates and writes nothing else.
    /// Calling the effects of our instruction `insn_effects`, we need:
    /// `effects::Empty` to include `insn_effects.write` or `effects::Allocator` to include
    /// `insn_effects.write`.
    /// We can simplify this to `effects::Empty.union(effects::Allocator).includes(insn_effects.write)`.
    /// But the union of `Allocator` and `Empty` is simply `Allocator`, so our entire function
    /// collapses to `effects::Allocator.includes(insn_effects.write)`.
    /// Note: These are restrictions on the `write` `EffectSet` only. Even instructions with
    /// `read: effects::Any` could potentially be omitted.
    fn is_elidable(&self) -> bool {
        // Comments intentionally have no semantic effect, but they are diagnostics that should
        // survive DCE so optimized HIR dumps retain the information callers inserted.
        if matches!(self, Insn::Comment { .. }) {
            return false;
        }

        abstract_heaps::Allocator.includes(self.effects_of().write_bits())
    }

    fn counts_against_inlining_budget(&self) -> bool {
        match self {
            // Don't count metadata-only instructions.
            Insn::Comment { .. }
            | Insn::IncrCounter { .. }
            | Insn::IncrCounterPtr { .. }
            | Insn::Snapshot { .. }
            | Insn::PatchPoint { .. }
            => false,
            _ => true,
        }
    }
}

/// Print adaptor for [`Insn`]. See [`PtrPrintMap`].
pub struct InsnPrinter<'a> {
    fun: Option<&'a Function>,
    inner: Insn,
    ptr_map: &'a PtrPrintMap,
}

fn get_local_var_id(iseq: IseqPtr, level: u32, ep_offset: u32) -> ID {
    let mut current_iseq = iseq;
    for _ in 0..level {
        current_iseq = unsafe { rb_get_iseq_body_parent_iseq(current_iseq) };
    }
    let local_idx = ep_offset_to_local_idx(current_iseq, ep_offset);
    unsafe { rb_zjit_local_id(current_iseq, local_idx.try_into().unwrap()) }
}

/// Get the name of a local variable given iseq, level, and ep_offset.
/// Returns
/// - `":name"` if iseq is available and name is a real identifier,
/// - `"<empty>"` for anonymous locals.
/// - `None` if iseq is not available.
///   (When `Insn` is printed in a panic/debug message the `Display::fmt` method is called, which can't access an iseq.)
///
/// This mimics local_var_name() from iseq.c.
fn get_local_var_name_for_printer(iseq: Option<IseqPtr>, level: u32, ep_offset: u32) -> Option<String> {
    let id = get_local_var_id(iseq?, level, ep_offset);

    if id_is_empty(id) {
        return Some(String::from("<empty>"));
    }

    Some(format!(":{}", id.contents_lossy()))
}


fn id_is_empty(id: ID) -> bool {
    id.0 == 0 || unsafe { rb_id2str(id) } == Qfalse
}

/// Construct a qualified method name for display/debug output.
/// Returns strings like "Array#length" for instance methods or "Foo.bar" for singleton methods.
pub(crate) fn qualified_method_name(class: VALUE, method_id: ID) -> String {
    let method_name = method_id.contents_lossy();
    // rb_zjit_singleton_class_p also checks if it's a class
    if unsafe { rb_zjit_singleton_class_p(class) } {
        let class_name = get_class_name(unsafe { rb_class_attached_object(class) });
        format!("{class_name}.{method_name}")
    } else {
        let class_name = get_class_name(class);
        format!("{class_name}#{method_name}")
    }
}

static REGEXP_FLAGS: &[(u32, &str)] = &[
    (ONIG_OPTION_MULTILINE, "MULTILINE"),
    (ONIG_OPTION_IGNORECASE, "IGNORECASE"),
    (ONIG_OPTION_EXTEND, "EXTENDED"),
    (ARG_ENCODING_FIXED, "FIXEDENCODING"),
    (ARG_ENCODING_NONE, "NOENCODING"),
];

impl<'a> std::fmt::Display for InsnPrinter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        macro_rules! write_separated {
            ($f:expr, $start:expr, $sep: expr, $vec:expr) => {
                {
                    let mut sep = $start;
                    for item in $vec {
                        write!($f, "{sep}{item}")?;
                        sep = $sep;
                    }
                }
            };
        }
        match &self.inner {
            Insn::Comment { message } => write!(f, "# {message}"),
            Insn::Const { val } => { write!(f, "Const {}", val.print(self.ptr_map)) }
            Insn::Param => { write!(f, "Param") }
            Insn::LoadArg { idx, id, .. } => { write!(f, "LoadArg :{id}@{idx}") }
            Insn::Entries { targets } => {
                write!(f, "Entries")?;
                write_separated!(f, " ", ", ", targets);
                Ok(())
            }
            Insn::NewArray { elements, .. } => {
                write!(f, "NewArray")?;
                write_separated!(f, " ", ", ", elements);
                Ok(())
            }
            Insn::ArrayAref { array, index, .. } => {
                write!(f, "ArrayAref {array}, {index}")
            }
            Insn::ArrayArefOrNil { array, index, length } => {
                write!(f, "ArrayArefOrNil {array}, {index}, {length}")
            }
            Insn::ArrayAset { array, index, val, ..} => {
                write!(f, "ArrayAset {array}, {index}, {val}")
            }
            Insn::ArrayAsetOrStore { array, index, length, val, .. } => {
                write!(f, "ArrayAsetOrStore {array}, {index}, {length}, {val}")
            }
            Insn::ArrayPop { array, .. } => {
                write!(f, "ArrayPop {array}")
            }
            Insn::ArrayLength { array } => {
                write!(f, "ArrayLength {array}")
            }
            Insn::AdjustBounds { index, length } => {
                write!(f, "AdjustBounds {index}, {length}")
            }
            Insn::NewHash { elements, .. } => {
                write!(f, "NewHash")?;
                let mut prefix = " ";
                for chunk in elements.chunks(2) {
                    if let [key, value] = chunk {
                        write!(f, "{prefix}{key}: {value}")?;
                        prefix = ", ";
                    }
                }
                Ok(())
            }
            Insn::NewRange { low, high, flag, .. } => {
                write!(f, "NewRange {low} {flag} {high}")
            }
            Insn::NewRangeFixnum { low, high, flag, .. } => {
                write!(f, "NewRangeFixnum {low} {flag} {high}")
            }
            Insn::ArrayMax { elements, .. } => {
                write!(f, "ArrayMax")?;
                write_separated!(f, " ", ", ", elements);
                Ok(())
            }
            Insn::ArrayMin { elements, .. } => {
                write!(f, "ArrayMin")?;
                write_separated!(f, " ", ", ", elements);
                Ok(())
            }
            Insn::ArrayHash { elements, .. } => {
                write!(f, "ArrayHash")?;
                write_separated!(f, " ", ", ", elements);
                Ok(())
            }
            Insn::ArrayInclude { elements, target, .. } => {
                write!(f, "ArrayInclude")?;
                write_separated!(f, " ", ", ", elements);
                write!(f, " | {target}")
            }
            Insn::ArrayPackBuffer { elements, fmt, buffer, .. } => {
                write!(f, "ArrayPackBuffer ")?;
                for element in elements {
                    write!(f, "{element}, ")?;
                }
                write!(f, "fmt: {fmt}")?;
                if let Some(buffer) = buffer {
                    write!(f, ", buf: {buffer}")?;
                }
                Ok(())
            }
            Insn::DupArrayInclude { ary, target, .. } => {
                write!(f, "DupArrayInclude {} | {}", ary.print(self.ptr_map), target)
            }
            Insn::ArrayDup { val, .. } => { write!(f, "ArrayDup {val}") }
            Insn::HashDup { val, .. } => { write!(f, "HashDup {val}") }
            Insn::HashAref { hash, key, .. } => { write!(f, "HashAref {hash}, {key}")}
            Insn::HashAset { hash, key, val, .. } => { write!(f, "HashAset {hash}, {key}, {val}")}
            Insn::ObjectAlloc { val, .. } => { write!(f, "ObjectAlloc {val}") }
            &Insn::ObjectAllocClass { class, .. } => {
                let class_name = get_class_name(class);
                write!(f, "ObjectAllocClass {class_name}:{}", class.print(self.ptr_map))
            }
            Insn::StringCopy { val, .. } => { write!(f, "StringCopy {val}") }
            Insn::StringConcat { strings, .. } => {
                write!(f, "StringConcat")?;
                write_separated!(f, " ", ", ", strings);
                Ok(())
            }
            Insn::StringGetbyte { string, index, .. } => {
                write!(f, "StringGetbyte {string}, {index}")
            }
            Insn::StringCoderangeOrScan { string, cached, .. } => {
                write!(f, "StringCoderangeOrScan {string}, {cached}")
            }
            Insn::StringSetbyteFixnum { string, index, value, .. } => {
                write!(f, "StringSetbyteFixnum {string}, {index}, {value}")
            }
            Insn::StringAppend { recv, other, .. } => {
                write!(f, "StringAppend {recv}, {other}")
            }
            Insn::StringAppendCodepoint { recv, other, .. } => {
                write!(f, "StringAppendCodepoint {recv}, {other}")
            }
            Insn::StringEqual { left, right } => {
                write!(f, "StringEqual {left}, {right}")
            }
            Insn::ToRegexp { values, opt, .. } => {
                write!(f, "ToRegexp")?;
                write_separated!(f, " ", ", ", values);

                let opt = *opt as u32;
                if opt != 0 {
                    write!(f, ", ")?;
                    let mut sep = "";
                    for (flag, name) in REGEXP_FLAGS {
                        if opt & flag != 0 {
                            write!(f, "{sep}{name}")?;
                            sep = "|";
                        }
                    }
                }

                Ok(())
            }
            Insn::Test { val } => { write!(f, "Test {val}") }
            Insn::IsMethodCfunc { val, cd, .. } => { write!(f, "IsMethodCFunc {val}, :{}", ruby_call_method_name(*cd)) }
            Insn::IsBitEqual { left, right } => write!(f, "IsBitEqual {left}, {right}"),
            Insn::IsBitNotEqual { left, right } => write!(f, "IsBitNotEqual {left}, {right}"),
            Insn::BoxBool { val } => write!(f, "BoxBool {val}"),
            Insn::BoxFixnum { val, .. } => write!(f, "BoxFixnum {val}"),
            Insn::UnboxFixnum { val } => write!(f, "UnboxFixnum {val}"),
            Insn::FixnumAref { recv, index } => write!(f, "FixnumAref {recv}, {index}"),
            Insn::Jump(target) => { write!(f, "Jump {target}") }
            Insn::CondBranch { val, if_true, if_false } => { write!(f, "CondBranch {val}, {if_true}, {if_false}") },
            Insn::SendDirect(insn) => {
                let SendDirectData { recv, cme, iseq, args, block, block_arg, jit_entry_idx, .. } = &**insn;
                let blockiseq = block.map(|bh| match bh { BlockHandler::BlockIseq(iseq) => iseq, BlockHandler::BlockArg => unreachable!() });
                let blockiseq_ptr = blockiseq.map_or(ptr::null(), |iseq| self.ptr_map.map_ptr(iseq));
                let method_name = unsafe { (**cme).called_id };
                write!(f, "SendDirect {recv}, {blockiseq_ptr:p}, :{method_name} ({:?})", self.ptr_map.map_ptr(*iseq))?;
                if let Some(block_arg) = block_arg {
                    write!(f, ", &{block_arg}")?;
                }
                if *jit_entry_idx != 0 {
                    write!(f, ", jit_entry_idx={jit_entry_idx}")?;
                }
                write_separated!(f, ", ", ", ", args);
                Ok(())
            }
            Insn::PushInlineFrame { recv, iseq, cme, num_args, captured: None, .. } => {
                let method_name = unsafe { (**cme).called_id };
                write!(f, "PushInlineFrame :{method_name}, {recv} ({:?})", self.ptr_map.map_ptr(*iseq))?;
                write!(f, ", num_args={num_args}")?;
                Ok(())
            }
            Insn::PushInlineFrame { recv, iseq, num_args, captured: Some(captured), .. } => {
                write!(f, "PushInlineBlockFrame ({:?}), {recv}, {captured}", self.ptr_map.map_ptr(*iseq))?;
                write!(f, ", num_args={num_args}")?;
                Ok(())
            }
            Insn::PopInlineFrame { .. } => {
                write!(f, "PopInlineFrame")
            }
            Insn::Send { recv, cd, args, block, reason, .. } => {
                // For tests, we want to check HIR snippets textually. Addresses change
                // between runs, making tests fail. Instead, pick an arbitrary hex value to
                // use as a "pointer" so we can check the rest of the HIR.
                match *block {
                    Some(BlockHandler::BlockIseq(blockiseq)) =>
                        write!(f, "Send {recv}, {:p}, :{}", self.ptr_map.map_ptr(blockiseq), ruby_call_method_name(*cd))?,
                    Some(BlockHandler::BlockArg) =>
                        write!(f, "Send {recv}, &block, :{}", ruby_call_method_name(*cd))?,
                    None =>
                        write!(f, "Send {recv}, :{}", ruby_call_method_name(*cd))?,
                }
                write_separated!(f, ", ", ", ", args);
                write!(f, " # SendFallbackReason: {reason}")?;
                Ok(())
            }
            Insn::SendForward { recv, cd, args, blockiseq, reason, .. } => {
                write!(f, "SendForward {recv}, {:p}, :{}", self.ptr_map.map_ptr(*blockiseq), ruby_call_method_name(*cd))?;
                write_separated!(f, ", ", ", ", args);
                write!(f, " # SendFallbackReason: {reason}")?;
                Ok(())
            }
            Insn::InvokeSuper { recv, blockiseq, args, reason, .. } => {
                write!(f, "InvokeSuper {recv}, {:p}", self.ptr_map.map_ptr(*blockiseq))?;
                write_separated!(f, ", ", ", ", args);
                write!(f, " # SendFallbackReason: {reason}")?;
                Ok(())
            }
            Insn::InvokeSuperForward { recv, blockiseq, args, reason, .. } => {
                write!(f, "InvokeSuperForward {recv}, {:p}", self.ptr_map.map_ptr(*blockiseq))?;
                write_separated!(f, ", ", ", ", args);
                write!(f, " # SendFallbackReason: {reason}")?;
                Ok(())
            }
            Insn::InvokeBlock { args, reason, .. } => {
                write!(f, "InvokeBlock")?;
                write_separated!(f, " ", ", ", args);
                write!(f, " # SendFallbackReason: {reason}")?;
                Ok(())
            }
            Insn::InvokeBlockIfunc { block_handler, args, .. } => {
                write!(f, "InvokeBlockIfunc {block_handler}")?;
                write_separated!(f, ", ", ", ", args);
                Ok(())
            }
            Insn::InvokeProc { recv, args, kw_splat, .. } => {
                write!(f, "InvokeProc {recv}")?;
                write_separated!(f, ", ", ", ", args);
                if *kw_splat {
                    write!(f, ", kw_splat")?;
                }
                Ok(())
            }
            Insn::InvokeBlockIseqDirect { iseq, captured, args, .. } => {
                write!(f, "InvokeBlockIseqDirect ({:?}), {captured}", self.ptr_map.map_ptr(*iseq))?;
                write_separated!(f, ", ", ", ", args);
                Ok(())
            }
            Insn::InvokeBuiltin { bf, args, leaf, .. } => {
                let bf_name = unsafe { CStr::from_ptr((**bf).name) }.to_str().unwrap();
                write!(f, "InvokeBuiltin{} {}",
                           if *leaf { " leaf" } else { "" },
                           // e.g. Code that use `Primitive.cexpr!`. From BUILTIN_INLINE_PREFIX.
                           if bf_name.starts_with("_bi") { "<inline_expr>" } else { bf_name })?;
                write_separated!(f, ", ", ", ", args);
                Ok(())
            }
            &Insn::EntryPoint { jit_entry_idx: Some(idx) } => write!(f, "EntryPoint JIT({idx})"),
            &Insn::EntryPoint { jit_entry_idx: None } => write!(f, "EntryPoint interpreter"),
            Insn::Return { val, pop_inlined_frames: 0 } => { write!(f, "Return {val}") }
            Insn::Return { val, pop_inlined_frames } => { write!(f, "Return {val} (pop {pop_inlined_frames} inlined frames)") }
            Insn::FixnumAdd  { left, right, .. } => { write!(f, "FixnumAdd {left}, {right}") },
            Insn::FixnumSub  { left, right, .. } => { write!(f, "FixnumSub {left}, {right}") },
            Insn::FixnumMult { left, right, .. } => { write!(f, "FixnumMult {left}, {right}") },
            Insn::FixnumDiv  { left, right, .. } => { write!(f, "FixnumDiv {left}, {right}") },
            Insn::FixnumMod  { left, right, .. } => { write!(f, "FixnumMod {left}, {right}") },
            Insn::FloatAdd   { recv, other, .. } => { write!(f, "FloatAdd {recv}, {other}") },
            Insn::FloatSub   { recv, other, .. } => { write!(f, "FloatSub {recv}, {other}") },
            Insn::FloatMul   { recv, other, .. } => { write!(f, "FloatMul {recv}, {other}") },
            Insn::FloatDiv   { recv, other, .. } => { write!(f, "FloatDiv {recv}, {other}") },
            Insn::FloatToInt { recv, .. } => { write!(f, "FloatToInt {recv}") },
            Insn::FloatLt    { left, right } => { write!(f, "FloatLt {left}, {right}") },
            Insn::FloatLe    { left, right } => { write!(f, "FloatLe {left}, {right}") },
            Insn::FloatGt    { left, right } => { write!(f, "FloatGt {left}, {right}") },
            Insn::FloatGe    { left, right } => { write!(f, "FloatGe {left}, {right}") },
            Insn::FixnumEq   { left, right, .. } => { write!(f, "FixnumEq {left}, {right}") },
            Insn::FixnumNeq  { left, right, .. } => { write!(f, "FixnumNeq {left}, {right}") },
            Insn::FixnumLt   { left, right, .. } => { write!(f, "FixnumLt {left}, {right}") },
            Insn::FixnumLe   { left, right, .. } => { write!(f, "FixnumLe {left}, {right}") },
            Insn::FixnumGt   { left, right, .. } => { write!(f, "FixnumGt {left}, {right}") },
            Insn::FixnumGe   { left, right, .. } => { write!(f, "FixnumGe {left}, {right}") },
            Insn::FixnumAnd  { left, right, .. } => { write!(f, "FixnumAnd {left}, {right}") },
            Insn::FixnumOr   { left, right, .. } => { write!(f, "FixnumOr {left}, {right}") },
            Insn::FixnumXor  { left, right, .. } => { write!(f, "FixnumXor {left}, {right}") },
            Insn::IntAnd     { left, right } => { write!(f, "IntAnd {left}, {right}") },
            Insn::IntOr      { left, right } => { write!(f, "IntOr {left}, {right}") },
            Insn::FixnumLShift { left, right, .. } => { write!(f, "FixnumLShift {left}, {right}") },
            Insn::FixnumRShift { left, right, .. } => { write!(f, "FixnumRShift {left}, {right}") },
            Insn::GuardType { val, guard_type, recompile, .. } => {
                write!(f, "GuardType {val}, {}", guard_type.print(self.ptr_map))?;
                if recompile.is_some() {
                    write!(f, " recompile")?;
                }
                return Ok(())
            },
            Insn::RefineType { val, new_type, .. } => { write!(f, "RefineType {val}, {}", new_type.print(self.ptr_map)) },
            Insn::HasType { val, expected, .. } => { write!(f, "HasType {val}, {}", expected.print(self.ptr_map)) },
            Insn::HasAncestor { val, class } => { write!(f, "HasAncestor {val}, {}", get_class_name(*class)) },
            Insn::GuardBitEquals { val, expected, recompile, .. } => {
                write!(f, "GuardBitEquals {val}, {}", expected.print(self.ptr_map))?;
                if recompile.is_some() {
                    write!(f, " recompile")?;
                }
                return Ok(())
            },
            Insn::GuardAnyBitSet { val, mask, mask_name, recompile, .. } => {
                let mask = mask.print(self.ptr_map);
                let recompile = if recompile.is_some() { " recompile" } else { "" };
                if let Some(name) = mask_name {
                    write!(f, "GuardAnyBitSet {val}, {name}={mask}{recompile}")
                } else {
                    write!(f, "GuardAnyBitSet {val}, {mask}{recompile}")
                }
            },
            Insn::GuardNoBitsSet { val, mask, mask_name: Some(name), .. } => { write!(f, "GuardNoBitsSet {val}, {name}={}", mask.print(self.ptr_map)) },
            Insn::GuardNoBitsSet { val, mask, .. } => { write!(f, "GuardNoBitsSet {val}, {}", mask.print(self.ptr_map)) },
            Insn::GuardNotRuby2KeywordsHash { val, recompile, .. } => {
                write!(f, "GuardNotRuby2KeywordsHash {val}")?;
                if recompile.is_some() {
                    write!(f, " recompile")?;
                }
                return Ok(())
            },
            Insn::GuardLess { left, right, .. } => write!(f, "GuardLess {left}, {right}"),
            Insn::GuardGreaterEq { left, right, recompile, .. } => {
                write!(f, "GuardGreaterEq {left}, {right}")?;
                if recompile.is_some() {
                    write!(f, " recompile")?;
                }
                Ok(())
            }
            &Insn::GetBlockParam { level, ep_offset, state, .. } => {
                let iseq = self.fun.map(|fun| fun.frame_state_iseq(state));
                let name = get_local_var_name_for_printer(iseq, level, ep_offset)
                    .map_or(String::new(), |x| format!("{x}, "));
                write!(f, "GetBlockParam {name}l{level}, EP@{ep_offset}")
            },
            Insn::PatchPoint { invariant, .. } => { write!(f, "PatchPoint {}", invariant.print(self.ptr_map)) },
            Insn::GetConstant { klass, id, allow_nil, .. } => {
                write!(f, "GetConstant {klass}, :{}, {allow_nil}", id.contents_lossy())
            }
            Insn::GetConstantPath { ic, .. } => { write!(f, "GetConstantPath {:p}", self.ptr_map.map_ptr(*ic)) },
            Insn::IsBlockGiven { block_handler } => { write!(f, "IsBlockGiven {block_handler}") },
            Insn::FixnumBitCheck {val, index} => { write!(f, "FixnumBitCheck {val}, {index}") },
            Insn::CCall { cfunc, recv, args, name, owner, return_type: _, elidable: _ } => {
                let display_name = if *owner == Qnil { name.contents_lossy().to_string() } else { qualified_method_name(*owner, *name) };
                write!(f, "CCall {recv}, :{}@{:p}", display_name, self.ptr_map.map_ptr(*cfunc))?;
                write_separated!(f, ", ", ", ", args);
                Ok(())
            },
            Insn::CCallWithFrame(insn) => {
                let CCallWithFrameData { cfunc, recv, args, name, cme, block, block_arg, .. } = &**insn;
                write!(f, "CCallWithFrame {recv}, :{}@{:p}", qualified_method_name(unsafe { (**cme).owner }, *name), self.ptr_map.map_ptr(*cfunc))?;
                write_separated!(f, ", ", ", ", args);
                match block {
                    Some(BlockHandler::BlockIseq(blockiseq)) =>
                        write!(f, ", block={:p}", self.ptr_map.map_ptr(*blockiseq))?,
                    Some(BlockHandler::BlockArg) =>
                        write!(f, ", block=&block")?,
                    None => {}
                }
                if let Some(block_arg) = block_arg {
                    write!(f, ", block=&{block_arg}")?;
                }
                Ok(())
            },
            Insn::CCallVariadic(insn) => {
                let CCallVariadicData { cfunc, recv, args, name, cme, block_arg, .. } = &**insn;
                write!(f, "CCallVariadic {recv}, :{}@{:p}", qualified_method_name(unsafe { (**cme).owner }, *name), self.ptr_map.map_ptr(*cfunc))?;
                write_separated!(f, ", ", ", ", args);
                if let Some(block_arg) = block_arg {
                    write!(f, ", block=&{block_arg}")?;
                }
                Ok(())
            },
            Insn::IncrCounterPtr { .. } => write!(f, "IncrCounterPtr"),
            Insn::Snapshot { state } => write!(f, "Snapshot {}", state.print(self.ptr_map)),
            Insn::Defined { op_type, v, .. } => {
                // op_type (enum defined_type) printing logic from iseq.c.
                // Not sure why rb_iseq_defined_string() isn't exhaustive.
                write!(f, "Defined ")?;
                let op_type = *op_type as u32;
                if op_type == DEFINED_FUNC {
                    write!(f, "func")?;
                } else if op_type == DEFINED_REF {
                    write!(f, "ref")?;
                } else if op_type == DEFINED_CONST_FROM {
                    write!(f, "constant-from")?;
                } else {
                    write!(f, "{}", String::from_utf8_lossy(unsafe { rb_iseq_defined_string(op_type).as_rstring_byte_slice().unwrap() }))?;
                };
                write!(f, ", {v}")
            }
            Insn::DefinedIvar { self_val, id, .. } => write!(f, "DefinedIvar {self_val}, :{}", id.contents_lossy()),
            Insn::GetIvar { self_val, id, .. } => write!(f, "GetIvar {self_val}, :{}", id.contents_lossy()),
            Insn::IvarReprofile { self_val, .. } => write!(f, "IvarReprofile {self_val}"),
            Insn::CheckMatch { target, pattern, flag, .. } => {
                const TYPE_MASK: u32 = 0x03;
                const ARRAY_FLAG: u32 = 0x04;

                let match_type = match *flag & TYPE_MASK {
                    VM_CHECKMATCH_TYPE_WHEN => "WHEN",
                    VM_CHECKMATCH_TYPE_CASE => "CASE",
                    VM_CHECKMATCH_TYPE_RESCUE => "RESCUE",
                    _ => return write!(f, "CheckMatch {target}, {pattern}, {flag}"),
                };
                let flag = if *flag & ARRAY_FLAG != 0 {
                    format!("{match_type}|ARRAY")
                } else {
                    match_type.to_string()
                };
                write!(f, "CheckMatch {target}, {pattern}, {flag}")
            }
            Insn::LoadPC => write!(f, "LoadPC"),
            Insn::LoadEC => write!(f, "LoadEC"),
            Insn::LoadSP => write!(f, "LoadSP"),
            &Insn::GetEP { level } => write!(f, "GetEP {level}"),
            Insn::LoadSelf => write!(f, "LoadSelf"),
            &Insn::LoadField { recv, id, offset, return_type: _, num_bits: _ } => {
                write!(f, "LoadField {recv}, :{id}@{:#x}", self.ptr_map.map_offset(offset))
            }
            &Insn::UnwrapSvar { val } => write!(f, "UnwrapSvar {val}"),
            &Insn::StoreField { recv, id, offset, val, num_bits: _ } => write!(f, "StoreField {recv}, :{id}@{:#x}, {val}", self.ptr_map.map_offset(offset)),
            &Insn::WriteBarrier { recv, val } => write!(f, "WriteBarrier {recv}, {val}"),
            Insn::SetIvar { self_val, id, val, .. } => write!(f, "SetIvar {self_val}, :{}, {val}", id.contents_lossy()),
            Insn::GetGlobal { id, .. } => write!(f, "GetGlobal :{}", id.contents_lossy()),
            Insn::SetGlobal { id, val, .. } => write!(f, "SetGlobal :{}, {val}", id.contents_lossy()),
            &Insn::IsBlockParamModified { flags } => {
                write!(f, "IsBlockParamModified {flags}")
            },
            &Insn::SetLocal { val, level, ep_offset, state } => {
                let iseq = self.fun.map(|fun| fun.frame_state_iseq(state));
                let name = get_local_var_name_for_printer(iseq, level, ep_offset).map_or(String::new(), |x| format!("{x}, "));
                write!(f, "SetLocal {name}l{level}, EP@{ep_offset}, {val}")
            },
            Insn::GetSpecialSymbol { symbol_type, .. } => write!(f, "GetSpecialSymbol {symbol_type:?}"),
            Insn::GetSpecialNumber { nth, .. } => write!(f, "GetSpecialNumber {nth}"),
            Insn::Once { body_iseq, .. } => write!(f, "Once {}", iseq_name(*body_iseq)),
            Insn::GetClassVar { id, .. } => write!(f, "GetClassVar :{}", id.contents_lossy()),
            Insn::SetClassVar { id, val, .. } => write!(f, "SetClassVar :{}, {val}", id.contents_lossy()),
            Insn::ToArray { val, .. } => write!(f, "ToArray {val}"),
            Insn::ToHash { val, .. } => write!(f, "ToHash {val}"),
            Insn::CheckArrayType { val, .. } => write!(f, "CheckArrayType {val}"),
            Insn::ToAryForExpand { val, .. } => write!(f, "ToAryForExpand {val}"),
            Insn::ToNewArray { val, .. } => write!(f, "ToNewArray {val}"),
            Insn::ArrayExtend { left, right, .. } => write!(f, "ArrayExtend {left}, {right}"),
            Insn::ArrayPush { array, val, .. } => write!(f, "ArrayPush {array}, {val}"),
            Insn::StringIntern { val, .. } => { write!(f, "StringIntern {val}") },
            Insn::AnyToString { val, .. } => { write!(f, "AnyToString {val}") },
            Insn::SideExit { reason, recompile, .. } => {
                if recompile.is_some() {
                    write!(f, "SideExit {reason} recompile")
                } else {
                    write!(f, "SideExit {reason}")
                }
            }
            Insn::PutSpecialObject { value_type, .. } => write!(f, "PutSpecialObject {value_type}"),
            Insn::Throw { throw_state, val, .. } => {
                write!(f, "Throw ")?;
                match throw_state & VM_THROW_STATE_MASK {
                    RUBY_TAG_NONE   => write!(f, "TAG_NONE"),
                    RUBY_TAG_RETURN => write!(f, "TAG_RETURN"),
                    RUBY_TAG_BREAK  => write!(f, "TAG_BREAK"),
                    RUBY_TAG_NEXT   => write!(f, "TAG_NEXT"),
                    RUBY_TAG_RETRY  => write!(f, "TAG_RETRY"),
                    RUBY_TAG_REDO   => write!(f, "TAG_REDO"),
                    RUBY_TAG_RAISE  => write!(f, "TAG_RAISE"),
                    RUBY_TAG_THROW  => write!(f, "TAG_THROW"),
                    RUBY_TAG_FATAL  => write!(f, "TAG_FATAL"),
                    tag => write!(f, "{tag}")
                }?;
                if throw_state & VM_THROW_NO_ESCAPE_FLAG != 0 {
                    write!(f, "|NO_ESCAPE")?;
                }
                write!(f, ", {val}")
            }
            Insn::IncrCounter(counter) => write!(f, "IncrCounter {counter:?}"),
            Insn::CheckInterrupts { .. } => write!(f, "CheckInterrupts"),
            Insn::IsA { val, class } => write!(f, "IsA {val}, {class}"),
            Insn::BreakPoint => write!(f, "BreakPoint"),
            Insn::Unreachable => write!(f, "Unreachable"),
        }
    }
}

impl std::fmt::Display for Insn {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.print(&PtrPrintMap::identity(), None).fmt(f)
    }
}

/// A basic block in a [`Function`].
#[derive(Default, Debug)]
pub struct Block {
    /// The index of the first YARV instruction for the Block in the ISEQ
    pub insn_idx: u32,
    params: Vec<InsnId>,
    insns: Vec<InsnId>,
}

impl Block {
    /// Return an iterator over params
    pub fn params(&self) -> Iter<'_, InsnId> {
        self.params.iter()
    }

    /// Return an iterator over insns
    pub fn insns(&self) -> Iter<'_, InsnId> {
        self.insns.iter()
    }
}

/// Pretty printer for [`Function`].
pub struct FunctionPrinter<'a> {
    fun: &'a Function,
    display_snapshot_and_tp_patchpoints: bool,
    ptr_map: PtrPrintMap,
}

impl<'a> FunctionPrinter<'a> {
    pub fn without_snapshot(fun: &'a Function) -> Self {
        let mut ptr_map = PtrPrintMap::identity();
        if cfg!(test) {
            ptr_map.map_ptrs = true;
        }
        Self { fun, display_snapshot_and_tp_patchpoints: false, ptr_map }
    }

    pub fn with_snapshot(fun: &'a Function) -> FunctionPrinter<'a> {
        let mut printer = Self::without_snapshot(fun);
        printer.display_snapshot_and_tp_patchpoints = true;
        printer
    }
}

/// Union-Find (Disjoint-Set) is a data structure for managing disjoint sets that has an interface
/// of two operations:
///
/// * find (what set is this item part of?)
/// * union (join these two sets)
///
/// Union-Find identifies sets by their *representative*, which is some chosen element of the set.
/// This is implemented by structuring each set as its own graph component with the representative
/// pointing at nothing. For example:
///
/// * A -> B -> C
/// * D -> E
///
/// This represents two sets `C` and `E`, with three and two members, respectively. In this
/// example, `find(A)=C`, `find(C)=C`, `find(D)=E`, and so on.
///
/// To union sets, call `make_equal_to` on any set element. That is, `make_equal_to(A, D)` and
/// `make_equal_to(B, E)` have the same result: the two sets are joined into the same graph
/// component. After this operation, calling `find` on any element will return `E`.
///
/// This is a useful data structure in compilers because it allows in-place rewriting without
/// linking/unlinking instructions and without replacing all uses. When calling `make_equal_to` on
/// any instruction, all of its uses now implicitly point to the replacement.
///
/// This does mean that pattern matching and analysis of the instruction graph must be careful to
/// call `find` whenever it is inspecting an instruction (or its operands). If not, this may result
/// in missing optimizations.
/// The forwarding pointers live in `Cell`s so that `find` can compress paths
/// through a shared reference. Wrapping the whole table in a `RefCell` instead
/// meant every one of the compiler's millions of `type_of`/`resolve` lookups
/// paid a borrow-flag round trip, and `type_of` had to take the *mutable*
/// borrow even though all it wanted was a lookup.
#[derive(Debug)]
struct UnionFind<T: Copy + Into<usize>> {
    forwarded: Vec<Cell<T>>,
}

impl<T: Copy + Into<usize> + PartialEq + std::convert::From<usize>> UnionFind<T> {
    fn new() -> UnionFind<T> {
        UnionFind { forwarded: vec![] }
    }

    /// Private. Return the internal representation of the forwarding pointer for a given element.
    fn at(&self, idx: T) -> T {
        self.forwarded.get(idx.into()).map_or(idx, Cell::get)
    }

    /// Private. Set the internal representation of the forwarding pointer for the given element
    /// `idx`. Extend the internal vector if necessary.
    fn set(&mut self, idx: T, value: T) {
        if idx.into() >= self.forwarded.len() {
            for i in self.forwarded.len()..=idx.into() {
                self.forwarded.push(Cell::new(i.into()));
            }
        }
        self.forwarded[idx.into()].set(value);
    }

    /// Find the set representative for `insn`. Perform path compression at the same time to speed
    /// up further find operations. For example, before:
    ///
    /// `A -> B -> C`
    ///
    /// and after `find(A)`:
    ///
    /// ```text
    /// A -> C
    /// B ---^
    /// ```
    pub fn find(&self, insn: T) -> T {
        let result = self.find_const(insn);
        if result != insn {
            // Path compression. `result != insn` means `at(insn)` found a
            // forwarding entry, so `insn` is in range and no growth is needed.
            self.forwarded[insn.into()].set(result);
        }
        result
    }

    /// Find the set representative for `insn` without doing path compression.
    fn find_const(&self, insn: T) -> T {
        let mut result = insn;
        loop {
            let found = self.at(result);
            if found == result { return found; }
            result = found;
        }
    }

    /// Union the two sets containing `insn` and `target` such that every element in `insn`s set is
    /// now part of `target`'s. Neither argument must be the representative in its set.
    pub fn make_equal_to(&mut self, insn: T, target: T) {
        let insn = self.find(insn);
        let target = self.find(target);
        self.set(insn, target);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ValidationError {
    BlockHasNoTerminator(BlockId),
    // The terminator and its actual position
    TerminatorNotAtEnd(BlockId, InsnId, usize),
    /// Expected length, actual length
    MismatchedBlockArity(BlockId, usize, usize),
    JumpTargetNotInRPO(BlockId),
    // The offending instruction, and the operand
    OperandNotDefined(BlockId, InsnId, InsnId),
    /// The offending block and instruction
    DuplicateInstruction(BlockId, InsnId),
    /// The offending instruction, its operand, expected type string, actual type string
    MismatchedOperandType(InsnId, InsnId, String, String),
    MiscValidationError(InsnId, String),
}

/// Call-site features that keep a `def foo(...)` callee on the interpreter's argument setup.
///
/// Everything here changes what the callee's `...` local has to describe, so passing the
/// call site's own callinfo through would misrepresent the arguments we copied:
/// `VM_CALL_FORWARDING` means the caller is itself forwarding and the callinfo to hand over
/// lives in *its* `...` local rather than at this call site, `VM_CALL_ARGS_BLOCKARG` puts a
/// block on the stack that `vm_caller_setup_arg_block` pops before argument setup, splats
/// and `**kwrest` mean the stack does not hold `vm_ci_argc(ci)` plain values, and
/// super/`__send__`/tailcall reach the callee through a different setup path entirely.
const FORWARDABLE_CALLEE_BLOCKERS: u32 = VM_CALL_ARGS_SPLAT | VM_CALL_KW_SPLAT
    | VM_CALL_ARGS_BLOCKARG | VM_CALL_FORWARDING | VM_CALL_TAILCALL
    | VM_CALL_OPT_SEND | VM_CALL_SUPER | VM_CALL_ZSUPER;

/// Check if we can emit SendDirect to a `def foo(...)` (forwardable) ISEQ.
///
/// `vm_call_iseq_forwardable` does not run `setup_parameters_complex` at all: it grows the
/// callee frame by `vm_ci_argc(ci)`, leaves the caller's arguments where they already are,
/// and stores the call site's callinfo in the `...` local so a later `sendforward` can
/// replay the call. [`super::codegen::gen_send_iseq_direct`] can do the same, because the
/// argument count is a property of the call site and therefore known at compile time.
///
/// A literal block needs no check of its own: `optimized_forward` in `iseq_set_arguments`
/// gives a forwardable ISEQ no block parameter and always sets `use_block`, so the frame's
/// specval that SendDirect already writes is the whole of it.
fn can_direct_send_forwardable(ci: *const rb_callinfo, args: &[InsnId]) -> Result<(), SendDirectFailure> {
    use Counter::*;
    let ci_flags = unsafe { rb_vm_ci_flag(ci) };
    if ci_flags & FORWARDABLE_CALLEE_BLOCKERS != 0 {
        return Err(SendDirectFailure::with_counters(
            ComplexArgPass,
            vec![complex_arg_pass_param_forwardable],
        ));
    }
    // The frame is grown by exactly the call site's argument count, so the HIR argument
    // list has to be the whole of it. Keyword arguments count here: they stay on the stack
    // as plain values and the callinfo records their names.
    if args.len() != unsafe { rb_vm_ci_argc(ci) } as usize {
        return Err(SendDirectFailure::new(ArgcParamMismatch));
    }
    // `IseqCall` stores argc as u16, and the callee frame has to fit the copied arguments.
    if u16::try_from(args.len()).is_err() {
        return Err(SendDirectFailure::new(OperandTooLarge));
    }
    Ok(())
}

/// Check if we can emit SendDirect to the given ISEQ with the given arguments.
/// `block_arg_passthrough` says the caller's `&blk` argument was taken out of `args` to become
/// the callee frame's block handler, so the frame setup reproduces `vm_caller_setup_arg_block`
/// for it and the call site's block-arg flag no longer blocks a direct send.
fn can_direct_send(iseq: *const rb_iseq_t, ci: *const rb_callinfo, args: &[InsnId], has_block: bool, block_arg_passthrough: bool) -> Result<(), SendDirectFailure> {
    let mut complex_arg_counters = vec![];
    let mut count_failure = |counter| complex_arg_counters.push(counter);
    let params = unsafe { iseq.params() };

    let callee_has_block_param = 0 != params.flags.has_block();
    let caller_passes_block_arg = has_block && !block_arg_passthrough
        && (unsafe { rb_vm_ci_flag(ci) } & VM_CALL_ARGS_BLOCKARG) != 0;

    use Counter::*;
    if 0 != params.flags.forwardable() {
        return can_direct_send_forwardable(ci, args);
    }
    if callee_has_block_param && caller_passes_block_arg
                                       { count_failure(complex_arg_pass_param_block) }
    // A `**rest` parameter collects the caller keywords the callee's keyword table does not
    // name, which `plan_send_direct_keyword_arguments` can build as one more Hash argument.
    // `ruby2_keywords` is left out: it needs the VM to move RHASH_PASS_AS_KEYWORDS across
    // the call.
    let has_kwrest = 0 != params.flags.has_kwrest();
    if has_kwrest && 0 != params.flags.ruby2_keywords()
                                       { count_failure(complex_arg_pass_param_kwrest) }

    // If the caller passes a block (literal or &block), we need to fall back to the
    // interpreter for two cases it handles that we don't:
    // 1. Methods with &nil reject blocks with ArgumentError
    // 2. Methods that don't use blocks emit "unused block" warnings
    let caller_passes_block = has_block || caller_passes_block_arg;
    if caller_passes_block && 0 != params.flags.accepts_no_block()
                                       { count_failure(complex_arg_pass_accepts_no_block) }
    if caller_passes_block && 0 == params.flags.use_block()
                                       { count_failure(complex_arg_pass_does_not_use_block) }

    if !complex_arg_counters.is_empty() {
        return Err(SendDirectFailure::with_counters(
            ComplexArgPass,
            complex_arg_counters,
        ));
    }

    let lead_num = params.lead_num;
    let opt_num = params.opt_num;
    let post_num = params.post_num;
    let keyword = params.keyword;
    let kw_req_num = if keyword.is_null() { 0 } else { unsafe { (*keyword).required_num } };
    let kw_total_num = if keyword.is_null() { 0 } else { unsafe { (*keyword).num } };
    let kwarg = unsafe { rb_vm_ci_kwarg(ci) };
    let caller_kw_count = if kwarg.is_null() { 0 } else { (unsafe { get_cikw_keyword_len(kwarg) }) as usize };
    let has_rest = 0 != params.flags.has_rest();
    let caller_positional = match args.len().checked_sub(caller_kw_count) {
        Some(count) => count,
        None => {
            return Err(SendDirectFailure::new(ArgcParamMismatch));
        }
    };

    // Match vm_args.c's setup_parameters_complex via args_kw_argv_to_hash:
    // keywords passed to a method with no keyword parameters can become one
    // positional hash before the argument count check.
    let keywords_as_positional_hash = caller_kw_count != 0 && keyword.is_null();
    let effective_positional = caller_positional + usize::from(keywords_as_positional_hash);
    let caller_positional_i32 = match c_int::try_from(effective_positional) {
        Ok(argc) => argc,
        Err(_) => {
            return Err(SendDirectFailure::new(ArgcParamMismatch));
        }
    };
    let min_positional = lead_num + post_num;
    let positional_ok = if has_rest {
        (min_positional..).contains(&caller_positional_i32)
    } else {
        (min_positional..=min_positional + opt_num).contains(&caller_positional_i32)
    };
    if !positional_ok {
        return Err(SendDirectFailure::new(ArgcParamMismatch));
    }

    // Plain keyword-to-positional-hash is safe to synthesize below. Keep VM
    // dispatch for callee modes that need keyword-sensitive handling: **nil
    // rejection and ruby2_keywords flag preservation.
    if keywords_as_positional_hash
        && (params.flags.accepts_no_kwarg() != 0 || params.flags.ruby2_keywords() != 0)
    {
        return Err(SendDirectFailure::with_counters(
            ComplexArgPass,
            vec![complex_arg_pass_keyword_to_positional_hash],
        ));
    }

    // After keyword-to-positional-hash, SendDirect receives no keyword slots;
    // the caller keywords are represented by one extra positional Hash.
    let effective_keyword_count = if keywords_as_positional_hash { 0 } else { caller_kw_count };
    // With `**rest` the caller may name more keywords than the table has; the extras end up
    // in the Hash. Every required keyword still has to be there, which the planning below
    // rechecks by name.
    let keyword_ok = c_int::try_from(effective_keyword_count)
        .as_ref()
        .map(|argc| if has_kwrest { (kw_req_num..).contains(argc) } else { (kw_req_num..=kw_total_num).contains(argc) })
        .unwrap_or(false);
    if !keyword_ok {
        return Err(SendDirectFailure::new(ArgcParamMismatch));
    }

    // Compute the final argc after keyword setup and rest packing:
    // send_argc = packed positional args + callee's total keywords (all kw slots are filled).
    // With *rest, SendDirect receives one rest-array slot instead of each rest
    // element, so cap positional argc at required/post + filled opts + rest slot.
    let passed_opt_num = (caller_positional_i32 - min_positional).min(opt_num) as usize;
    // Without *rest, use the converted positional count so the synthesized
    // keyword Hash is included in the SendDirect argument count.
    let send_positional_argc = if has_rest { min_positional as usize + passed_opt_num + 1 } else { effective_positional };
    let send_argc = send_positional_argc + kw_total_num as usize + usize::from(has_kwrest);

    // IseqCall stores the JIT entry index and argc as u16.
    if u16::try_from(send_argc).is_err() {
        return Err(SendDirectFailure::new(OperandTooLarge));
    }

    Ok(())
}

/// The one lowering an `expandarray` site is compiled for. `vm_expandarray()` handles arrays,
/// values with a `#to_ary`, values without one, and arrays that are too short, but a given site
/// almost always sees just one of those, so we compile for that case and recompile if it changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpandArrayShape {
    /// The value is an `Array` long enough for its targets: read the elements straight out of it.
    Array,
    /// The value has no `#to_ary`, so it destructures as the one-element array `[value]`.
    Scalar,
    /// Anything could show up (or we can't recompile): go through the same conversion
    /// `vm_expandarray()` does and nil-fill out-of-bounds reads. Never side-exits.
    General,
    /// The site never ran while profiling, so we don't know its shape yet.
    NoProfile,
}

/// Policy that controls how optimization passes generate code.
/// Determined at compile time based on the ISEQ's compilation history.
#[derive(Debug)]
struct CompilePolicy {
    /// When true, optimization passes should avoid generating guards that
    /// side-exit, and instead use fallback paths (e.g. C calls) on mismatch.
    /// Set when this is the final version of an ISEQ after recompilation.
    no_side_exits: bool,
}

#[derive(Debug, Clone, Copy)]
struct SetIvarSpec {
    profiled_type: ProfiledType,
    ivar_index: attr_index_t,
    next_shape: ShapeId,
}

/// How a shape dispatch reacts to a receiver whose shape is not one of the profiled ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShapeMiss {
    /// Guard on the profiled shape and side-exit, so the ISEQ can be recompiled around the
    /// shape it actually sees. Only valid where the profiled shape is a real prediction for
    /// this program point.
    SideExit,
    /// Branch on the profiled shape and let the generic C helper handle everything else. Costs
    /// a compare and a branch on the fast path, but never leaves JIT code. The miss is worth
    /// recording so that a later version of the ISEQ can add the shape.
    CallFallback,
    /// Same, but the miss must not be recorded. Recording it makes the ISEQ re-profile, which
    /// throws away the type profile the dispatch arms were built from — a bad trade when the
    /// profiled shape was never a prediction for this program point in the first place.
    CallFallbackWithoutReprofile,
}

impl ShapeMiss {
    fn calls_fallback(self) -> bool {
        !matches!(self, ShapeMiss::SideExit)
    }

    /// Whether a shape the profile genuinely predicts may still be guarded with a side exit,
    /// even though a *chain* miss calls the fallback. Two shape dispatches are predictions the
    /// profiler can act on: a site with no profile at all (there is nothing to build a chain
    /// out of until it exits and gets one) and a site whose single profiled shape accounts for
    /// every receiver the profiler saw. Everything else is a shape-polymorphic site, where an
    /// exit would only rebuild the same chain out of the same buckets.
    ///
    /// `CallFallbackWithoutReprofile` says the profiled shape was never a prediction for this
    /// program point, so nothing about it is worth an exit.
    fn speculates_on_predicted_shape(self) -> bool {
        !matches!(self, ShapeMiss::CallFallbackWithoutReprofile)
    }
}

impl CompilePolicy {
    fn new(iseq: *const rb_iseq_t) -> Self {
        // When a previous version was invalidated and we've reached the version
        // limit, avoid speculative optimizations that may side-exit.
        let no_side_exits = if iseq.is_null() {
            false
        } else {
            let payload = get_or_create_iseq_payload(iseq);
            payload.versions.iter().any(
                |v| unsafe { v.as_ref() }.is_invalidated()
            ) && payload.versions.len() + 1 >= payload.version_limit()
        };
        Self { no_side_exits }
    }
}

/// A wrapper around [`InsnId`] that indicates the instruction's operands have been resolved.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedInsnId(pub InsnId);

impl ResolvedInsnId {
    /// Return a mutable reference to the instruction at `insn_id` (after resolving via
    /// union-find). Assumes the operands are resolved through union-find already. Use
    /// [`Function::resolve`] to resolve operands before calling this.
    pub fn insn_mut(self, fun: &mut Function) -> &mut Insn {
        &mut fun.insns[self.0.to_usize()]
    }

    /// Return a reference to the instruction at `insn_id` (after resolving via union-find).
    /// Assumes the operands are resolved through union-find already. Use [`Function::resolve`] to
    /// resolve operands before calling this.
    pub fn insn(self, fun: &Function) -> &Insn {
        &fun.insns[self.0.to_usize()]
    }
}

/// A [`Function`], which is analogous to a Ruby ISeq, is a control-flow graph of [`Block`]s
/// containing instructions.
#[derive(Debug)]
pub struct Function {
    // ISEQ this function refers to
    iseq: *const rb_iseq_t,
    /// Whether previously, a function for this ISEQ was invalidated due to
    /// singleton class creation (violation of NoSingletonClass invariant).
    was_invalidated_for_singleton_class_creation: bool,
    /// Whether `self` is guaranteed to be a heap (non-immediate) object. When set,
    /// the `self`-producing instructions (`LoadSelf` and the `SelfParam` `LoadArg`)
    /// are typed `HeapBasicObject` instead of `BasicObject`. Sourced from
    /// `IseqPayload::self_is_heap_object`.
    self_is_heap_object: bool,
    /// Controls code generation strategy for optimization passes.
    policy: CompilePolicy,
    /// The types for the parameters of this function. They are copied to the type
    /// of entry block params after infer_types() fills Empty to all insn_types.
    param_types: Vec<Type>,

    insns: Vec<Insn>,
    union_find: UnionFind<InsnId>,
    insn_types: Vec<Type>,
    blocks: Vec<Block>,
    /// Superblock that targets all entry blocks. The sole root for RPO/dominator computation.
    pub entries_block: BlockId,
    /// Entry block for the interpreter
    entry_block: BlockId,
    /// Entry block for JIT-to-JIT calls. Length will be `opt_num+1`, for callers
    /// fulfilling `(0..=opt_num)` optional parameters.
    jit_entry_blocks: Vec<BlockId>,
    profiles: Option<ProfileOracle>,
    /// Profiled types (class *and* shape) of values that have been guarded or refined to the
    /// class of that profiled type. `Type` only records the class, so this keeps the shape
    /// reachable for later specializations of the same value.
    ///
    /// This matters most for inlined callees: an ISEQ only starts profiling once the interpreter
    /// enters it `rb_zjit_profile_threshold` times, so a method that is only ever reached from
    /// JIT code has no profile of its own. When it is inlined, its `self` is the caller's guarded
    /// receiver, and this map is the only place its shape survives.
    guarded_profiled_types: HashMap<InsnId, ProfiledType>,
    /// Sends that sit behind a [`Insn::HasAncestor`] guard, mapped to the class that guard
    /// checked. `type_specialize` dispatches these on the method resolved from that class
    /// instead of on the receiver's exact class. Keyed by the `Send`'s own instruction ID;
    /// IDs are never reused, so a stale entry can only ever be read back by the same send.
    ancestor_dispatch: HashMap<InsnId, VALUE>,
    /// Rough estimate for the number of (actually executable) instructions in the function. Does
    /// not count Snapshot, PatchPoint, etc.
    /// Currently updated by `infer_types` as a heuristic but that is not a guarantee.
    num_instructions: usize,
    /// How many callees this function has inlined past its cumulative inlining budget because
    /// doing so let a `yield` inside them dispatch directly. Capped at
    /// [`MAX_YIELD_INLINE_BONUSES`].
    yield_inline_bonuses: usize,
    /// Whether the current [`Self::inline_methods`] pass found the function already past its
    /// cumulative inlining budget. Only yield-unlocking callees are considered from then on.
    inline_budget_exhausted: bool,
    /// Set when this function is compiled as an exception handler entry
    /// (`jit_exec_exception`). The interpreter entry block then starts in the
    /// middle of the ISEQ instead of at an opt-table entry. See
    /// [`ExceptionEntry`].
    exception_entry: Option<ExceptionEntry>,
    /// For `Send` instructions on a `send`/`__send__` call site that HIR build has already
    /// guarded to one method name, the ID of that method. `type_specialize` uses it (and
    /// drops the leading method-name argument) to resolve the call like a direct call.
    send_mid_overrides: HashMap<InsnId, ID>,
}

/// Where an exception-handler entry (`body->jit_exception`) resumes the ISEQ.
/// The interpreter jumps to a catch-table continuation, so both the resume PC
/// and the VM stack depth at that PC are fixed at compile time; the generated
/// entry guards `cfp->pc` to make sure a different continuation doesn't run
/// this code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExceptionEntry {
    /// Instruction index in the ISEQ where execution resumes
    pub insn_idx: u32,
    /// Number of VM stack slots live at `insn_idx`, i.e. `cfp->sp - vm_base_ptr(cfp)`
    pub stack_size: usize,
}

/// Arguments and metadata prepared for lowering an ISEQ call to `SendDirect`.
struct SendDirectArgs {
    state: InsnId,
    args: Vec<InsnId>,
    kw_bits: u32,
    jit_entry_idx: u16,
}

/// One SendDirect argument before its HIR value is materialized.
enum SendDirectArg {
    /// A HIR value already present in the original Send argument vector.
    Existing(InsnId),
    /// A Ruby value to materialize as a Const instruction on the selected path.
    Constant(VALUE),
    /// Explicit caller keywords to materialize as one positional Hash.
    KeywordHash(Vec<SendDirectArg>),
    /// Values to materialize as the Array passed to a callee rest parameter.
    RestArray(Vec<SendDirectArg>),
}

/// Side-effect-free SendDirect argument setup before HIR emission.
struct SendDirectCall {
    /// Argument slots in the shape expected by SendDirect.
    args: Vec<SendDirectArg>,
    /// Optional keyword slots omitted by the caller.
    kw_bits: u32,
    /// Optional positional entry selected from the caller's argument count.
    jit_entry_idx: u16,
}

/// Why a SendDirect call could not be built, including feature counters that
/// should only be recorded when the caller keeps the dynamic fallback.
struct SendDirectFailure {
    reason: SendFallbackReason,
    counters: Vec<Counter>,
}

/// Call context in which SendDirect argument planning failed.
enum SendDirectFallbackContext {
    Send,
    Super,
}

impl SendDirectFailure {
    fn new(reason: SendFallbackReason) -> Self {
        Self { reason, counters: vec![] }
    }

    fn with_counters(reason: SendFallbackReason, counters: Vec<Counter>) -> Self {
        Self { reason, counters }
    }

    fn record(
        &self,
        function: &mut Function,
        block: BlockId,
        send_insn: InsnId,
        context: SendDirectFallbackContext,
    ) {
        for &counter in &self.counters {
            function.count(block, counter);
        }
        let context_counter = match context {
            SendDirectFallbackContext::Send => Counter::send_direct_fallback_context_send,
            SendDirectFallbackContext::Super => Counter::send_direct_fallback_context_super,
        };
        function.count(block, context_counter);
        function.set_dynamic_send_reason(send_insn, self.reason);
    }
}

unsafe extern "C" {
    fn rb_simple_iseq_p(iseq: IseqPtr) -> bool;
}

/// Minimum share of the profiled executions of a call site that a guard chain must cover before
/// we emit it. Each arm costs a comparison and a branch on every execution, and the arms we can
/// emit only cover part of the profile: handlers or receiver types we cannot specialize, and
/// samples that did not fit in the profile at all, take the fallback. A chain built out of the
/// cold tail of the profile pays for its guards on every execution and still performs the same
/// dynamic dispatch, so require the covered share to be at least this large.
const CHAIN_COVERAGE_THRESHOLD: f64 = 0.5;

/// Maximum number of classes we scan below a class while proving that none of them overrides
/// the called method. Hierarchies larger than this give up on the ancestor guard rather than
/// spend unbounded compile time on the proof.
///
/// It has to be large enough to cover the whole process: the most valuable roots are the
/// shallow ones. `Object#class`, `Object#is_a?` and `Object#respond_to?` are the methods
/// rubocop's megamorphic sites resolve most often, and proving them unoverridden means walking
/// every class the process has loaded (~3,200 for rubocop).
const ANCESTOR_GUARD_MAX_SUBCLASSES: u32 = 100_000;

/// A call site that is megamorphic in the receiver's class but monomorphic in the method it
/// resolves to: every profiled receiver class inherits one shared method entry.
#[derive(Clone, Copy)]
struct AncestorDispatch {
    /// Class that roots the hierarchy the shared method entry is visible from. Guarding that
    /// the receiver is an instance of this class or of a subclass is enough to know which
    /// method the call resolves to.
    klass: VALUE,
}

/// How to dispatch a call site whose receiver profile has more than one class in it.
enum SendChainPlan {
    /// Guard that the receiver inherits from one class and call the method it resolves there.
    Ancestor(AncestorDispatch),
    /// Guard each profiled receiver class in turn.
    Classes(TypeDistributionSummary),
}

/// Decide whether a megamorphic call site can dispatch on the method it resolves to rather than
/// on the receiver's class.
///
/// A site over 100 AST node classes that all inherit one `Node#send_type?` is megamorphic in the
/// receiver class but has a single call target. A class chain can only ever cover the classes
/// that fit in the profile, and pays a comparison per arm to reach the same method; guarding
/// "the receiver inherits from `Node`" covers every subclass, including ones the profile never
/// saw, in one check.
///
/// The guard alone doesn't pin the target down: a subclass could override the method. So we also
/// prove at compile time that no class below the defining class defines the method, and register
/// that as an invariant ([`Invariant::NoMethodOverride`]) so a later definition invalidates the
/// code.
fn ancestor_dispatch_target(summary: &TypeDistributionSummary, cd: *const rb_call_data) -> Option<AncestorDispatch> {
    // The guard reads the prime classext of the receiver's class.
    if invariants::non_root_box_created() {
        incr_counter!(send_ancestor_guard_reject_not_a_class);
        return None;
    }
    let ci = unsafe { (*cd).ci };
    let mid = unsafe { vm_ci_mid(ci) };

    // Every bucket has to resolve the method to the same entry. That is the evidence that the
    // site is really dispatching one method: if the buckets disagree, an ancestor guard would
    // send some of them to the wrong place, and we would need the class chain anyway.
    let mut shared_cme: Option<*const rb_callable_method_entry_t> = None;
    for &profiled_type in summary.buckets() {
        if profiled_type.is_empty() { break; }
        // The guard below only looks at the class of heap objects; an immediate receiver takes
        // the fallthrough no matter what, so a profile with immediates in it wants a class chain.
        if profiled_type.flags().is_immediate() {
            incr_counter!(send_ancestor_guard_reject_immediate);
            return None;
        }
        let cme = unsafe { rb_callable_method_entry(profiled_type.class(), mid) };
        if cme.is_null() { return None; }
        match shared_cme {
            None => shared_cme = Some(cme),
            Some(seen) if seen == cme => {}
            Some(_) => {
                incr_counter!(send_ancestor_guard_reject_cme_differs);
                return None;
            }
        }
    }
    let cme = shared_cme?;

    // Root the hierarchy at the class the method is visible from. For a method defined in a
    // class that is the defining class itself; for one defined in a module it is the ICLASS the
    // include created, whose includer is the class that gained the method.
    let defined_class = unsafe { (*cme).defined_class };
    if defined_class == VALUE(0) || defined_class.special_const_p() {
        incr_counter!(send_ancestor_guard_reject_not_a_class);
        return None;
    }
    let klass = match defined_class.builtin_type() {
        RUBY_T_CLASS => defined_class,
        RUBY_T_ICLASS => {
            let includer = unsafe { rb_zjit_iclass_includer(defined_class) };
            if includer == VALUE(0) || includer.special_const_p() || includer.builtin_type() != RUBY_T_CLASS {
                incr_counter!(send_ancestor_guard_reject_not_a_class);
                return None;
            }
            includer
        }
        _ => {
            incr_counter!(send_ancestor_guard_reject_not_a_class);
            return None;
        }
    };

    // The lookup from the root has to land on the same entry, both because the generated call
    // targets it and because `type_specialize` re-resolves from the root.
    if unsafe { rb_callable_method_entry(klass, mid) } != cme {
        incr_counter!(send_ancestor_guard_reject_cme_differs);
        return None;
    }

    // The guard reads the cached superclass array, which is only filled in for classes whose
    // ancestry is fully built and whose depth didn't saturate.
    // RCLASS_MAX_SUPERCLASS_DEPTH: past it the depth saturates and stops identifying a slot
    // in the array, which is what the guard indexes.
    let depth = unsafe { rb_zjit_class_superclass_depth(klass) };
    if depth >= u16::MAX as std::os::raw::c_uint {
        incr_counter!(send_ancestor_guard_reject_not_a_class);
        return None;
    }

    // Finally, the part that makes the guard sufficient: nothing below the root may define the
    // method today. `Invariant::NoMethodOverride` keeps it that way.
    if !invariants::no_method_override_below(klass, mid, ANCESTOR_GUARD_MAX_SUBCLASSES) {
        incr_counter!(send_ancestor_guard_reject_overridden);
        return None;
    }

    incr_counter!(send_ancestor_guard_sites);
    Some(AncestorDispatch { klass })
}

/// Emit the single-arm guard chain for an [`AncestorDispatch`]: one `is_a?`-style check with a
/// [`Insn::Send`] that `type_specialize` will resolve statically, and the dynamic send as the
/// fallthrough for receivers that don't inherit the method.
///
/// Returns the join block, which the caller must continue emitting into, and the block parameter
/// holding the call result.
fn gen_send_ancestor_chain(
    fun: &mut Function,
    profiles: &mut ProfileOracle,
    dispatch: &AncestorDispatch,
    block: BlockId,
    insn_idx: u32,
    exit_state: &FrameState,
    exit_id: InsnId,
    recv: InsnId,
    cd: *const rb_call_data,
    block_handler: Option<BlockHandler>,
    args: Vec<InsnId>,
    opcode: u32,
) -> (BlockId, InsnId) {
    let join_block = fun.new_block(insn_idx);
    let join_param = fun.push_insn(join_block, Insn::Param);

    // gen_has_ancestor reads the prime classext of the receiver's class.
    fun.assume_root_box(block, exit_id);
    let has_ancestor = fun.push_insn(block, Insn::HasAncestor { val: recv, class: dispatch.klass });
    let iftrue_block = fun.new_block(insn_idx);
    let fall_through = fun.new_block(insn_idx);
    fun.push_insn(block, Insn::CondBranch {
        val: has_ancestor,
        if_true: BranchEdge { target: iftrue_block, args: vec![] },
        if_false: BranchEdge { target: fall_through, args: vec![] },
    });

    // Take a fresh Snapshot so the specialized send doesn't resolve the receiver from the
    // megamorphic profile that is keyed at exit_id, and copy the other operands' profile
    // entries over so argument-driven specializations still see them.
    let snapshot = fun.push_insn(iftrue_block, Insn::Snapshot { state: Box::new(exit_state.clone()) });
    profiles.copy_entries_except(exit_id, snapshot, recv, fun, None);
    let send = fun.push_insn(iftrue_block, Insn::Send { recv, cd, block: block_handler, args: args.clone(), state: snapshot, reason: Uncategorized(opcode.into()) });
    fun.record_ancestor_dispatch(send, dispatch.klass);
    fun.count(iftrue_block, Counter::send_ancestor_guard_count);
    fun.push_insn(iftrue_block, Insn::Jump(BranchEdge { target: join_block, args: vec![send] }));

    fun.count(fall_through, Counter::send_ancestor_guard_fallback_count);
    let fallback = fun.push_insn(fall_through, Insn::Send { recv, cd, block: block_handler, args, state: exit_id, reason: SendAncestorGuardFallback });
    fun.push_insn(fall_through, Insn::Jump(BranchEdge { target: join_block, args: vec![fallback] }));

    (join_block, join_param)
}

/// True if the `invokeblock` call flags permit inlining the block dispatch.
fn can_direct_invoke_block(flags: u32) -> bool {
    (flags & (VM_CALL_ARGS_SPLAT | VM_CALL_KW_SPLAT | VM_CALL_KWARG | VM_CALL_ARGS_BLOCKARG)) == 0
}

/// How a direct block dispatch has to reshape the `yield`ed arguments before the JIT-to-JIT
/// call, mirroring the `arg_setup_block` tail of `vm_callee_setup_block_arg()`. A block is
/// never an arity error: extra arguments are dropped and missing ones are `nil`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BlockArgAdapt {
    /// The arity already matches; pass the arguments through.
    Exact,
    /// Keep only the first `n` arguments and drop the rest.
    Truncate(usize),
    /// Append `n` `nil`s.
    NilFill(usize),
}

/// How one arm of a polymorphic `invokeblock` chain prepares its arguments.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BlockDispatchArgs {
    /// Reshape the arguments statically.
    Adapt(BlockArgAdapt),
    /// Destructure the lone yielded argument into this many values, branching to the generic
    /// fallback when it is not an Array of exactly that length.
    AutoSplat(usize),
}

/// The reshape needed for `yield` with `argc` positional args to dispatch by inlining the
/// block ISEQ frame, or the reason it cannot. The block must take the simple callee-setup
/// path (`rb_simple_iseq_p`), avoid the run-time arg0 auto-splat, and contain no `throw`
/// (break / non-local return). Anything else falls back to the generic `invokeblock`
/// dispatch, with the returned reason attached to the fallback instruction.
///
/// A mismatched arity is *not* a reason to fall back. `vm_callee_setup_block_arg()` fills the
/// missing parameters with `nil` and drops the extra arguments, both of which are static once
/// the block ISEQ is known, so the dispatch can do the same to its register arguments. Only
/// the auto-splat is dynamic, and that one is modelled separately by
/// [`block_autosplat_arity`] because it needs a run-time Array check.
fn direct_invoke_block_adapt(iseq: IseqPtr, argc: usize) -> Result<BlockArgAdapt, SendFallbackReason> {
    if !unsafe { rb_simple_iseq_p(iseq) } {
        return Err(InvokeBlockNotSimpleIseq);
    }
    let lead_num = unsafe { rb_get_iseq_body_param_lead_num(iseq) } as usize;
    // The `arg_setup_block` auto-splat: a lone argument yielded to a block with lead
    // parameters that did not opt out with `|x|` is destructured with `rb_check_array_type()`.
    // Whether it splats at all depends on the run-time value, so this can't be a static
    // reshape; the callers that can afford the branch go through `block_autosplat_arity`.
    if argc == 1 && lead_num >= 1 && !unsafe { rb_get_iseq_flags_ambiguous_param0(iseq) } {
        return Err(InvokeBlockAmbiguousParam0);
    }
    let adapt = match argc.cmp(&lead_num) {
        std::cmp::Ordering::Equal => BlockArgAdapt::Exact,
        std::cmp::Ordering::Less => BlockArgAdapt::NilFill(lead_num - argc),
        std::cmp::Ordering::Greater => BlockArgAdapt::Truncate(lead_num),
    };
    // `break` out of a directly-invoked block frame does not unwind correctly:
    // vm_throw_start() matches the block owner's `cfp->pc` against the CATCH_TYPE_BREAK
    // entry's `cont`, and the PC the owner's frame reports after this dispatch does not
    // match, so the break is reported as an orphan ("break from proc-closure").
    // A plain non-local `return` is looked up by frame type and EP instead of by PC, so
    // blocks that only throw TAG_RETURN -- what this dispatch is for -- are fine.
    if crate::codegen::block_iseq_may_throw(iseq) && !block_iseq_throws_only_return(iseq) {
        return Err(InvokeBlockMayThrow);
    }
    Ok(adapt)
}

/// The number of values a lone `yield`ed argument is auto-splatted into when `iseq` is the
/// block, or `None` when a one-argument `yield` reaches the block's parameters unchanged.
///
/// This mirrors the `arg_setup_block` case of `setup_parameters_complex()`: a block given a
/// single argument destructures it with `rb_check_array_type` when it takes more than one
/// positional parameter. `ambiguous_param0` is how `|x|` opts *out* of that and receives the
/// array whole; `|x,|` clears it to opt back in.
///
/// Restricted to `rb_simple_iseq_p` ISEQs -- no optional, rest, post, keyword, kwrest, or
/// block parameter -- so `lead_num` is both the minimum and the maximum arity and the
/// expansion is exactly `lead_num` elements with no nil-filling or truncation to model.
fn block_autosplat_arity(iseq: IseqPtr) -> Option<usize> {
    if !unsafe { rb_simple_iseq_p(iseq) } {
        return None;
    }
    if unsafe { rb_get_iseq_flags_ambiguous_param0(iseq) } {
        return None;
    }
    let lead_num = unsafe { rb_get_iseq_body_param_lead_num(iseq) } as usize;
    // `lead_num == 1` is the `|x,|` shape, which auto-splats and then truncates to one
    // element. Expanding it would have to model the truncation, and it is rare; skip it.
    if lead_num < 2 {
        return None;
    }
    Some(lead_num)
}

/// How many values to auto-splat a lone `yield`ed argument into at this site, or `None` to
/// leave the argument alone.
///
/// Only reports an arity when the expansion is what lets the site take one of the two
/// single-ISEQ direct dispatches. The polymorphic, IFUNC, and generic paths hand `args`
/// straight to `rb_vm_invokeblock()` along with this site's call data, which still says one
/// argument, so those must keep seeing the unexpanded argument.
fn autosplat_direct_dispatch_arity(
    mode: AddIseqMode,
    flags: u32,
    yield_iseq: IseqPtr,
    block_handler_class: Option<VALUE>,
    argc: usize,
) -> Option<usize> {
    if argc != 1 || !can_direct_invoke_block(flags) {
        return None;
    }
    // Mirror the priority of the dispatch selection: a literal block of the frame we are
    // inlined into wins over the profiled one, because dispatching to it needs no guard.
    let inlined_literal = match mode {
        AddIseqMode::Inlined { blockiseq: Some(bi), .. } if get_lvar_level(yield_iseq) == 0 => Some(bi),
        _ => None,
    };
    let profiled = block_handler_class.and_then(|obj| {
        if unsafe { rb_IMEMO_TYPE_P(obj, imemo_iseq) == 1 } { Some(obj.as_iseq()) } else { None }
    });
    for candidate in [inlined_literal, profiled].into_iter().flatten() {
        let Some(splat_num) = block_autosplat_arity(candidate) else { continue };
        if direct_invoke_block_adapt(candidate, splat_num).is_ok() {
            return Some(splat_num);
        }
    }
    None
}

unsafe extern "C" {
    fn rb_jit_iseq_has_ensure_catch_entry(iseq: IseqPtr) -> bool;
}

/// How much bigger a callee may be when inlining it is what lets the `yield` inside it
/// dispatch to the caller's literal block directly instead of through
/// `rb_vm_invokeblock()`. Multiplies `--zjit-inline-threshold`, so tuning that option down
/// (or to 0, which disables inlining) still applies here. The default 30 * 3 = 90 clears the
/// iterators this shape shows up with: the Ruby-level `Array#each` needs 45.
const YIELD_INLINE_THRESHOLD_FACTOR: usize = 3;

/// How many callees a compiled function may inline past its cumulative budget because doing so
/// is what lets a `yield` inside them dispatch directly.
///
/// The size threshold above is not the binding constraint for these: `Array#each` and
/// `Array#map` are called from all over a Rails request, and the methods that call them are
/// big -- big enough that the caller's budget is spent before the iterator is reached. Every
/// such site leaves its `yield` on `rb_vm_invokeblock()` for the life of the process, because
/// the standalone `Array#each` sees hundreds of different blocks and no chain can cover them.
///
/// A flat multiple of the budget would be the obvious relaxation, but it scales with the
/// caller's size, which is backwards: the callers that need this most are the ones already
/// over budget, and the ones that would abuse it most are the ones with dozens of `.each`
/// calls. A small fixed allowance instead bounds the extra growth to a few iterator bodies per
/// function, no matter how big the function is.
const MAX_YIELD_INLINE_BONUSES: usize = 3;

/// True if the ISEQ contains an `invokeblock`, i.e. a `yield`.
fn iseq_contains_invokeblock(iseq: IseqPtr) -> bool {
    let encoded_size = unsafe { rb_iseq_encoded_size(iseq) };
    let mut insn_idx: u32 = 0;
    while insn_idx < encoded_size {
        let pc = unsafe { rb_iseq_pc_at_idx(iseq, insn_idx) };
        let opcode = unsafe { rb_iseq_bare_opcode_at_pc(iseq, pc) } as u32;
        if opcode == YARVINSN_invokeblock {
            return true;
        }
        insn_idx = insn_idx.saturating_add(unsafe { rb_insn_len(VALUE(opcode as usize)) }.try_into().unwrap());
    }
    false
}

/// True if every `throw` in the ISEQ is a plain non-local `return` (`TAG_RETURN` with no
/// extra flags), and there is at least one of them.
fn block_iseq_throws_only_return(iseq: IseqPtr) -> bool {
    let encoded_size = unsafe { rb_iseq_encoded_size(iseq) };
    let mut insn_idx: u32 = 0;
    let mut saw_throw = false;

    while insn_idx < encoded_size {
        let pc = unsafe { rb_iseq_pc_at_idx(iseq, insn_idx) };
        let opcode = unsafe { rb_iseq_bare_opcode_at_pc(iseq, pc) } as u32;

        if opcode == YARVINSN_throw {
            if unsafe { *rb_iseq_pc_at_idx(iseq, insn_idx + 1) }.as_u32() != RUBY_TAG_RETURN as u32 {
                return false;
            }
            saw_throw = true;
        }

        insn_idx = insn_idx.saturating_add(unsafe { rb_insn_len(VALUE(opcode as usize)) }.try_into().unwrap());
    }

    saw_throw
}

/// True if a `yield` to `block_iseq` can be compiled by inlining the block's body into the
/// yielding frame *and* rewriting the block's non-local `return` into a plain return of the
/// compiled function.
///
/// This only pays off, and is only sound, in a narrow shape:
///
/// * The block must be a literal block of `owner_iseq`, the frame ZJIT is compiling, so that
///   `return` inside it unwinds to exactly the frame the compiled function returns from. The
///   caller establishes that; here we only check `owner_iseq` is a method, because a `return`
///   inside a block nested in another block escapes to the enclosing *method*, not to the
///   block frame we would be returning from.
/// * Every `throw` must be `TAG_RETURN`. `break`, `retry`, and `redo` unwind to frames we
///   would still have to find at runtime.
/// * None of the frames the `return` unwinds through may have an `ensure`.
///   `vm_exec_handle_exception` runs their CATCH_TYPE_ENSURE entries while unwinding; a
///   plain return cannot. `yield_iseq` is the frame containing the `yield` (an inlined
///   callee) and `owner_iseq` is where the unwinding stops. Only `ensure` matters: a
///   `TAG_RETURN` unwind consults no other catch type.
fn block_return_inlinable(block_iseq: IseqPtr, yield_iseq: IseqPtr, owner_iseq: IseqPtr) -> bool {
    if unsafe { rb_get_iseq_body_type(owner_iseq) } != ISEQ_TYPE_METHOD {
        return false;
    }
    if !block_iseq_throws_only_return(block_iseq) {
        return false;
    }
    for iseq in [block_iseq, yield_iseq, owner_iseq] {
        if unsafe { rb_jit_iseq_has_ensure_catch_entry(iseq) } {
            return false;
        }
    }
    // The inlined body is emitted into the caller, so apply the same eligibility and size
    // limits the method inliner uses.
    if !Function::can_inline(block_iseq) {
        return false;
    }
    let threshold = get_option!(inline_threshold);
    if threshold == 0 || unsafe { get_iseq_encoded_size(block_iseq) } as usize > threshold {
        return false;
    }
    true
}

/// Emit an inlined copy of `block_iseq`'s body in place of a `yield` to it, and return the
/// SSA value holding the block's result. `block` is advanced to the continuation the inlined
/// body returns to, the way the polymorphic dispatch below advances it to a join block.
///
/// This mirrors [`Function::inline_methods`] for a block frame instead of a method frame:
/// the callee body's entry params are aliased to this frame's values, a real block frame is
/// pushed so side exits and frame walks see a well-formed CFP chain, and the body's `leave`
/// paths jump to the continuation, which pops the frame back off.
///
/// Returns `None` without touching `fun` if the block body fails to translate.
fn inline_block_at_yield(
    fun: &mut Function,
    profiles: &mut ProfileOracle,
    block: &mut BlockId,
    block_iseq: IseqPtr,
    args: &[InsnId],
    caller_argc: usize,
    call_state: InsnId,
    exit_id: InsnId,
    exit_state: &FrameState,
    insn_idx: u32,
) -> Option<InsnId> {
    // The block frame sits one level below the frame that yields to it.
    let block_depth = exit_state.depth + 1;

    // Snapshot the HIR length so a failed translation can be rolled back. Nothing is added
    // to `block` until the translation succeeds, so only the append-only tables need it.
    let pre_insns_len = fun.insns.len();
    let pre_insn_types_len = fun.insn_types.len();
    let pre_blocks_len = fun.blocks.len();

    let continuation = fun.new_block(insn_idx);
    // The caller state the inlined body's side exits restore: this frame stopped at the
    // `invokeblock`, with the arguments popped off its stack. That is `caller_argc` values,
    // which is not `args.len()` when a lone yielded Array was auto-splatted into `args`.
    let caller_stack_size = exit_state.stack().len() - caller_argc;
    let post_yield_caller = fun.new_insn(Insn::Snapshot {
        state: Box::new(exit_state.with_stack_size(caller_stack_size)),
    });

    let mode = AddIseqMode::Inlined {
        return_block: continuation,
        caller: post_yield_caller,
        depth: block_depth,
        // direct_invoke_block_adapt() checked the block takes the simple callee-setup path,
        // so it has no optionals and only one opt-table entry.
        jit_entry_idx: 0,
        blockiseq: None,
        block_return_pops: Some(block_depth as u32),
    };
    let add_result = match add_iseq_to_hir(fun, block_iseq, mode) {
        Ok(result) => result,
        Err(_) => {
            fun.insns.truncate(pre_insns_len);
            fun.insn_types.truncate(pre_insn_types_len);
            fun.blocks.truncate(pre_blocks_len);
            incr_counter!(inline_reject_compile_failure);
            return None;
        }
    };
    fun.num_instructions += fun.insns.len() - pre_insns_len;
    incr_counter!(inline_block_count);

    // Read the block handler out of this frame's EP, as the guard-free dispatch in
    // push_invoke_block_iseq_direct() does: PushInlineFrame wrote this exact block into the
    // frame's EP from a compile-time constant, so neither the tag nor the ISEQ identity
    // needs a guard. captured->self is the block's self, and codegen reads captured->ep for
    // the new frame's specval.
    let ep = fun.get_ep(*block, 0);
    let block_handler = fun.load_ep_env_field(*block, ep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, VM_ENV_DATA_INDEX_SPECVAL, types::CInt64);
    let untag_mask = fun.push_insn(*block, Insn::Const { val: Const::CInt64(!0x3) });
    let captured = fun.push_insn(*block, Insn::IntAnd { left: block_handler, right: untag_mask });
    let block_self = fun.load_field(*block, captured, FieldName::SelfParam, 0, types::BasicObject);

    // Map the body entry's params ([self, local0, local1, ...]) onto our values. The block
    // is gated to `rb_simple_iseq_p` with exact arity, so the leading locals are exactly the
    // arguments and the rest are non-parameter locals that start out nil.
    let body_entry = add_result.body_entry_block.expect("inlined compilation always produces a body entry block");
    let body_params: Vec<InsnId> = fun.blocks[body_entry.to_usize()].params.clone();
    if let Some(&self_param) = body_params.first() {
        fun.make_equal_to(self_param, block_self);
    }
    for (idx, &param_id) in body_params.iter().skip(1).enumerate() {
        if let Some(&arg) = args.get(idx) {
            fun.make_equal_to(param_id, arg);
        } else {
            let nil = fun.push_insn(*block, Insn::Const { val: Const::Value(Qnil) });
            fun.make_equal_to(param_id, nil);
        }
    }
    // The params are aliased rather than passed as branch arguments, so the Jump below
    // passes none. Clear them to keep validation happy.
    fun.blocks[body_entry.to_usize()].params.clear();

    fun.push_insn_id(*block, post_yield_caller);
    fun.push_insn(*block, Insn::PushInlineFrame {
        iseq: block_iseq,
        cme: std::ptr::null(),
        recv: block_self,
        num_args: args.len().try_into().expect("checked in HIR"),
        blockiseq: None,
        captured: Some(captured),
        // The frame is laid out on top of `call_state`'s stack, which ends in exactly
        // `args`. That is not the interpreter's own stack whenever the yielded arguments
        // were reshaped for the block's parameters -- auto-splatted, truncated, or
        // nil-filled -- which is why the two snapshots are separate.
        state: call_state,
        guard_state: exit_id,
    });
    fun.push_insn(*block, Insn::Jump(BranchEdge { target: body_entry, args: vec![] }));

    // Every `leave` in the block body jumps here with its value; a `return` skips this and
    // returns from the compiled function instead.
    let return_val = fun.push_insn(continuation, Insn::Param);
    fun.push_insn(continuation, Insn::PopInlineFrame { iseq: block_iseq, argc: args.len(), state: call_state });

    profiles.append(&add_result.profiles);
    *block = continuation;
    Some(return_val)
}

impl Function {
    fn new(iseq: *const rb_iseq_t) -> Function {
        Function {
            iseq,
            was_invalidated_for_singleton_class_creation: false,
            self_is_heap_object: false,
            policy: CompilePolicy::new(iseq),
            insns: vec![],
            insn_types: vec![],
            union_find: UnionFind::<InsnId>::new(),
            blocks: vec![Block::default(), Block::default()],
            entries_block: BlockId(0),
            entry_block: BlockId(1),
            jit_entry_blocks: vec![],
            param_types: vec![],
            profiles: None,
            guarded_profiled_types: HashMap::default(),
            ancestor_dispatch: HashMap::default(),
            num_instructions: 0,
            yield_inline_bonuses: 0,
            inline_budget_exhausted: false,
            exception_entry: None,
            send_mid_overrides: HashMap::default(),
        }
    }

    pub fn iseq(&self) -> *const rb_iseq_t {
        self.iseq
    }

    /// If this function was compiled as an exception handler entry, where it resumes
    pub fn exception_entry(&self) -> Option<ExceptionEntry> {
        self.exception_entry
    }

    // Add an instruction to the function without adding it to any block
    fn new_insn(&mut self, insn: Insn) -> InsnId {
        let id = InsnId::from(self.insns.len());
        if insn.has_output() {
            self.insn_types.push(types::Any);
        } else {
            self.insn_types.push(types::Empty);
        }
        self.insns.push(insn);
        id
    }

    // Add an instruction to an SSA block
    pub fn push_insn(&mut self, block: BlockId, insn: Insn) -> InsnId {
        let is_param = matches!(insn, Insn::Param);
        let id = self.new_insn(insn);
        if is_param {
            self.blocks[block.to_usize()].params.push(id);
        } else {
            self.blocks[block.to_usize()].insns.push(id);
        }
        id
    }

    pub fn push_comment(&mut self, block: BlockId, message: String) -> InsnId {
        self.push_insn(block, Insn::Comment { message })
    }

    pub fn load_pc(&mut self, block: BlockId) -> InsnId {
        self.push_insn(block, Insn::LoadPC)
    }

    pub fn load_ec(&mut self, block: BlockId) -> InsnId {
        self.push_insn(block, Insn::LoadEC)
    }

    pub fn load_sp(&mut self, block: BlockId) -> InsnId {
        self.push_insn(block, Insn::LoadSP)
    }

    pub fn load_self(&mut self, block: BlockId) -> InsnId {
        self.push_insn(block, Insn::LoadSelf)
    }

    pub fn get_ep(&mut self, block: BlockId, level: u32) -> InsnId {
        self.push_insn(block, Insn::GetEP { level })
    }

    pub fn load_field(&mut self, block: BlockId, recv: InsnId, id: FieldName, offset: i32, return_type: Type) -> InsnId {
        let num_bits = return_type.num_bits();
        self.push_insn(block, Insn::LoadField { recv, id, offset, return_type, num_bits })
    }

    pub fn load_string_length(&mut self, block: BlockId, str: InsnId) -> InsnId {
        self.load_field(block, str, FieldName::len, RUBY_OFFSET_RSTRING_LEN, types::CInt64)
    }

    /// Load `captured->code.iseq` from a `struct rb_captured_block *`.
    fn load_captured_code_iseq(&mut self, block: BlockId, captured: InsnId) -> InsnId {
        let offset: i32 = std::mem::offset_of!(rb_captured_block, code).try_into().unwrap();
        self.load_field(block, captured, FieldName::code_iseq, offset, types::CPtr)
    }

    /// Reshape `args` the way `vm_callee_setup_block_arg()` would for a block whose parameters
    /// need `adapt`, and rebuild the frame state the callee frame is pushed on top of.
    ///
    /// The direct dispatch passes arguments in registers and derives the caller's saved SP from
    /// `state.stack().len() - args.len()`, so the state's stack has to end in the reshaped
    /// arguments even though the interpreter only ever had the original ones there. This is the
    /// same split the auto-splat expansion makes.
    fn adapt_block_args(&mut self, block: BlockId, adapt: BlockArgAdapt, args: Vec<InsnId>, state: InsnId) -> (Vec<InsnId>, InsnId) {
        let original_argc = args.len();
        let args = match adapt {
            BlockArgAdapt::Exact => return (args, state),
            BlockArgAdapt::Truncate(lead_num) => args[..lead_num].to_vec(),
            BlockArgAdapt::NilFill(num_nils) => {
                let nil = self.push_insn(block, Insn::Const { val: Const::Value(Qnil) });
                let mut args = args;
                args.resize(original_argc + num_nils, nil);
                args
            }
        };
        let adapted = self.frame_state(state).with_replaced_args(&args, original_argc);
        let state = self.push_insn(block, Insn::Snapshot { state: Box::new(adapted) });
        (args, state)
    }

    /// Untag an ISEQ block handler into its `struct rb_captured_block *`:
    /// captured = block_handler & ~0x3
    fn untag_block_handler(&mut self, block: BlockId, block_handler: InsnId) -> InsnId {
        let untag_mask = self.push_insn(block, Insn::Const { val: Const::CInt64(!0x3) });
        self.push_insn(block, Insn::IntAnd { left: block_handler, right: untag_mask })
    }

    /// Dispatch `yield` to a known ISEQ block without guarding tag or ISEQ. When the enclosing method is
    /// inlined and the caller passed a literal block, [`Insn::PushInlineFrame`] wrote that exact
    /// block into this frame's EP from a compile-time constant, so both guards are unnecessary.
    ///
    /// `state` describes the caller frame the callee is pushed on top of, so its stack must
    /// end in `args`. `guard_state` is what the guards below side-exit to, which is the
    /// interpreter's own state at this `invokeblock`; the two differ when a lone yielded Array
    /// was auto-splatted into `args`, because the interpreter still has just the Array.
    #[allow(clippy::too_many_arguments)]
    fn push_invoke_block_iseq_direct(&mut self, block: BlockId, block_iseq: IseqPtr, level: u32, adapt: BlockArgAdapt, args: Vec<InsnId>, state: InsnId, guard_state: InsnId, guarded: bool) -> InsnId {
        let ep = self.get_ep(block, level);
        let block_handler = self.load_ep_env_field(block, ep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, VM_ENV_DATA_INDEX_SPECVAL, types::CInt64);

        if guarded {
            // Guard the handler is an ISEQ block: VM_BH_ISEQ_BLOCK_P is `& 0x3 == 0x1`.
            let tag_mask = self.push_insn(block, Insn::Const { val: Const::CInt64(0x3) });
            let tag = self.push_insn(block, Insn::IntAnd { left: block_handler, right: tag_mask });
            self.push_insn(block, Insn::GuardBitEquals { val: tag, expected: Const::CInt64(0x1), reason: Box::new(SideExitReason::InvokeBlockHandlerNotIseq), state: guard_state, recompile: Some(Recompile) });
        }

        let captured = self.untag_block_handler(block, block_handler);

        if guarded {
            // Guard captured->code.iseq is the profiled block iseq. Compare the raw imemo pointer:
            // type inference (from_value) can't type an iseq imemo, so guard it as a CPtr identity.
            let captured_iseq = self.load_captured_code_iseq(block, captured);
            self.push_insn(block, Insn::GuardBitEquals { val: captured_iseq, expected: Const::CPtr(block_iseq as *const u8), reason: Box::new(SideExitReason::InvokeBlockIseqChanged), state: guard_state, recompile: Some(Recompile) });
        }

        let (args, state) = self.adapt_block_args(block, adapt, args, state);
        self.push_insn(block, Insn::InvokeBlockIseqDirect { iseq: block_iseq, captured, args, state })
    }


    /// Dispatch `yield` without entering the interpreter's send path, joining on the generic
    /// `rb_vm_invokeblock` fallback for anything this can't handle. Each candidate is a branch,
    /// not a guard: the site can legitimately see more than one handler, so a side exit would
    /// keep failing and recompiling.
    ///
    /// Two kinds are handled. An IFUNC handler goes straight to `rb_vm_yield_with_cfunc`; it
    /// needs nothing but the handler's tag, so this test is always emitted. `iseqs` are the
    /// ISEQ blocks the profile saw that can be invoked JIT-to-JIT, in frequency order.
    /// `ifunc_first` puts the IFUNC test ahead of the ISEQ chain because the profile's most
    /// frequent handler was an IFUNC.
    ///
    /// Returns the block compilation should continue from and the result instruction.
    fn dispatch_invoke_block(
        &mut self,
        block: BlockId,
        insn_idx: u32,
        level: u32,
        cd: *const rb_call_data,
        iseqs: &[(IseqPtr, BlockArgAdapt)],
        ifunc_first: bool,
        args: Vec<InsnId>,
        state: InsnId,
        fallback_reason: SendFallbackReason,
    ) -> (BlockId, InsnId) {
        let ep = self.get_ep(block, level);
        let block_handler = self.load_ep_env_field(block, ep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, VM_ENV_DATA_INDEX_SPECVAL, types::CInt64);
        // The handler kind is in the low two bits: VM_BH_ISEQ_BLOCK_P is `& 0x3 == 0x1` and
        // VM_BH_IFUNC_P is `& 0x3 == 0x3`.
        let tag_mask = self.push_insn(block, Insn::Const { val: Const::CInt64(0x3) });
        let tag = self.push_insn(block, Insn::IntAnd { left: block_handler, right: tag_mask });

        let join_block = self.new_block(insn_idx);
        let join_param = self.push_insn(join_block, Insn::Param);
        let fallback_block = self.new_block(insn_idx);

        // Test the handler kinds in profiled-frequency order, so the hottest one is first, and
        // let the last test fall through straight to the generic fallback.
        let iseqs_first = !ifunc_first && !iseqs.is_empty();
        #[allow(unused_assignments)]
        let mut cur_block = block;
        let next_miss_block = |fun: &mut Self, last: bool| {
            if last { fallback_block } else { fun.new_block(insn_idx) }
        };

        if iseqs_first {
            let miss_block = next_miss_block(self, false);
            self.push_iseq_block_dispatch(cur_block, insn_idx, iseqs, tag, block_handler, &args, state, join_block, miss_block);
            cur_block = miss_block;
        }

        // The IFUNC test: VM_BH_IFUNC_P, then hand the captured block to rb_vm_yield_with_cfunc.
        let ifunc_tag = self.push_insn(cur_block, Insn::Const { val: Const::CInt64(0x3) });
        let is_ifunc = self.push_insn(cur_block, Insn::IsBitEqual { left: tag, right: ifunc_tag });
        let ifunc_block = self.new_block(insn_idx);
        let miss_block = next_miss_block(self, iseqs_first || iseqs.is_empty());
        self.push_insn(cur_block, Insn::CondBranch {
            val: is_ifunc,
            if_true: BranchEdge { target: ifunc_block, args: vec![] },
            if_false: BranchEdge { target: miss_block, args: vec![] },
        });
        let ifunc_result = self.push_insn(ifunc_block, Insn::InvokeBlockIfunc { cd, block_handler, args: args.clone(), state });
        self.push_insn(ifunc_block, Insn::Jump(BranchEdge { target: join_block, args: vec![ifunc_result] }));
        cur_block = miss_block;

        if !iseqs_first && !iseqs.is_empty() {
            self.push_iseq_block_dispatch(cur_block, insn_idx, iseqs, tag, block_handler, &args, state, join_block, fallback_block);
        }

        let fallback_result = self.push_insn(fallback_block, Insn::InvokeBlock {
            cd, args, state, reason: fallback_reason,
        });
        self.push_insn(fallback_block, Insn::Jump(BranchEdge { target: join_block, args: vec![fallback_result] }));

        (join_block, join_param)
    }

    /// Emit the ISEQ half of [`Function::dispatch_invoke_block`] into `block`: check the handler
    /// is an ISEQ block, then compare `captured->code.iseq` against each candidate in turn and
    /// invoke the matching one JIT-to-JIT. Every path that doesn't match branches to
    /// `miss_block`; every path that does jumps to `join_block` with its result.
    #[allow(clippy::too_many_arguments)]
    fn push_iseq_block_dispatch(
        &mut self,
        block: BlockId,
        insn_idx: u32,
        iseqs: &[(IseqPtr, BlockArgAdapt)],
        tag: InsnId,
        block_handler: InsnId,
        args: &[InsnId],
        state: InsnId,
        join_block: BlockId,
        miss_block: BlockId,
    ) {
        let iseq_tag = self.push_insn(block, Insn::Const { val: Const::CInt64(0x1) });
        let tag_matches = self.push_insn(block, Insn::IsBitEqual { left: tag, right: iseq_tag });
        let dispatch_block = self.new_block(insn_idx);
        self.push_insn(block, Insn::CondBranch {
            val: tag_matches,
            if_true: BranchEdge { target: dispatch_block, args: vec![] },
            if_false: BranchEdge { target: miss_block, args: vec![] },
        });

        // captured = block_handler & ~0x3 (struct rb_captured_block *)
        let untag_mask = self.push_insn(dispatch_block, Insn::Const { val: Const::CInt64(!0x3) });
        let captured = self.push_insn(dispatch_block, Insn::IntAnd { left: block_handler, right: untag_mask });
        let captured_iseq = self.load_captured_code_iseq(dispatch_block, captured);

        let mut compare_block = dispatch_block;
        for (idx, &(block_iseq, adapt)) in iseqs.iter().enumerate() {
            let expected = self.push_insn(compare_block, Insn::Const { val: Const::CPtr(block_iseq as *const u8) });
            let iseq_matches = self.push_insn(compare_block, Insn::IsBitEqual { left: captured_iseq, right: expected });
            let direct_block = self.new_block(insn_idx);
            let iseq_miss_block = if idx + 1 == iseqs.len() { miss_block } else { self.new_block(insn_idx) };
            self.push_insn(compare_block, Insn::CondBranch {
                val: iseq_matches,
                if_true: BranchEdge { target: direct_block, args: vec![] },
                if_false: BranchEdge { target: iseq_miss_block, args: vec![] },
            });
            let (call_args, call_state) = self.adapt_block_args(direct_block, adapt, args.to_vec(), state);
            let direct_result = self.push_insn(direct_block, Insn::InvokeBlockIseqDirect { iseq: block_iseq, captured, args: call_args, state: call_state });
            self.push_insn(direct_block, Insn::Jump(BranchEdge { target: join_block, args: vec![direct_result] }));
            compare_block = iseq_miss_block;
        }
    }

    // Add an instruction to an SSA block
    fn push_insn_id(&mut self, block: BlockId, insn_id: InsnId) -> InsnId {
        self.blocks[block.to_usize()].insns.push(insn_id);
        insn_id
    }

    /// Return the number of instructions
    pub fn num_insns(&self) -> usize {
        self.insns.len()
    }

    /// Return the deepest inlining nesting present in this function's frames.
    /// The top-level frame is depth 0, so a function that inlines nothing
    /// returns 0. Codegen uses this to size the per-function JITFrame slot
    /// region, which needs one slot per simultaneously live frame, i.e.
    /// `inlining_depth() + 1` slots. This is a measurement of what the function
    /// actually contains, not a configured limit.
    pub fn inlining_depth(&self) -> InlineDepth {
        self.insns.iter().filter_map(|insn| match insn {
            Insn::Snapshot { state } => Some(state.depth),
            _ => None,
        }).max().unwrap_or(0)
    }

    /// Return a resolved, freshly allocated FrameState at the given instruction index.
    pub fn frame_state(&self, insn_id: InsnId) -> FrameState {
        match self.find(insn_id) {
            Insn::Snapshot { state } => *state,
            insn => panic!("Unexpected non-Snapshot {insn} when looking up FrameState"),
        }
    }

    /// Return an unresolved FrameState reference at the given instruction index.
    pub fn frame_state_ref(&self, insn_id: InsnId) -> &FrameState {
        match self.find_ref(insn_id) {
            Insn::Snapshot { state } => state,
            insn => panic!("Unexpected non-Snapshot {insn} when looking up FrameState"),
        }
    }

    /// Return a FrameState's iseq at the given instruction index.
    pub fn frame_state_iseq(&self, insn_id: InsnId) -> *const rb_iseq_t {
        self.frame_state_ref(insn_id).iseq
    }

    /// Return a FrameState's interpreter instruction index.
    pub fn frame_state_insn_idx(&self, insn_id: InsnId) -> YarvInsnIdx {
        self.frame_state_ref(insn_id).insn_idx
    }

    /// Return the inlining depth recorded on the `Snapshot` at the given
    /// instruction index. This peeks the field directly so callers that only
    /// need the depth avoid cloning the whole `FrameState`, including its stack
    /// and locals, the way [`Function::frame_state`] does.
    fn frame_depth(&self, insn_id: InsnId) -> InlineDepth {
        let insn_id = self.union_find.find_const(insn_id);
        match &self.insns[insn_id.to_usize()] {
            Insn::Snapshot { state } => state.depth,
            insn => panic!("Unexpected non-Snapshot {insn} when looking up frame depth"),
        }
    }

    /// Return whether the representative of `insn_id` is a `SendDirect` without
    /// cloning the instruction or resolving its operands.
    fn is_send_direct(&self, insn_id: InsnId) -> bool {
        matches!(self.find_ref(insn_id), Insn::SendDirect(..))
    }

    fn new_block(&mut self, insn_idx: u32) -> BlockId {
        let id = BlockId::from(self.blocks.len());
        let block = Block {
            insn_idx,
            .. Block::default()
        };
        self.blocks.push(block);
        id
    }

    fn remove_block(&mut self, block_id: BlockId) {
        if BlockId::from(self.blocks.len() - 1) != block_id {
            panic!("Can only remove the last block");
        }
        self.blocks.pop();
    }

    /// Return an iterator over the successor blocks of `block`. NB: the iteration order is
    /// intentionally undefined and the same BlockId may be yielded multiple times.
    fn successors(&self, block: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        // Read the terminator directly rather than through `find`, which clones the whole `Insn`.
        // `find` also resolves the id through union-find, but a terminator is never unioned: it
        // produces no value, and `make_equal_to` asserts `has_output()`. So the instruction stored
        // at this id is always the terminator itself, and reading it by reference matches what
        // `find` would return.
        let terminator = &self.insns[self.blocks[block.to_usize()].insns.last().unwrap().to_usize()];

        let (first, second, rest): (Option<BlockId>, Option<BlockId>, &[BlockId]) = match terminator {
            Insn::CondBranch { if_true, if_false, .. } => (Some(if_true.target), Some(if_false.target), &[]),
            Insn::Jump(edge) => (Some(edge.target), None, &[]),
            Insn::Entries { targets } => (None, None, targets.as_slice()),

            // Terminators such as `Return`, `SideExit`, `Throw`, and
            // `Unreachable` have no successors. A block that does not end in a
            // terminator is malformed, but we still want to traverse a poorly
            // constructed CFG when debugging, so we treat it as having no
            // successors; the validation routines report the missing
            // terminator separately.
            _ => (None, None, &[]),
        };

        first.into_iter().chain(second).chain(rest.iter().copied())
    }

    /// Return a reference to the Block at the given index.
    pub fn block(&self, block_id: BlockId) -> &Block {
        &self.blocks[block_id.to_usize()]
    }

    /// Return a reference to the entry block.
    pub fn entry_block(&self) -> &Block {
        &self.blocks[self.entry_block.to_usize()]
    }

    /// Return the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn assume_single_ractor_mode(&mut self, block: BlockId, state: InsnId) -> bool {
        if unsafe { rb_jit_multi_ractor_p() } {
            false
        } else {
            self.push_insn(block, Insn::PatchPoint { invariant: Invariant::SingleRactorMode, state });
            true
        }
    }

    /// Assume that only the root box is active, so we can safely read from the prime classext.
    /// Returns true if safe to assume so and emits a PatchPoint.
    pub fn assume_root_box(&mut self, block: BlockId, state: InsnId) -> bool {
        if invariants::non_root_box_created() {
            false
        } else {
            self.push_insn(block, Insn::PatchPoint { invariant: Invariant::RootBoxOnly, state });
            true
        }
    }

    /// Assume that objects of a given class will have no singleton class.
    /// Returns true if safe to assume so and emits a PatchPoint.
    /// Returns false if we've already seen a singleton class for this class,
    /// to avoid an invalidation loop.
    pub fn assume_no_singleton_classes(&mut self, block: BlockId, klass: VALUE, state: InsnId) -> bool {
        if !klass.instance_can_have_singleton_class() {
            // This class can never have a singleton class, so no patchpoint needed.
            return true;
        }
        if klass.is_singleton_class() {
            // When a value has a singleton class, its effective class can't change anymore.
            // No patchpoint needed.
            return true;
        }
        if self.was_invalidated_for_singleton_class_creation && invariants::has_singleton_class_of(klass) {
            // A previous compilation of this ISEQ was invalidated for singleton class
            // creation. Avoid repeating the invalidation.
            return false;
        }
        self.push_insn(block, Insn::PatchPoint { invariant: Invariant::NoSingletonClass { klass }, state });
        true
    }

    pub fn assume_bop_not_redefined(&mut self, block: BlockId, klass: RedefinitionFlag, bop: ruby_basic_operators, state: InsnId) -> bool {
        if !unsafe { rb_BASIC_OP_UNREDEFINED_P(bop, klass) } {
            return false;
        }
        self.push_insn(block, Insn::PatchPoint { invariant: Invariant::BOPRedefined { klass, bop }, state });
        true
    }

    pub fn guard_bop_not_redefined(&mut self, block: BlockId, klass: RedefinitionFlag, bop: ruby_basic_operators, state: InsnId) -> bool {
        if self.assume_bop_not_redefined(block, klass, bop, state) {
            return true;
        }
        self.push_insn(block, Insn::SideExit { state, reason: Box::new(SideExitReason::PatchPoint(Invariant::BOPRedefined { klass, bop })), recompile: None });
        false
    }

    /// Emit the patch points that keep `cme` the right target for a specialized send.
    ///
    /// `ancestor_class` is set when the receiver was guarded to inherit from a class rather
    /// than to be an exact class. Then the target only stays right as long as nothing below
    /// that class defines the method, which is a separate assumption from the method not being
    /// redefined where it is defined.
    fn assume_cme_for_send(
        &mut self,
        block: BlockId,
        klass: VALUE,
        mid: ID,
        cme: *const rb_callable_method_entry_t,
        state: InsnId,
        ancestor_class: Option<VALUE>,
    ) {
        if let Some(ancestor_class) = ancestor_class {
            self.push_insn(block, Insn::PatchPoint {
                invariant: Invariant::NoMethodOverride { klass: ancestor_class, method: mid, cme },
                state,
            });
        }
        self.push_insn(block, Insn::PatchPoint { invariant: Invariant::MethodRedefined { klass, method: mid, cme }, state });
    }

    /// [`Function::assume_no_singleton_classes`] for a specialized send.
    ///
    /// A send behind an ancestor guard needs no such assumption: the guard rejects any receiver
    /// whose class is a singleton class, so the receivers that reach the specialized call are
    /// exactly the ones whose lookup the [`Invariant::NoMethodOverride`] assumption covers. The
    /// assumption would not even be the right one, since it is about instances of `klass` while
    /// the receivers here are instances of its subclasses.
    fn assume_no_singleton_classes_for_send(&mut self, block: BlockId, klass: VALUE, state: InsnId, ancestor_class: Option<VALUE>) -> bool {
        if ancestor_class.is_some() {
            return true;
        }
        self.assume_no_singleton_classes(block, klass, state)
    }

    /// Remember that `send` sits behind a [`Insn::HasAncestor`] guard for `klass`, so
    /// `type_specialize` can dispatch it on the method `klass` resolves rather than on the
    /// receiver's exact class.
    fn record_ancestor_dispatch(&mut self, send: InsnId, klass: VALUE) {
        self.ancestor_dispatch.insert(send, klass);
    }

    /// The class a send's receiver was guarded to inherit from, if any. See
    /// [`Function::record_ancestor_dispatch`].
    fn ancestor_dispatch_class(&self, send: InsnId) -> Option<VALUE> {
        self.ancestor_dispatch.get(&send).copied()
    }

    pub fn count(&mut self, block: BlockId, counter: Counter) {
        if get_option!(stats) {
            self.push_insn(block, Insn::IncrCounter(counter));
        }
    }

    /// Do a shallow look up of the instruction ID in the union-find. Does inspect or rewrite any
    /// operands.
    pub fn find_id(&self, insn_id: InsnId) -> InsnId {
        self.union_find.find_const(insn_id)
    }

    /// Return a copy of the instruction where the instruction and its operands have been read from
    /// the union-find table (to find the current most-optimized version of this instruction). See
    /// [`UnionFind`] for more.
    ///
    /// This is _the_ function for reading [`Insn`]. Use frequently. Example:
    ///
    /// ```ignore
    /// match func.find(insn_id) {
    ///   IfTrue { val, target } if func.is_truthy(val) => {
    ///     let jump = self.new_insn(Insn::Jump(target));
    ///     func.make_equal_to(insn_id, jump);
    ///   }
    ///   _ => {}
    /// }
    /// ```
    ///
    /// You should prefer [`Function::resolve`] and [`ResolvedInsnId::insn`] when you want to read an
    /// instruction without cloning it.
    pub fn find(&self, insn_id: InsnId) -> Insn {
        macro_rules! find {
            ( $x:expr ) => {
                {
                    // TODO(max): Figure out why borrow_mut().find() causes `already borrowed:
                    // BorrowMutError`
                    self.union_find.find_const($x)
                }
            };
        }
        let insn_id = find!(insn_id);
        let mut result = self.insns[insn_id.to_usize()].clone();
        result.for_each_operand_mut(&mut |operand: &mut InsnId| {
            *operand = find!(*operand);
        });
        result
    }

    /// Return a reference to the instruction at `insn_id` (after resolving via union-find)
    /// without cloning it. Unlike [`ResolvedInsnId::insn`], this does not require the operands
    /// to have been resolved first, so the returned instruction's operands may be stale. Use it
    /// when the caller only inspects the opcode, or resolves the operands itself.
    pub fn find_ref(&self, insn_id: InsnId) -> &Insn {
        &self.insns[self.find_id(insn_id).to_usize()]
    }

    /// Make the operands of the instruction at `find(insn_id)` point to the current representative
    /// of each operand.
    ///
    /// Meant to be used in tandem with [`ResolvedInsnId::insn`]. Example:
    ///
    /// ```ignore
    /// match func.resolve(insn_id).insn(func) {
    ///   ...
    /// }
    /// ```
    pub fn resolve(&mut self, insn_id: InsnId) -> ResolvedInsnId {
        let Self { insns, union_find, .. } = self;
        let insn_id = union_find.find_const(insn_id);
        insns[insn_id.to_usize()].for_each_operand_mut(|operand: &mut InsnId| {
            *operand = union_find.find_const(*operand);
        });
        ResolvedInsnId(insn_id)
    }

    /// Point every instruction's operands at their union-find representatives, so that a
    /// consumer can read instructions by reference instead of cloning one to resolve it.
    ///
    /// Optimizer passes resolve as they go, but only the instructions they actually look
    /// at, so after optimization some operands still name an instruction that has since
    /// been forwarded. Codegen dealt with that by cloning: once for the instruction it is
    /// compiling, and again for the `Snapshot` it asks about for nearly every one of them,
    /// which means a `Box` plus copies of the frame's stack and locals vectors per
    /// instruction. Doing the resolution once, here, removes all of that.
    ///
    /// Walks the instruction arena rather than the blocks: a `Snapshot` that no block
    /// lists any more can still be reachable as another instruction's `state` operand.
    pub fn canonicalize_operands(&mut self) {
        let Self { insns, union_find, .. } = self;
        let union_find = union_find;
        for insn in insns.iter_mut() {
            insn.for_each_operand_mut(|operand: &mut InsnId| {
                *operand = union_find.find_const(*operand);
            });
        }
    }

    /// Check the postcondition of [`Self::canonicalize_operands`]. Codegen reads
    /// instructions and frame states by reference and has no way to resolve them itself,
    /// so a stale operand there would silently generate code for the wrong value.
    #[cfg(debug_assertions)]
    pub fn assert_operands_canonical(&self) {
        let union_find = &self.union_find;
        for (idx, insn) in self.insns.iter().enumerate() {
            insn.for_each_operand(|operand| {
                assert_eq!(operand, union_find.find_const(operand),
                    "operand {operand} of insn {idx} was not canonicalized");
            });
        }
    }

    /// Update DynamicSendReason for the instruction at insn_id
    fn set_dynamic_send_reason(&mut self, insn_id: InsnId, dynamic_send_reason: SendFallbackReason) {
        use Insn::*;
        // Always set the reason: convert_no_profile_sends depends on it to identify
        // sends that should be converted to side exits for exit-based recompilation.
        match self.insns.get_mut(insn_id.to_usize()).unwrap() {
            Send { reason, .. }
            | SendForward { reason, .. }
            | InvokeSuper { reason, .. }
            | InvokeSuperForward { reason, .. }
            | InvokeBlock { reason, .. }
            => *reason = dynamic_send_reason,
            _ => unreachable!("unexpected instruction {} at {insn_id}", self.find(insn_id))
        }
    }

    /// Replace `insn` with the new instruction `replacement`, which will get appended to `insns`.
    fn make_equal_to(&mut self, insn: InsnId, replacement: InsnId) {
        assert!(self.insns[insn.to_usize()].has_output(),
                "Don't use make_equal_to for instruction with no output");
        assert!(self.insns[replacement.to_usize()].has_output(),
                "Can't replace instruction that has output with instruction that has no output");
        // Don't push it to the block
        self.union_find.make_equal_to(insn, replacement);
    }

    pub fn type_of(&self, insn: InsnId) -> Type {
        debug_assert!(self.insns[insn.to_usize()].has_output());
        self.insn_types[self.union_find.find(insn).to_usize()]
    }

    /// Check if the type of `insn` is a subtype of `ty`.
    pub fn is_a(&self, insn: InsnId, ty: Type) -> bool {
        self.type_of(insn).is_subtype(ty)
    }

    /// Type an instruction an inline cfunc body handed back, if nothing has typed it yet. The
    /// inline pushes its instructions into a scratch block that is spliced in after the pass has
    /// already walked past this point, so the value it returns never gets typed by the pass
    /// itself.
    ///
    /// A `Param` is left alone. Several inlines are identity-shaped -- `Hash#[]=` and
    /// `Array#[]=` hand back the value that was stored, `Kernel#itself` and `Array#<<` hand back
    /// the receiver -- and the operand they hand back has been through [`Self::resolve`], so it
    /// can be a join block's parameter: the result of a polymorphic dispatch emitted moments
    /// earlier in this same pass, say. A parameter's type comes from the edges that jump to its
    /// block rather than from the instruction, so there is nothing to infer here; `infer_types`
    /// computes it once the whole function is in its final shape.
    fn infer_inlined_type(&mut self, insn: InsnId) {
        if !self.type_of(insn).bit_equal(types::Any) { return; }
        if matches!(self.insns[insn.to_usize()], Insn::Param) { return; }
        self.insn_types[insn.to_usize()] = self.infer_type(insn);
    }

    fn infer_type(&self, insn: InsnId) -> Type {
        debug_assert!(self.insns[insn.to_usize()].has_output());
        match &self.insns[insn.to_usize()] {
            Insn::Param => unimplemented!("params should not be present in block.insns"),
            Insn::LoadArg { val_type, .. } => *val_type,
            Insn::SetGlobal { .. } | Insn::Jump(_) | Insn::Entries { .. } | Insn::EntryPoint { .. }
            | Insn::Comment { .. }
            | Insn::CondBranch { .. } | Insn::Return { .. } | Insn::Throw { .. }
            | Insn::PatchPoint { .. } | Insn::SetIvar { .. } | Insn::SetClassVar { .. } | Insn::ArrayExtend { .. }
            | Insn::ArrayPush { .. } | Insn::SideExit { .. } | Insn::SetLocal { .. }
            | Insn::IncrCounter(_) | Insn::IncrCounterPtr { .. }
            | Insn::CheckInterrupts { .. } | Insn::BreakPoint | Insn::Unreachable
            | Insn::StoreField { .. } | Insn::WriteBarrier { .. } | Insn::HashAset { .. } | Insn::ArrayAset { .. }
            | Insn::IvarReprofile { .. } | Insn::ArrayAsetOrStore { .. }
            | Insn::PushInlineFrame { .. } | Insn::PopInlineFrame { .. } =>
                panic!("Cannot infer type of instruction with no output: {}. See Insn::has_output().", self.insns[insn.to_usize()]),
            Insn::Const { val: Const::Value(val) } => Type::from_value(*val),
            Insn::Const { val: Const::CBool(val) } => Type::from_cbool(*val),
            Insn::Const { val: Const::CInt8(val) } => Type::from_cint(types::CInt8, *val as i64),
            Insn::Const { val: Const::CInt16(val) } => Type::from_cint(types::CInt16, *val as i64),
            Insn::Const { val: Const::CInt32(val) } => Type::from_cint(types::CInt32, *val as i64),
            Insn::Const { val: Const::CInt64(val) } => Type::from_cint(types::CInt64, *val),
            Insn::Const { val: Const::CUInt8(val) } => Type::from_cint(types::CUInt8, *val as i64),
            Insn::Const { val: Const::CUInt16(val) } => Type::from_cint(types::CUInt16, *val as i64),
            Insn::Const { val: Const::CUInt32(val) } => Type::from_cint(types::CUInt32, *val as i64),
            Insn::Const { val: Const::CAttrIndex(val) } => Type::from_cint(types::CAttrIndex, *val as i64),
            Insn::Const { val: Const::CShape(val) } => Type::from_cint(types::CShape, val.0 as i64),
            Insn::Const { val: Const::CUInt64(val) } => Type::from_cint(types::CUInt64, *val as i64),
            Insn::Const { val: Const::CPtr(val) } => Type::from_cptr(*val),
            Insn::Const { val: Const::CDouble(val) } => Type::from_double(*val),
            Insn::Test { val } if self.type_of(*val).is_known_falsy() => Type::from_cbool(false),
            Insn::Test { val } if self.type_of(*val).is_known_truthy() => Type::from_cbool(true),
            Insn::Test { .. } => types::CBool,
            Insn::IsMethodCfunc { .. } => types::CBool,
            Insn::IsBitEqual { .. } => types::CBool,
            Insn::IsBitNotEqual { .. } => types::CBool,
            Insn::BoxBool { .. } => types::BoolExact,
            Insn::BoxFixnum { .. } => types::Fixnum,
            Insn::UnboxFixnum { val } => self
                .type_of(*val)
                .fixnum_value()
                .map_or(types::CInt64, |fixnum| Type::from_cint(types::CInt64, fixnum)),
            Insn::FixnumAref { .. } => types::Fixnum,
            Insn::StringCopy { .. } => types::StringExact,
            Insn::StringIntern { .. } => types::Symbol,
            Insn::StringConcat { .. } => types::StringExact,
            Insn::StringGetbyte { .. } => types::Fixnum,
            Insn::StringCoderangeOrScan { .. } => types::CInt64,
            Insn::StringSetbyteFixnum { .. } => types::Fixnum,
            Insn::StringAppend { .. } => types::StringExact,
            Insn::StringAppendCodepoint { .. } => types::StringExact,
            Insn::StringEqual { .. } => types::BoolExact,
            Insn::ToRegexp { .. } => types::RegexpExact,
            Insn::NewArray { .. } => types::ArrayExact,
            Insn::ArrayDup { .. } => types::ArrayExact,
            Insn::ArrayAref { .. } => types::BasicObject,
            Insn::ArrayArefOrNil { .. } => types::BasicObject,
            Insn::ArrayPop { .. } => types::BasicObject,
            Insn::ArrayLength { .. } => types::CInt64,
            Insn::AdjustBounds { .. } => types::CInt64,
            Insn::HashAref { .. } => types::BasicObject,
            Insn::NewHash { .. } => types::HashExact,
            Insn::HashDup { .. } => types::HashExact,
            Insn::NewRange { .. } => types::RangeExact,
            Insn::NewRangeFixnum { .. } => types::RangeExact,
            Insn::ObjectAlloc { .. } => types::HeapBasicObject,
            Insn::ObjectAllocClass { class, .. } => Type::from_class(*class),
            Insn::CCallWithFrame(insn) => insn.return_type,
            Insn::CCall { return_type, .. } => *return_type,
            Insn::CCallVariadic(insn) => insn.return_type,
            Insn::CheckMatch { .. } => types::BasicObject,
            Insn::GuardType { val, guard_type, .. } => self.type_of(*val).intersection(*guard_type),
            Insn::RefineType { val, new_type, .. } => self.type_of(*val).intersection(*new_type),
            &Insn::HasType { val, expected } if self.is_a(val, expected) => Type::from_cbool(true),
            &Insn::HasType { val, expected } if !self.type_of(val).could_be(expected) => Type::from_cbool(false),
            Insn::HasType { .. } => types::CBool,
            Insn::HasAncestor { .. } => types::CBool,
            Insn::GuardBitEquals { val, expected, .. } => self.type_of(*val).intersection(Type::from_const(*expected)),
            Insn::GuardAnyBitSet { val, .. } => self.type_of(*val),
            Insn::GuardNoBitsSet { val, .. } => self.type_of(*val),
            Insn::GuardNotRuby2KeywordsHash { val, .. } => self.type_of(*val),
            Insn::GuardLess { left, .. } => self.type_of(*left),
            Insn::GuardGreaterEq { left, .. } => self.type_of(*left),
            Insn::FixnumAdd  { .. } => types::Fixnum,
            Insn::FixnumSub  { .. } => types::Fixnum,
            Insn::FixnumMult { .. } => types::Fixnum,
            // FIXNUM_MIN / -1 overflows to a Bignum, so the result is Integer, not Fixnum.
            // Downstream Fixnum ops insert their own GuardType(Fixnum)
            Insn::FixnumDiv  { .. } => types::Integer,
            Insn::FixnumMod  { .. } => types::Fixnum,
            Insn::FloatAdd   { .. } => types::Float,
            Insn::FloatSub   { .. } => types::Float,
            Insn::FloatMul   { .. } => types::Float,
            Insn::FloatDiv   { .. } => types::Float,
            Insn::FloatToInt { .. } => types::Integer,
            Insn::FloatLt    { .. } => types::BoolExact,
            Insn::FloatLe    { .. } => types::BoolExact,
            Insn::FloatGt    { .. } => types::BoolExact,
            Insn::FloatGe    { .. } => types::BoolExact,
            Insn::FixnumEq   { .. } => types::BoolExact,
            Insn::FixnumNeq  { .. } => types::BoolExact,
            Insn::FixnumLt   { .. } => types::BoolExact,
            Insn::FixnumLe   { .. } => types::BoolExact,
            Insn::FixnumGt   { .. } => types::BoolExact,
            Insn::FixnumGe   { .. } => types::BoolExact,
            Insn::FixnumAnd  { .. } => types::Fixnum,
            Insn::FixnumOr   { .. } => types::Fixnum,
            Insn::FixnumXor  { .. } => types::Fixnum,
            Insn::IntAnd { .. } => types::CInt64,
            Insn::IntOr { left, .. } => self.type_of(*left).unspecialized(),
            Insn::FixnumLShift { .. } => types::Fixnum,
            Insn::FixnumRShift { .. } => types::Fixnum,
            Insn::PutSpecialObject { .. } => types::BasicObject,
            Insn::SendDirect(_) => types::BasicObject,
            Insn::Send { .. } => types::BasicObject,
            Insn::SendForward { .. } => types::BasicObject,
            Insn::InvokeSuper { .. } => types::BasicObject,
            Insn::InvokeSuperForward { .. } => types::BasicObject,
            Insn::InvokeBlock { .. } => types::BasicObject,
            Insn::InvokeBlockIfunc { .. } => types::BasicObject,
            Insn::InvokeProc { .. } => types::BasicObject,
            Insn::InvokeBlockIseqDirect { .. } => types::BasicObject,
            Insn::InvokeBuiltin { return_type, .. } => *return_type,
            Insn::Defined { pushval, .. } => Type::from_value(*pushval).union(types::NilClass),
            Insn::DefinedIvar { pushval, .. } => Type::from_value(*pushval).union(types::NilClass),
            Insn::GetConstant { .. } => types::BasicObject,
            Insn::GetConstantPath { .. } => types::BasicObject,
            Insn::IsBlockGiven { .. } => types::BoolExact,
            Insn::FixnumBitCheck { .. } => types::BoolExact,
            Insn::ArrayMax { .. } => types::BasicObject,
            Insn::ArrayMin { .. } => types::BasicObject,
            Insn::ArrayInclude { .. } => types::BoolExact,
            Insn::ArrayPackBuffer { .. } => types::String,
            Insn::DupArrayInclude { .. } => types::BoolExact,
            Insn::ArrayHash { .. } => types::Fixnum,
            Insn::GetGlobal { .. } => types::BasicObject,
            Insn::GetIvar { .. } => types::BasicObject,
            Insn::LoadPC => types::CPtr,
            Insn::LoadSP => types::CPtr,
            Insn::LoadEC => types::CPtr,
            Insn::GetEP { .. } => types::CPtr,
            Insn::LoadSelf => if self.self_is_heap_object { types::HeapBasicObject } else { types::BasicObject },
            &Insn::LoadField { return_type, .. } => return_type,
            Insn::UnwrapSvar { .. } => types::RubyValue,
            Insn::GetSpecialSymbol { .. } => types::StringExact.union(types::NilClass),
            Insn::GetSpecialNumber { .. } => types::StringExact.union(types::NilClass),
            Insn::Once { .. } => types::BasicObject,
            Insn::GetClassVar { .. } => types::BasicObject,
            Insn::ToNewArray { .. } => types::ArrayExact,
            Insn::ToArray { .. } => types::ArrayExact,
            Insn::ToHash { .. } => types::HashExact.union(types::NilClass),
            Insn::CheckArrayType { .. } => types::Array.union(types::NilClass),
            Insn::ToAryForExpand { .. } => types::Array,
            Insn::AnyToString { .. } => types::StringExact,
            Insn::IsBlockParamModified { .. } => types::CBool,
            Insn::GetBlockParam { .. } => types::BasicObject,
            // The type of Snapshot doesn't really matter; it's never materialized. It's used only
            // as a reference for FrameState, which we use to generate side-exit code.
            Insn::Snapshot { .. } => types::Any,
            Insn::IsA { .. } => types::BoolExact,
        }
    }

    /// Set self.param_types. They are copied to the param types of jit_entry_blocks.
    fn set_param_types(&mut self) {
        let iseq = self.iseq;
        let params = unsafe { iseq.params() };
        let param_size = params.size.to_usize();
        let rest_param_idx = iseq_rest_param_idx(params);

        self.param_types.push(types::BasicObject); // self
        for local_idx in 0..param_size {
            let param_type = if Some(local_idx as i32) == rest_param_idx {
                types::ArrayExact // Rest parameters are always ArrayExact
            } else {
                types::BasicObject
            };
            self.param_types.push(param_type);
        }
    }

    /// Copy self.param_types to the param types of jit_entry_blocks.
    fn copy_param_types(&mut self) {
        for jit_entry_block in self.jit_entry_blocks.iter() {
            let entry_params = self.blocks[jit_entry_block.to_usize()].params.iter();
            let param_types = self.param_types.iter();
            assert!(
                param_types.len() >= entry_params.len(),
                "param types should be initialized before type inference",
            );
            for (param, param_type) in std::iter::zip(entry_params, param_types) {
                // We know that function parameters are BasicObject or some subclass
                self.insn_types[param.to_usize()] = *param_type;
            }
        }
    }

    fn infer_types(&mut self) {
        // Reset all types
        self.insn_types.fill(types::Empty);

        // Fill entry parameter types
        self.copy_param_types();

        // Assign `new_type` to `insn` if it differs from the recorded type.
        // Returns `true` if a write actually happened, `false` if the type
        // Macro instead of closure so the borrow checker sees individual field
        // accesses rather than an `&mut self` borrow that conflicts with
        // `&self.insns` held by an outer match.
        macro_rules! set_type {
            ($insn:expr, $new_type:expr) => {{
                let insn = $insn;
                let new_type = $new_type;
                let old_type = self.insn_types[self.union_find.find(insn).to_usize()];
                if old_type.bit_equal(new_type) {
                    false
                } else {
                    self.insn_types[insn.to_usize()] = new_type;
                    true
                }
            }};
        }

        let mut reachable = BlockSet::with_capacity(self.blocks.len());
        reachable.insert(self.entries_block);

        // Repeatedly walk the graph in RPO order, computing types until fixpoint. For each
        // iteration over the CFG, track the following two attributes to detect the fixpoint:
        //
        // 1. if new types were inferred
        // 2. if back edges were traversed
        //
        // For point (1), if no new types were inferred it means no new information is available.
        // Further repetitions will not change the result.
        //
        // For point (2), if the RPO walk does not traverse a back edge, type information can only
        // be propagated forwards. It follows that a node's type can only be inferred from its
        // predecessors. RPO ordering ensures all of a node's predecessors have been processed;
        // therefore, a single walk of the RPO ordering will reach the fixpoint.
        let rpo = self.reverse_post_order();
        // Map BlockId -> rpo index. Used to detect back edge traversal. If an edge targets a block
        // with rpo index <= the current rpo index it's a back edge. Note that `rpo_order` must be
        // of size `self.blocks.len()` to support all possible block IDs; however, `rpo` only
        // includes reachable blocks. Any blocks not present in `rpo` default to `usize::MAX`.
        let mut rpo_order = vec![usize::MAX; self.blocks.len()];
        for (idx, &block_id) in rpo.iter().enumerate() {
            rpo_order[block_id.to_usize()] = idx;
        }
        // One scratch buffer for the per-edge argument type snapshots below,
        // instead of a fresh Vec for every jump on every round of the fixpoint.
        let mut arg_types: Vec<Type> = Vec::new();
        loop {
            let mut changed = false;
            let mut traversed_back_edge = false;
            let mut num_instructions = 0;
            for (rpo_index, &block) in rpo.iter().enumerate() {
                if !reachable.get(block) { continue; }
                for i in 0..self.blocks[block.to_usize()].insns.len() {
                    let insn_id = self.blocks[block.to_usize()].insns[i];
                    if self.insns[insn_id.to_usize()].counts_against_inlining_budget() {
                        num_instructions += 1;
                    }
                    // Instructions without output, including branch instructions, can't be targets
                    // of make_equal_to, so we don't need find() here.
                    let insn_type = match &self.insns[insn_id.to_usize()] {
                        Insn::CondBranch { val, if_true, if_false } => {
                            assert!(!self.type_of(*val).bit_equal(types::Empty));
                            if self.type_of(*val).could_be(Type::from_cbool(true)) {
                                reachable.insert(if_true.target);
                                // Snapshot arg types before any param updates so phi-style
                                // updates happen in parallel (the args of a self-loop may name
                                // params of `target` itself).
                                arg_types.clear();
                                arg_types.extend(if_true.args.iter().map(|a| self.type_of(*a)));
                                for (idx, arg_type) in arg_types.drain(..).enumerate() {
                                    let param = self.blocks[if_true.target.to_usize()].params[idx];
                                    changed |= set_type!(param, self.type_of(param).union(arg_type));
                                }
                                traversed_back_edge |= rpo_order[if_true.target.to_usize()] <= rpo_index;
                            }
                            if self.type_of(*val).could_be(Type::from_cbool(false)) {
                                reachable.insert(if_false.target);
                                arg_types.clear();
                                arg_types.extend(if_false.args.iter().map(|a| self.type_of(*a)));
                                for (idx, arg_type) in arg_types.drain(..).enumerate() {
                                    let param = self.blocks[if_false.target.to_usize()].params[idx];
                                    changed |= set_type!(param, self.type_of(param).union(arg_type));
                                }
                                traversed_back_edge |= rpo_order[if_false.target.to_usize()] <= rpo_index;
                            }
                            continue;
                        }
                        &Insn::Jump(BranchEdge { target, ref args }) => {
                            reachable.insert(target);
                            arg_types.clear();
                            arg_types.extend(args.iter().map(|a| self.type_of(*a)));
                            for (idx, arg_type) in arg_types.drain(..).enumerate() {
                                let param = self.blocks[target.to_usize()].params[idx];
                                changed |= set_type!(param, self.type_of(param).union(arg_type));
                            }
                            traversed_back_edge |= rpo_order[target.to_usize()] <= rpo_index;
                            continue;
                        }
                        Insn::Entries { targets } => {
                            for &target in targets {
                                reachable.insert(target);
                            }
                            continue;
                        }
                        insn if insn.has_output() => self.infer_type(insn_id),
                        _ => continue,
                    };
                    changed |= set_type!(insn_id, insn_type);
                }
            }
            if !changed || !traversed_back_edge {
                self.num_instructions = num_instructions;
                break;
            }
        }
    }

    fn chase_insn(&self, insn: InsnId) -> InsnId {
        let id = self.union_find.find_const(insn);
        match self.insns[id.to_usize()] {
            Insn::GuardType { val, .. }
            | Insn::GuardBitEquals { val, .. }
            | Insn::GuardAnyBitSet { val, .. }
            | Insn::GuardNoBitsSet { val, .. }
            | Insn::GuardNotRuby2KeywordsHash { val, .. } => self.chase_insn(val),
            | Insn::RefineType { val, .. } => self.chase_insn(val),
            _ => id,
        }
    }

    /// Return the profiled type of the HIR instruction at the given Snapshot, if it is known to be
    /// monomorphic or skewed polymorphic. This historical type
    /// record is not a guarantee and must be checked with a GuardType or similar.
    fn profiled_type_of_at(&self, insn: InsnId, state: InsnId) -> Option<ProfiledType> {
        match self.resolve_receiver_type_from_profile(insn, state) {
            ReceiverTypeResolution::Monomorphic { profiled_type }
            | ReceiverTypeResolution::SkewedPolymorphic { profiled_type } => Some(profiled_type),
            _ => None,
        }
    }

    /// The type to guard a `&blk` argument with so that it can be passed straight through as
    /// the callee's block handler, or `None` when it is not an ordinary `Proc`.
    ///
    /// `vm_caller_setup_arg_block` returns the block argument itself as the block handler only
    /// for a Proc; a Symbol, a `rb_block_param_proxy` or anything with a `to_proc` takes another
    /// path, so only an exact `Proc` qualifies. Exact class rather than `rb_obj_is_proc`, which
    /// also accepts Proc subclasses and objects with a singleton class: those keep the dynamic
    /// send instead of needing a second check here.
    fn proc_block_arg_type(&self, block_arg: InsnId, state: InsnId) -> Option<Type> {
        let profiled_type = self.profiled_type_of_at(block_arg, state)?;
        if profiled_type.flags().is_immediate() { return None; }
        if profiled_type.class() != unsafe { rb_cProc } { return None; }
        Some(Type::from_profiled_type(profiled_type))
    }

    /// Validate and normalize SendDirect arguments without emitting HIR.
    ///
    /// `rest_prepacked` is set when [`Self::try_forward_splat_to_rest`] already put the rest
    /// Array where the callee expects it, so the repacking below has to be skipped.
    fn build_send_direct_args(&self, args: &[InsnId], ci: *const rb_callinfo, iseq: IseqPtr, has_block: bool, block_arg_passthrough: bool, rest_prepacked: bool) -> Result<SendDirectCall, SendDirectFailure> {
        can_direct_send(iseq, ci, args, has_block, block_arg_passthrough)?;
        // A forwardable callee takes the caller's arguments verbatim: no reordering, no
        // synthesized keyword Hash, no rest packing. `gen_send_iseq_direct` copies them into
        // the callee frame and stores the callinfo in the `...` local.
        //
        // `can_direct_send_forwardable` has already rejected `VM_CALL_ARGS_BLOCKARG` call
        // sites, so a `&blk` passthrough never reaches here: the block argument is still on
        // the stack at this point and the callinfo we would store in `...` counts it out.
        if 0 != unsafe { iseq.params() }.flags.forwardable() {
            return Ok(SendDirectCall {
                args: args.iter().copied().map(SendDirectArg::Existing).collect(),
                kw_bits: 0,
                jit_entry_idx: 0,
            });
        }
        let args = args.iter().copied().map(SendDirectArg::Existing).collect();
        let (args, kw_bits) = Self::plan_send_direct_keyword_arguments(args, ci, iseq)
            .map_err(SendDirectFailure::new)?;
        let (args, jit_entry_idx) = if rest_prepacked {
            (args, 0)
        } else {
            Self::plan_send_direct_rest_parameter(args, iseq)
                .map_err(SendDirectFailure::new)?
        };

        Ok(SendDirectCall {
            args,
            kw_bits,
            jit_entry_idx,
        })
    }

    /// Materialize a validated SendDirect call in the selected runtime path.
    fn emit_send_direct_args(&mut self, block: BlockId, call: SendDirectCall, original_args: &[InsnId], state: InsnId) -> SendDirectArgs {
        let args: Vec<_> = call
            .args
            .into_iter()
            .map(|arg| self.emit_send_direct_arg(block, arg, state))
            .collect();

        // If args were reordered or synthesized, create a new snapshot with the updated stack.
        let send_state = if args != original_args {
            let new_state = self.frame_state(state).with_replaced_args(&args, original_args.len());
            self.push_insn(block, Insn::Snapshot { state: Box::new(new_state) })
        } else {
            state
        };

        SendDirectArgs {
            state: send_state,
            args,
            kw_bits: call.kw_bits,
            jit_entry_idx: call.jit_entry_idx,
        }
    }

    fn emit_send_direct_arg(&mut self, block: BlockId, arg: SendDirectArg, state: InsnId) -> InsnId {
        match arg {
            SendDirectArg::Existing(value) => value,
            SendDirectArg::Constant(value) => {
                self.push_insn(block, Insn::Const { val: Const::Value(value) })
            }
            SendDirectArg::KeywordHash(elements) => {
                let elements = elements
                    .into_iter()
                    .map(|arg| self.emit_send_direct_arg(block, arg, state))
                    .collect();
                self.push_insn(block, Insn::NewHash { elements, state })
            }
            SendDirectArg::RestArray(elements) => {
                let elements = elements
                    .into_iter()
                    .map(|arg| self.emit_send_direct_arg(block, arg, state))
                    .collect();
                self.push_insn(block, Insn::NewArray { elements, state })
            }
        }
    }

    /// Reorder keyword arguments to match the callee's expected order, and synthesize
    /// default values for any optional keywords not provided by the caller.
    ///
    /// The output always contains all of the callee's keyword arguments (required + optional),
    /// so the returned vec may be larger than the input args.
    ///
    /// Returns Ok with (processed_args, kw_bits) if successful, or Err with the fallback reason if not.
    /// - kw_bits: bitmask indicating which optional keywords were NOT provided by the caller
    ///            (used by checkkeyword to determine if non-constant defaults need evaluation)
    fn plan_send_direct_keyword_arguments(
        args: Vec<SendDirectArg>,
        ci: *const rb_callinfo,
        iseq: IseqPtr,
    ) -> Result<(Vec<SendDirectArg>, u32), SendFallbackReason> {
        let kwarg = unsafe { rb_vm_ci_kwarg(ci) };
        let callee_keyword = unsafe { rb_get_iseq_body_param_keyword(iseq) };
        if callee_keyword.is_null() {
            if kwarg.is_null() {
                // Neither caller nor callee have keywords - nothing to do
                return Ok((args, 0));
            }

            let params = unsafe { iseq.params() };
            let ci_flags = unsafe { rb_vm_ci_flag(ci) };
            if ci_flags & VM_CALL_KW_SPLAT != 0 {
                // Caller **kw is one runtime Hash, not explicit keyword slots, so
                // there is no static key/value list to repack here.
                return Err(SendDirectKeywordMismatch);
            }

            if params.flags.accepts_no_kwarg() != 0 || params.flags.ruby2_keywords() != 0 {
                // These callee modes need VM keyword setup even without a keyword table:
                // **nil rejects keywords, and ruby2_keywords requires RHASH_PASS_AS_KEYWORDS.
                return Err(SendDirectKeywordMismatch);
            }

            // Match vm_args.c's setup_parameters_complex via args_kw_argv_to_hash:
            // explicit caller keywords passed to a method with no keyword table
            // become one final positional Hash before regular parameter setup.
            let caller_kw_count = unsafe { get_cikw_keyword_len(kwarg) } as usize;
            let kw_args_start = args.len() - caller_kw_count;
            let mut processed_args = args;
            let keyword_values = processed_args.split_off(kw_args_start);
            let mut elements = Vec::with_capacity(caller_kw_count * 2);
            for (i, value) in keyword_values.into_iter().enumerate() {
                let keyword = unsafe { get_cikw_keywords_idx(kwarg, i as i32) };
                elements.push(SendDirectArg::Constant(keyword));
                elements.push(value);
            }

            processed_args.push(SendDirectArg::KeywordHash(elements));
            return Ok((processed_args, 0));
        }

        // kwarg may be null if caller passes no keywords but callee has optional keywords
        let caller_kw_count = if kwarg.is_null() { 0 } else { (unsafe { get_cikw_keyword_len(kwarg) }) as usize };
        let callee_kw_count = unsafe { (*callee_keyword).num } as usize;

        // When there are 31+ keywords, CRuby uses a hash instead of a fixnum bitmask
        // for kw_bits. Fall back to VM dispatch for this rare case.
        if callee_kw_count >= VM_KW_SPECIFIED_BITS_MAX as usize {
            return Err(SendDirectTooManyKeywords);
        }

        let callee_kw_required = unsafe { (*callee_keyword).required_num } as usize;
        let callee_kw_table = unsafe { (*callee_keyword).table };
        let default_values = unsafe { (*callee_keyword).default_values };

        // A `**rest` parameter soaks up whatever the keyword table does not name, so the
        // caller is free to pass more keywords than there are slots. Without one, it is not.
        let has_kwrest = 0 != unsafe { iseq.params() }.flags.has_kwrest();
        if !has_kwrest && caller_kw_count > callee_kw_count {
            return Err(SendDirectKeywordCountMismatch);
        }

        // The keyword arguments are the last arguments in the args vector.
        let kw_args_start = args.len() - caller_kw_count;

        // Build a mapping from caller keywords to their positions.
        let mut caller_kw_order: Vec<ID> = Vec::with_capacity(caller_kw_count);
        if !kwarg.is_null() {
            for i in 0..caller_kw_count {
                let sym = unsafe { get_cikw_keywords_idx(kwarg, i as i32) };
                let id = unsafe { rb_sym2id(sym) };
                caller_kw_order.push(id);
            }
        }

        // Verify all caller keywords are expected by callee (no unknown keywords).
        // Without **kwrest, unexpected keywords should raise ArgumentError at runtime.
        if !has_kwrest {
            for &caller_id in &caller_kw_order {
                let mut found = false;
                for i in 0..callee_kw_count {
                    let expected_id = unsafe { *callee_kw_table.add(i) };
                    if caller_id == expected_id {
                        found = true;
                        break;
                    }
                }
                if !found {
                    // Caller is passing an unknown keyword - this will raise ArgumentError.
                    // Fall back to VM dispatch to handle the error.
                    return Err(SendDirectKeywordMismatch);
                }
            }
        }

        // Move caller keyword values out of the positional prefix. Wrap them in
        // Option so reordering can take each value without cloning it.
        let mut processed_args = args;
        let keyword_values = processed_args.split_off(kw_args_start);
        let mut keyword_values: Vec<_> = keyword_values.into_iter().map(Some).collect();

        // Reorder keyword arguments to match callee expectation.
        // Track which optional keywords were not provided via kw_bits.
        let mut kw_bits: u32 = 0;
        let mut reordered_kw_args = Vec::with_capacity(callee_kw_count);
        for i in 0..callee_kw_count {
            let expected_id = unsafe { *callee_kw_table.add(i) };

            // Find where this keyword is in the caller's order
            let mut found = false;
            for (j, &caller_id) in caller_kw_order.iter().enumerate() {
                if caller_id == expected_id {
                    reordered_kw_args.push(keyword_values[j].take().unwrap());
                    found = true;
                    break;
                }
            }

            if !found {
                // Required keyword not provided by caller which will raise an ArgumentError.
                if i < callee_kw_required {
                    return Err(SendDirectMissingKeyword);
                }

                // Optional keyword not provided - use default value
                let default_idx = i - callee_kw_required;
                let default_value = unsafe { *default_values.add(default_idx) };

                if default_value == Qundef {
                    // Non-constant default (e.g., `def foo(a: compute())`).
                    // Set the bit so checkkeyword knows to evaluate the default at runtime.
                    // Push Qnil as a placeholder; the callee's checkkeyword will detect this
                    // and branch to evaluate the default expression.
                    kw_bits |= 1 << default_idx;
                    reordered_kw_args.push(SendDirectArg::Constant(Qnil));
                } else {
                    // Constant default value - use it directly
                    reordered_kw_args.push(SendDirectArg::Constant(default_value));
                }
            }
        }

        // Replace the keyword arguments with the reordered ones.
        processed_args.extend(reordered_kw_args);

        // `**rest` takes the keywords the loop above did not claim, in the order the caller
        // wrote them, which is what `make_rest_kw_hash` builds from the leftover slots.
        if has_kwrest {
            let mut leftover = Vec::with_capacity(keyword_values.len() * 2);
            for (idx, value) in keyword_values.into_iter().enumerate() {
                let Some(value) = value else { continue };
                let keyword = unsafe { get_cikw_keywords_idx(kwarg, idx as i32) };
                leftover.push(SendDirectArg::Constant(keyword));
                leftover.push(value);
            }
            // A callee with no named keywords goes through `args_setup_kw_rest_parameter`
            // instead of `args_setup_kw_parameters`, and that one skips the allocation for
            // an anonymous `**` with nothing to collect: the slot is left nil, not `{}`.
            let params = unsafe { iseq.params() };
            let nil_anon_kwrest = leftover.is_empty()
                && 0 == params.flags.has_kw()
                && 0 != params.flags.anon_kwrest();
            processed_args.push(if nil_anon_kwrest {
                SendDirectArg::Constant(Qnil)
            } else {
                SendDirectArg::KeywordHash(leftover)
            });
        }

        Ok((processed_args, kw_bits))
    }

    /// Compute the positional optional entry index and pack positional arguments
    /// into a rest array when the callee has a *rest parameter.
    /// Mirrors vm_args.c's setup_parameters_complex / args_setup_rest_parameter:
    /// positional arguments between required/optional and post parameters become
    /// the callee's *rest array before entering the method body.
    ///
    /// The input args must already have keyword arguments normalized to the callee's
    /// keyword table order by plan_send_direct_keyword_arguments. This function only reshapes
    /// the positional section before those keyword slots.
    ///
    /// Returns Ok with (processed_args, jit_entry_idx) if successful, or Err with
    /// the fallback reason if direct send isn't possible.
    /// - processed_args: arguments to use for SendDirect after optional rest packing
    /// - jit_entry_idx: number of positional optional parameters provided by the caller
    fn plan_send_direct_rest_parameter(
        args: Vec<SendDirectArg>,
        iseq: IseqPtr,
    ) -> Result<(Vec<SendDirectArg>, u16), SendFallbackReason> {
        let params = unsafe { iseq.params() };
        let lead_num = params.lead_num as usize;
        let opt_num = params.opt_num as usize;
        let post_num = params.post_num as usize;
        let kw_num = callee_kw_num(iseq) + usize::from(params.flags.has_kwrest() != 0);

        let positional_argc = args.len().checked_sub(kw_num).ok_or(ArgcParamMismatch)?;
        let min_positional_argc = lead_num + post_num;
        if positional_argc < min_positional_argc { return Err(ArgcParamMismatch); }

        // For computing the optional positional entry point, only count positional
        // args and exclude the always-present lead and post slots. Do this before
        // rest packing changes the SendDirect argument count.
        // See: vm_args.c's setup_parameters_complex and args_setup_opt_parameters.
        let passed_opt_num = (positional_argc - min_positional_argc).min(opt_num);
        let jit_entry_idx = passed_opt_num.try_into().map_err(|_| OperandTooLarge)?;

        // Methods without *rest still need the jit_entry_idx computed above,
        // but their positional args do not need repacking.
        if params.flags.has_rest() == 0 {
            return Ok((args, jit_entry_idx));
        }

        // Rebuild [lead, filled opts, rest elements..., post, kw...] into the
        // argument shape expected by SendDirect:
        // [lead, filled opts, rest array, post, kw...].
        let rest_start = lead_num + passed_opt_num;
        let rest_end = positional_argc - post_num;
        let mut packed_args = args;
        let mut rest_elements = packed_args.split_off(rest_start);
        let suffix = rest_elements.split_off(rest_end - rest_start);
        packed_args.push(SendDirectArg::RestArray(rest_elements));
        packed_args.extend(suffix);

        Ok((packed_args, jit_entry_idx))
    }

    /// Resolve the receiver type for method dispatch optimization.
    ///
    /// Takes the receiver's Type, receiver HIR instruction, and ISEQ instruction index.
    /// First checks if the receiver's class is statically known, otherwise consults profile data.
    ///
    /// Returns:
    /// - `StaticallyKnown` if the receiver's exact class is known at compile-time
    /// - Result of [`Self::resolve_receiver_type_from_profile`] if we need to check profile data
    fn resolve_receiver_type(&self, recv: InsnId, recv_type: Type, state: InsnId) -> ReceiverTypeResolution {
        match self.resolve_receiver_type_from_profile(recv, state) {
            ReceiverTypeResolution::NoProfile => {
                // An earlier guard may have narrowed this value to a profiled type. That profile
                // still carries the shape, which `Type` does not, so prefer it over the static
                // class: it is what lets attr_reader calls in inlined callees (whose own ISEQ was
                // never profiled) load the ivar directly instead of calling rb_ivar_get.
                if let Some(profiled_type) = self.recorded_profiled_type(recv) {
                    if recv_type.is_subtype(Type::from_profiled_type(profiled_type)) {
                        return ReceiverTypeResolution::Monomorphic { profiled_type };
                    }
                }
                // Use known type information as a fallback because it doesn't have shape
                // information (and we can generally eliminate duplicate guards).
                if let Some(class) = recv_type.runtime_exact_ruby_class() {
                    ReceiverTypeResolution::StaticallyKnown { class }
                } else {
                    ReceiverTypeResolution::NoProfile
                }
            }
            resolution => resolution,
        }
    }

    /// Best-effort static type of `insn` while the HIR is still being built, where `type_of()`
    /// has not been filled in yet and answers `Any` for everything. Returns `BasicObject` for
    /// block parameters, whose type is only known once all predecessors have been visited.
    fn type_at_construction(&self, insn: InsnId) -> Type {
        if matches!(self.insns[insn.to_usize()], Insn::Param) {
            return types::BasicObject;
        }
        self.infer_type(insn)
    }

    /// Pick the single shape to compile an `expandarray` site for. See [`ExpandArrayShape`].
    fn expandarray_shape(&self, profiles: &ProfileOracle, val: InsnId, state: InsnId) -> ExpandArrayShape {
        // The final version of an ISEQ can't recompile, so it has to handle every value.
        if self.policy.no_side_exits {
            return ExpandArrayShape::General;
        }

        // Static types beat the profile. `a, b = nil`, the preamble of every Ragel-generated
        // parser, destructures a literal, and we want to compile it well even with no profile.
        let val_type = self.type_at_construction(val);
        if val_type.is_subtype(types::ArrayExact) {
            return ExpandArrayShape::Array;
        }
        if !val_type.could_be(types::Array) {
            return ExpandArrayShape::Scalar;
        }

        let summary = self.profile_summary(profiles, val, state);
        if summary.bucket(0).is_empty() {
            // The instruction never ran during the profiling window. Exit and recompile once we
            // know what it destructures.
            ExpandArrayShape::NoProfile
        } else if !summary.is_monomorphic() {
            ExpandArrayShape::General
        } else if summary.bucket(0).is_array_exact() {
            ExpandArrayShape::Array
        } else {
            ExpandArrayShape::Scalar
        }
    }

    fn profile_summary(&self, profiles: &ProfileOracle, recv: InsnId, state: InsnId) -> TypeDistributionSummary {
        let Some(entries) = profiles.get(state) else {
            return TypeDistributionSummary::empty();
        };
        let recv = self.chase_insn(recv);
        for (entry_insn, entry_type_summary) in entries {
            if self.chase_insn(*entry_insn) == recv {
                return entry_type_summary.clone();
            }
        }
        TypeDistributionSummary::empty()
    }

    fn polymorphic_summary(&self, profiles: &ProfileOracle, recv: InsnId, state: InsnId) -> Option<TypeDistributionSummary> {
        let Some(entries) = profiles.get(state) else {
            return None;
        };
        let recv = self.chase_insn(recv);
        for (entry_insn, entry_type_summary) in entries {
            if self.chase_insn(*entry_insn) == recv {
                if entry_type_summary.is_polymorphic() || entry_type_summary.is_skewed_polymorphic() {
                    return Some(entry_type_summary.clone());
                }
                return None;
            }
        }
        None
    }

    /// Return the receiver profile of a call site when it is worth guarding the profiled types
    /// in-line, with a dynamic send as the fallthrough.
    ///
    /// Polymorphic profiles always qualify: every observed type has a bucket, so the chain
    /// covers the whole profile. Megamorphic profiles qualify when the buckets still cover
    /// most of the observed executions: such a site is only megamorphic because it saw more
    /// types than there are buckets, which for a skewed distribution still leaves the buckets
    /// serving nearly every call. Sending those dynamically because a handful of executions
    /// used a rare receiver class gives up on the common case.
    fn send_chain_plan(&self, profiles: &ProfileOracle, recv: InsnId, state: InsnId, cd: *const rb_call_data) -> Option<SendChainPlan> {
        let entries = profiles.get(state)?;
        let recv = self.chase_insn(recv);
        for (entry_insn, entry_type_summary) in entries {
            if self.chase_insn(*entry_insn) != recv { continue; }
            if entry_type_summary.is_polymorphic() || entry_type_summary.is_skewed_polymorphic() {
                return Some(SendChainPlan::Classes(entry_type_summary.clone()));
            }
            if entry_type_summary.is_megamorphic() || entry_type_summary.is_skewed_megamorphic() {
                // Prefer one guard over the method the site really dispatches, when there is
                // one: it covers subclasses the profile never saw, which is most of what a
                // megamorphic site sees. Try it even below the coverage threshold, since the
                // sites with the least bucket coverage are exactly the ones a class chain
                // helps least.
                if let Some(dispatch) = ancestor_dispatch_target(entry_type_summary, cd) {
                    return Some(SendChainPlan::Ancestor(dispatch));
                }
                // Everything that did not fit in a bucket has to take the fallthrough.
                let covered = entry_type_summary.coverage(|_, profiled_type| !profiled_type.is_empty());
                if covered >= CHAIN_COVERAGE_THRESHOLD {
                    return Some(SendChainPlan::Classes(entry_type_summary.clone()));
                }
            }
            return None;
        }
        None
    }

    fn monomorphic_summary(&self, profiles: &ProfileOracle, recv: InsnId, state: InsnId) -> Option<ProfiledType> {
        let Some(entries) = profiles.get(state) else {
            return None;
        };
        let recv = self.chase_insn(recv);
        for (entry_insn, entry_type_summary) in entries {
            if self.chase_insn(*entry_insn) == recv {
                if entry_type_summary.is_monomorphic() {
                    return Some(entry_type_summary.bucket(0));
                }
                return None;
            }
        }
        None
    }

    /// Resolve the receiver type for method dispatch optimization from profile data.
    ///
    /// Returns:
    /// - `Monomorphic`/`SkewedPolymorphic` if we have usable profile data
    /// - `Polymorphic` if the receiver has multiple types
    /// - `Megamorphic`/`SkewedMegamorphic` if the receiver has too many types to optimize
    ///   (SkewedMegamorphic may be optimized in the future, but for now we don't)
    /// - `NoProfile` if we have no type information
    fn resolve_receiver_type_from_profile(&self, recv: InsnId, state: InsnId) -> ReceiverTypeResolution {
        let Some(profiles) = self.profiles.as_ref() else {
            return ReceiverTypeResolution::NoProfile;
        };
        let Some(entries) = profiles.get(state) else {
            return ReceiverTypeResolution::NoProfile;
        };
        let recv = self.chase_insn(recv);

        for (entry_insn, entry_type_summary) in entries {
            if self.chase_insn(*entry_insn) == recv {
                if entry_type_summary.is_monomorphic() {
                    let profiled_type = entry_type_summary.bucket(0);
                    return ReceiverTypeResolution::Monomorphic { profiled_type };
                } else if entry_type_summary.is_skewed_polymorphic() {
                    let profiled_type = entry_type_summary.bucket(0);
                    return ReceiverTypeResolution::SkewedPolymorphic { profiled_type };
                } else if entry_type_summary.is_skewed_megamorphic() {
                    let profiled_type = entry_type_summary.bucket(0);
                    return ReceiverTypeResolution::SkewedMegamorphic { profiled_type };
                } else if entry_type_summary.is_polymorphic() {
                    return ReceiverTypeResolution::Polymorphic;
                } else if entry_type_summary.is_megamorphic() {
                    return ReceiverTypeResolution::Megamorphic;
                }
            }
        }

        ReceiverTypeResolution::NoProfile
    }

    pub fn assume_expected_cfunc(&mut self, block: BlockId, class: VALUE, method_id: ID, cfunc: *mut c_void, state: InsnId) -> bool {
        let cme = unsafe { rb_callable_method_entry(class, method_id) };
        if cme.is_null() { return false; }
        let def_type = unsafe { get_cme_def_type(cme) };
        if def_type != VM_METHOD_TYPE_CFUNC { return false; }
        if unsafe { get_mct_func(get_cme_def_body_cfunc(cme)) } != cfunc {
            return false;
        }
        self.gen_patch_points_for_optimized_ccall(block, class, method_id, cme, state);
        if !self.assume_no_singleton_classes(block, class, state) {
            return false;
        }
        true
    }

    pub fn likely_a(&self, val: InsnId, ty: Type, state: InsnId) -> bool {
        if self.type_of(val).is_subtype(ty) {
            return true;
        }
        let Some(profiled_type) = self.profiled_type_of_at(val, state) else {
            return false;
        };
        Type::from_profiled_type(profiled_type).is_subtype(ty)
    }

    pub fn coerce_to(&mut self, block: BlockId, val: InsnId, guard_type: Type, state: InsnId) -> InsnId {
        if self.is_a(val, guard_type) { return val; }
        self.push_insn(block, Insn::GuardType { val, guard_type, state, recompile: None })
    }

    pub fn guard_type_recompile(&mut self, block: BlockId, val: InsnId, guard_type: Type, state: InsnId, recompile: Recompile) -> InsnId {
        let result = self.push_insn(block, Insn::GuardType { val, guard_type, state, recompile: Some(recompile) });
        self.insn_types[result.to_usize()] = self.infer_type(result);
        result
    }

    /// Guard `val` against the class of `profiled_type` and remember the full profiled type
    /// (including its shape) for the guarded value. See [`Function::guarded_profiled_types`].
    fn guard_profiled_type(&mut self, block: BlockId, val: InsnId, profiled_type: ProfiledType, state: InsnId) -> InsnId {
        let result = self.guard_type_recompile(block, val, Type::from_profiled_type(profiled_type), state, Recompile);
        self.record_profiled_type(result, profiled_type);
        result
    }

    /// Remember that `val` was narrowed to `profiled_type`'s class, so later passes can recover
    /// the shape that `Type` cannot carry.
    fn record_profiled_type(&mut self, val: InsnId, profiled_type: ProfiledType) {
        if !profiled_type.is_empty() {
            self.guarded_profiled_types.insert(val, profiled_type);
        }
    }

    /// Recover the profiled type recorded for `val` or for any value it was guarded/refined from.
    /// Guards only narrow the class, so walking the guard chain is safe: every link describes the
    /// same run-time value.
    fn recorded_profiled_type(&self, val: InsnId) -> Option<ProfiledType> {
        let mut insn = val;
        loop {
            let id = self.union_find.find_const(insn);
            if let Some(&profiled_type) = self.guarded_profiled_types.get(&id) {
                return Some(profiled_type);
            }
            match self.insns[id.to_usize()] {
                Insn::GuardType { val, .. }
                | Insn::GuardBitEquals { val, .. }
                | Insn::GuardAnyBitSet { val, .. }
                | Insn::GuardNoBitsSet { val, .. }
                | Insn::RefineType { val, .. } => insn = val,
                _ => return None,
            }
        }
    }

    /// Every profiled type recorded for `recv` at `state` that shares `profiled_type`'s class,
    /// most frequent first and de-duplicated by shape. A polymorphic dispatch arm branches on the
    /// class alone, so its profile can hold several shapes for that class; an ivar dispatch wants
    /// an arm for each of them rather than sending all but one to the C fallback. Shapes an ivar
    /// dispatch cannot index (too-complex, immediates) are dropped, and `profiled_type` itself is
    /// always first so callers keep the type they already validated.
    fn profiled_shape_variants(&self, recv: InsnId, state: InsnId, profiled_type: ProfiledType) -> Vec<ProfiledType> {
        let mut variants = vec![profiled_type];
        let Some(profiles) = self.profiles.as_ref() else { return variants };
        let Some(entries) = profiles.get(state) else { return variants };
        let expected = Type::from_profiled_type(profiled_type);
        let recv = self.chase_insn(recv);
        for (entry_insn, summary) in entries {
            if self.chase_insn(*entry_insn) != recv { continue; }
            for &other in summary.buckets() {
                if other.is_empty() { continue; }
                if other.flags().is_immediate() || other.shape().is_complex() { continue; }
                if !Type::from_profiled_type(other).bit_equal(expected) { continue; }
                if variants.iter().any(|kept| kept.shape() == other.shape()) { continue; }
                variants.push(other);
            }
            break;
        }
        variants
    }

    fn count_complex_call_features(&mut self, block: BlockId, ci_flags: c_uint, state: InsnId) {
        use Counter::*;
        if 0 != ci_flags & VM_CALL_ARGS_SPLAT {
            self.count(block, complex_arg_pass_caller_splat);
            self.count_caller_splat_profile(block, state);
        }
        if 0 != ci_flags & VM_CALL_ARGS_BLOCKARG  { self.count(block, complex_arg_pass_caller_blockarg);   }
        if 0 != ci_flags & VM_CALL_KWARG          { self.count(block, complex_arg_pass_caller_kwarg);      }
        if 0 != ci_flags & VM_CALL_KW_SPLAT       { self.count(block, complex_arg_pass_caller_kw_splat);   }
        if 0 != ci_flags & VM_CALL_TAILCALL       { self.count(block, complex_arg_pass_caller_tailcall);   }
        if 0 != ci_flags & VM_CALL_SUPER          { self.count(block, complex_arg_pass_caller_super);      }
        if 0 != ci_flags & VM_CALL_ZSUPER         { self.count(block, complex_arg_pass_caller_zsuper);     }
        if 0 != ci_flags & VM_CALL_FORWARDING     { self.count(block, complex_arg_pass_caller_forwarding); }
    }

    /// Try to turn `foo(a, *b)` into a direct call by handing `b` straight to `foo`'s rest
    /// parameter, for the case where the splat lands exactly on that parameter:
    ///
    ///     def foo(x, *rest) = ...
    ///     foo(a, *b)                 # rest is b's elements, in order, and nothing else
    ///
    /// `setup_parameters_complex` flattens the splat onto the stack and then packs the same
    /// elements straight back into a fresh Array for `rest`, so the whole round trip is
    /// `rest = b.dup` no matter how long `b` is. That is what makes this worth doing separately
    /// from [`Self::try_expand_splat_args`]: expanding needs a length to bake in, so a call site
    /// whose splat length varies -- a filter pipeline forwarding `*args` through two frames, say
    /// -- can never take it, while forwarding does not care.
    ///
    /// Returns arguments already in `SendDirect` shape (`[leads..., rest array]`), so the caller
    /// must skip the repacking [`Self::setup_rest_parameter`] would otherwise do.
    fn try_forward_splat_to_rest(&mut self, block: BlockId, args: &[InsnId], iseq: IseqPtr, state: InsnId) -> Option<Vec<InsnId>> {
        let &splat = args.last()?;
        let params = unsafe { iseq.params() };
        // Only the shape where the splat *is* the rest parameter: any optional, post, or keyword
        // parameter would take some of the splatted elements for itself, which needs the length.
        if params.flags.has_rest() == 0
            || params.opt_num != 0
            || params.post_num != 0
            || callee_kw_num(iseq) != 0
            || params.flags.has_kw() != 0
            || params.flags.has_kwrest() != 0
            || params.flags.has_block() != 0
            || params.flags.forwardable() != 0
            || params.flags.ruby2_keywords() != 0
        {
            return None;
        }
        // The caller's own positional arguments have to at least fill the leading parameters;
        // otherwise the splat's first elements are consumed by them and the split depends on the
        // length again. Anything the caller passes past the leads belongs to the rest parameter,
        // ahead of the splatted elements.
        let lead_num = params.lead_num as usize;
        let positional_argc = args.len() - 1;
        if positional_argc < lead_num {
            return None;
        }
        // Bound the code a single call site grows by, the way the expansion path does.
        if positional_argc - lead_num > MAX_SPLAT_EXPANSION {
            return None;
        }
        // The Array guard below has no recompile behind it, so a call site that splats something
        // else -- an Array subclass, or a `to_a` conversion -- would side-exit on every call
        // instead of taking the dynamic send it takes today. Only rewrite what was profiled as an
        // Array. The expansion path gets this from its length profile, which is only recorded for
        // Arrays.
        if !self.likely_a(splat, types::ArrayExact, state) {
            return None;
        }

        let array = self.coerce_to(block, splat, types::ArrayExact, state);
        // A ruby2_keywords-flagged Hash last in the splat makes CALLER_SETUP_ARG reinterpret it
        // as keywords rather than leave it in the rest array. The expansion path guards the
        // element it read out; here the element has to be read just for the guard, and reading
        // past the end of an empty array is nil, which the guard passes.
        let length = self.push_insn(block, Insn::ArrayLength { array });
        let last_index = self.push_insn(block, Insn::Const { val: Const::CInt64(-1) });
        let last_index = self.push_insn(block, Insn::AdjustBounds { index: last_index, length });
        let last = self.push_insn(block, Insn::ArrayArefOrNil { array, index: last_index, length });
        self.push_insn(block, Insn::GuardNotRuby2KeywordsHash { val: last, state, recompile: Some(Recompile) });

        // The rest parameter is a fresh Array the callee may mutate, and the splat operand is
        // whatever expression produced it, so it has to be copied. Positional arguments past the
        // leading parameters go in front of the splatted elements, which is what the interpreter's
        // flatten-then-repack does.
        let extras = &args[lead_num..args.len() - 1];
        let rest = if extras.is_empty() {
            self.push_insn(block, Insn::ArrayDup { val: array, state })
        } else {
            let rest = self.push_insn(block, Insn::NewArray { elements: extras.to_vec(), state });
            self.push_insn(block, Insn::ArrayExtend { left: rest, right: array, state });
            rest
        };
        let mut forwarded = args[..lead_num].to_vec();
        forwarded.push(rest);
        Some(forwarded)
    }

    /// Try to turn `foo(a, *b)` into `foo(a, b[0], ..., b[n-1])` using the splat length the
    /// interpreter observed at this call site. Returns the new argument list, with guards
    /// emitted into `block`, or None when there is no usable profile.
    ///
    /// The caller keeps `state` (the pre-send Snapshot, which still has the splat array on the
    /// stack) as the deopt target, so a failing guard just re-runs the splat send in the
    /// interpreter.
    fn try_expand_splat_args(&mut self, block: BlockId, args: &[InsnId], state: InsnId) -> Option<Vec<InsnId>> {
        let &splat = args.last()?;
        let frame_state = self.frame_state(state);
        // A ruby2_keywords method forwarding its rest array is the one place a flagged Hash
        // shows up regularly. Expanding there would guard, fail, and side-exit on every call,
        // so leave those call sites on the dynamic path instead.
        if 0 != unsafe { frame_state.iseq.params() }.flags.ruby2_keywords() {
            return None;
        }
        let summary = get_or_create_iseq_payload(frame_state.iseq).profile.get_splat_length_summary(frame_state.insn_idx)?;
        // Only speculate on a length that has never varied. A guard failure exits to the
        // interpreter, so a call site with several lengths is better off dispatching dynamically.
        if !summary.is_monomorphic() {
            return None;
        }
        // `None` means a non-Array was splatted (`to_a` conversion), which we don't handle.
        let length = (*summary.buckets().first()?)?;
        let length: usize = length.try_into().ok()?;
        // Bound how much code a single call site can grow.
        if length > MAX_SPLAT_EXPANSION {
            return None;
        }

        let array = self.coerce_to(block, splat, types::ArrayExact, state);
        let array_length = self.push_insn(block, Insn::ArrayLength { array });
        self.push_insn(block, Insn::GuardBitEquals {
            val: array_length,
            expected: Const::CInt64(length as i64),
            reason: Box::new(SideExitReason::SplatLengthChanged),
            state,
            recompile: Some(Recompile),
        });

        let mut expanded = args[..args.len() - 1].to_vec();
        for i in 0..length {
            let index = self.push_insn(block, Insn::Const { val: Const::CInt64(i as i64) });
            expanded.push(self.push_insn(block, Insn::ArrayAref { array, index }));
        }
        // A ruby2_keywords-flagged Hash in the last position makes the interpreter treat it as
        // keywords instead of a positional argument, so guard that it isn't one.
        if let Some(&last) = expanded.last() {
            self.push_insn(block, Insn::GuardNotRuby2KeywordsHash { val: last, state, recompile: Some(Recompile) });
        }
        Some(expanded)
    }

    fn count_caller_splat_profile(&mut self, block: BlockId, state: InsnId) {
        let (iseq, insn_idx) = {
            let frame_state = self.frame_state_ref(state);
            (frame_state.iseq, frame_state.insn_idx)
        };
        let summary = get_or_create_iseq_payload(iseq).profile.get_splat_length_summary(insn_idx);
        let counter = match summary {
            None => Counter::caller_splat_profile_no_profiles,
            Some(summary) if summary.is_monomorphic() => Counter::caller_splat_profile_monomorphic,
            Some(summary) if summary.is_polymorphic() => Counter::caller_splat_profile_polymorphic,
            Some(summary) if summary.is_skewed_polymorphic() => Counter::caller_splat_profile_skewed_polymorphic,
            Some(summary) if summary.is_megamorphic() => Counter::caller_splat_profile_megamorphic,
            Some(summary) if summary.is_skewed_megamorphic() => Counter::caller_splat_profile_skewed_megamorphic,
            Some(_) => unreachable!(),
        };
        self.count(block, counter);
    }

    /// Return true if `self_val` is a known frozen object of a type whose `bop` is still the
    /// basic operation, i.e. if `rewrite_if_frozen` would elide the call.
    fn can_rewrite_if_frozen(&self, self_val: InsnId, klass: u32, bop: u32) -> bool {
        (unsafe { rb_BASIC_OP_UNREDEFINED_P(bop, klass) })
            && self.type_of(self_val).ruby_object().is_some_and(|obj| obj.is_frozen())
    }

    fn rewrite_if_frozen(&mut self, block: BlockId, orig_insn_id: InsnId, self_val: InsnId, klass: u32, bop: u32, state: InsnId) {
        debug_assert!(self.can_rewrite_if_frozen(self_val, klass, bop));
        self.push_insn(block, Insn::PatchPoint { invariant: Invariant::BOPRedefined { klass, bop }, state });
        self.make_equal_to(orig_insn_id, self_val);
    }

    pub fn try_inline_object_alloc(&mut self, block: BlockId, recv: InsnId, state: InsnId) -> Option<InsnId> {
        let recv_type = self.type_of(recv);
        if recv_type.is_subtype(types::Class) {
            if let Some(class) = recv_type.ruby_object() {
                // See class_get_alloc_func in object.c; if the class isn't initialized, is
                // a singleton class, or has a custom allocator, ObjectAlloc might raise an
                // exception or run arbitrary code.
                if class_has_leaf_allocator(class) {
                    return Some(self.push_insn(block, Insn::ObjectAllocClass { class, state }));
                }
            }
        }
        None
    }

    /// Return the (redefinition flag, BOP) pair to elide a no-argument `freeze` call on
    /// `self_val`, if the receiver is a known frozen object of a supported type.
    fn freeze_rewrite_bop(&self, self_val: InsnId) -> Option<(u32, u32)> {
        let klass = if self.is_a(self_val, types::StringExact) {
            STRING_REDEFINED_OP_FLAG
        } else if self.is_a(self_val, types::ArrayExact) {
            ARRAY_REDEFINED_OP_FLAG
        } else if self.is_a(self_val, types::HashExact) {
            HASH_REDEFINED_OP_FLAG
        } else {
            return None;
        };
        self.can_rewrite_if_frozen(self_val, klass, BOP_FREEZE).then_some((klass, BOP_FREEZE))
    }

    /// Same as `freeze_rewrite_bop`, but for a no-argument `-@` call.
    fn uminus_rewrite_bop(&self, self_val: InsnId) -> Option<(u32, u32)> {
        if !self.is_a(self_val, types::StringExact) {
            return None;
        }
        self.can_rewrite_if_frozen(self_val, STRING_REDEFINED_OP_FLAG, BOP_UMINUS)
            .then_some((STRING_REDEFINED_OP_FLAG, BOP_UMINUS))
    }

    pub fn load_rbasic_flags(&mut self, block: BlockId, recv: InsnId) -> InsnId {
        // Technically this also includes the shape (_shape_id) because the (shape, flags) tuple is
        // a (u32, u32) inside a u64 at RUBY_OFFSET_RBASIC_FLAGS (offset 0). It's fine to load the
        // shape alongside the flags, but make sure not to *store* the shape accidentally by
        // writing a u64.
        self.load_field(block, recv, FieldName::RBASIC_FLAGS, RUBY_OFFSET_RBASIC_FLAGS, types::CUInt64)
    }

    fn load_ep_flags(&mut self, block: BlockId, ep: InsnId) -> InsnId {
        self.load_ep_env_field(block, ep, FieldName::VM_ENV_DATA_INDEX_FLAGS, VM_ENV_DATA_INDEX_FLAGS as i32, types::CUInt64)
    }

    fn load_ep_env_field(&mut self, block: BlockId, ep: InsnId, id: FieldName, index: i32, return_type: Type) -> InsnId {
        self.load_field(block, ep, id, SIZEOF_VALUE_I32 * index, return_type)
    }

    pub fn guard_not_frozen(&mut self, block: BlockId, recv: InsnId, state: InsnId) {
        let flags = self.load_rbasic_flags(block, recv);
        self.push_insn(block, Insn::GuardNoBitsSet { val: flags, mask: Const::CUInt64(RUBY_FL_FREEZE as u64), mask_name: Some(ID!(RUBY_FL_FREEZE)), reason: Box::new(SideExitReason::GuardNotFrozen), state });
    }

    pub fn guard_not_shared(&mut self, block: BlockId, recv: InsnId, state: InsnId) {
        let flags = self.load_rbasic_flags(block, recv);
        self.push_insn(block, Insn::GuardNoBitsSet { val: flags, mask: Const::CUInt64(RUBY_ELTS_SHARED as u64), mask_name: Some(ID!(RUBY_ELTS_SHARED)), reason: Box::new(SideExitReason::GuardNotShared), state });
    }

    /// Guard that the string `recv` can be written to in place: it is modifiable (not frozen,
    /// tmp-locked, or chilled) and owns its buffer (not shared or nofree). See
    /// STR_DEPENDANT_MASK in string.c.
    pub fn guard_string_not_dependant(&mut self, block: BlockId, recv: InsnId, state: InsnId) {
        let flags = self.load_rbasic_flags(block, recv);
        self.push_insn(block, Insn::GuardNoBitsSet { val: flags, mask: Const::CUInt64(RSTRING_DEPENDANT_MASK as u64), mask_name: Some(ID!(RSTRING_DEPENDANT_MASK)), reason: Box::new(SideExitReason::GuardNotDependant), state });
    }

    /// `iseq` is the ISEQ that `ep_offset` is relative to, which is the ISEQ that
    /// produced the bytecode being translated. When `add_iseq_to_hir` runs for the
    /// top-level compile that is `self.iseq`, but when it runs as part of inlining
    /// it is the callee being inlined, not `self.iseq` (which is the outer caller).
    /// Decoding the offset against the wrong ISEQ trips `ep_offset_to_local_idx`'s
    /// `local_idx < local_table_size` assertion whenever the two ISEQs disagree on
    /// `local_table_size`, so callers thread the active ISEQ through explicitly.
    fn get_local_from_ep(
        &mut self,
        block: BlockId,
        iseq: IseqPtr,
        ep: InsnId,
        ep_offset: u32,
        level: u32,
        return_type: Type,
    ) -> InsnId {
        let local_id = get_local_var_id(iseq, level, ep_offset);
        let ep_offset = i32::try_from(ep_offset)
            .unwrap_or_else(|_| panic!("Could not convert ep_offset {ep_offset} to i32"));
        let offset = -(SIZEOF_VALUE_I32 * ep_offset);

        self.load_field(block, ep, local_id.into(), offset, return_type)
    }

    /// See `get_local_from_ep` for why `iseq` is threaded through explicitly rather
    /// than read from `self.iseq`.
    fn get_local_from_sp(
        &mut self,
        block: BlockId,
        iseq: IseqPtr,
        sp: InsnId,
        ep_offset: u32,
        return_type: Type,
    ) -> InsnId {
        let local_id = get_local_var_id(iseq, 0, ep_offset);
        let ep_offset = i32::try_from(ep_offset)
            .unwrap_or_else(|_| panic!("Could not convert ep_offset {ep_offset} to i32"));
        let offset = -(SIZEOF_VALUE_I32 * (ep_offset + 1));

        self.load_field(block, sp, local_id.into(), offset, return_type)
    }

    fn try_inline_invoke_builtin(&mut self, block: BlockId, insn: Insn) -> InsnId {
        let Insn::InvokeBuiltin { bf, recv, ref args, state, .. } = insn else {
            panic!("try_inline_invoke_builtin called with non-InvokeBuiltin instruction");
        };
        let args = args.clone();
        if let Some(replacement) = self.try_inline_builtin_body(block, bf, recv, &args, state) {
            return replacement;
        }
        return self.push_insn(block, insn);
    }

    /// Try replacing an InvokeBuiltin call with the inline HIR provided by the
    /// builtin's annotation (if any), appending the replacement instructions to
    /// `block`. Returns the instruction to use as the result, or None if the
    /// builtin couldn't be inlined.
    fn try_inline_builtin_body(&mut self, block: BlockId, bf: *const rb_builtin_function, recv: InsnId, args: &[InsnId], state: InsnId) -> Option<InsnId> {
        let props = ZJITState::get_method_annotations().get_builtin_properties(bf).unwrap_or_default();
        // Try inlining the cfunc into HIR
        let tmp_block = self.new_block(u32::MAX);
        if let Some(replacement) = (props.inline)(self, tmp_block, recv, args, state) {
            // Copy contents of tmp_block to block
            assert_ne!(block, tmp_block);
            let insns = std::mem::take(&mut self.blocks[tmp_block.to_usize()].insns);
            self.blocks[block.to_usize()].insns.extend(insns);
            self.count(block, Counter::inline_cfunc_optimized_send_count);
            self.infer_inlined_type(replacement);
            self.remove_block(tmp_block);
            return Some(replacement);
        }
        None
    }

    /// Rewrite eligible Send opcodes into SendDirect
    /// opcodes if we know the target ISEQ statically. This removes run-time method lookups and
    /// opens the door for inlining.
    /// Also try and inline constant caches, specialize object allocations, and more.
    fn type_specialize(&mut self) {
        for entry_block in self.reverse_post_order() {
            let old_insns = std::mem::take(&mut self.blocks[entry_block.to_usize()].insns);
            assert!(self.blocks[entry_block.to_usize()].insns.is_empty());
            // Rewriting an instruction into a branch splits the block: everything after it,
            // including the original terminator, is emitted into the join block instead.
            let mut block = entry_block;
            for insn_id in old_insns {
                let resolved = self.resolve(insn_id);
                match resolved.insn(self) {
                    // Elide `freeze`/`-@` on an object that is already known to be frozen. If the
                    // receiver is not a known frozen object, fall through to the generic Send
                    // specialization below instead of giving up on the call site: `String#-@`,
                    // `Integer#-@`, `Object#freeze` etc. are ordinary methods that can be
                    // specialized like any other.
                    &Insn::Send { recv, block: None, ref args, state, cd, .. }
                        if ruby_call_method_id(cd) == ID!(freeze) && args.is_empty()
                            && self.freeze_rewrite_bop(recv).is_some() => {
                        let (klass, bop) = self.freeze_rewrite_bop(recv).unwrap();
                        self.rewrite_if_frozen(block, insn_id, recv, klass, bop, state);
                    }
                    &Insn::Send { recv, block: None, ref args, state, cd, .. }
                        if ruby_call_method_id(cd) == ID!(minusat) && args.is_empty()
                            && self.uminus_rewrite_bop(recv).is_some() => {
                        let (klass, bop) = self.uminus_rewrite_bop(recv).unwrap();
                        self.rewrite_if_frozen(block, insn_id, recv, klass, bop, state);
                    }
                    &Insn::Send { mut recv, cd, state, block: send_block, .. } => {
                        let mut has_block = send_block.is_some();
                        // A send behind an ancestor guard resolves its target from the class the
                        // guard checked instead of from the receiver profile: the guard plus the
                        // NoMethodOverride invariant emitted below make that lookup the one every
                        // receiver reaching this arm performs.
                        let ancestor_class = self.ancestor_dispatch_class(insn_id);
                        let (klass, profiled_type) = if let Some(ancestor_class) = ancestor_class {
                            (ancestor_class, None)
                        } else { match self.resolve_receiver_type(recv, self.type_of(recv), state) {
                            ReceiverTypeResolution::StaticallyKnown { class } => (class, None),
                            ReceiverTypeResolution::Monomorphic { profiled_type }
                            | ReceiverTypeResolution::SkewedPolymorphic { profiled_type } => (profiled_type.class(), Some(profiled_type)),
                            ReceiverTypeResolution::SkewedMegamorphic { .. }
                            | ReceiverTypeResolution::Megamorphic => {
                                self.set_dynamic_send_reason(insn_id, SendMegamorphic);
                                self.push_insn_id(block, insn_id);
                                continue;
                            }
                            ReceiverTypeResolution::Polymorphic => {
                                self.set_dynamic_send_reason(insn_id, SendPolymorphic);
                                self.push_insn_id(block, insn_id);
                                continue;
                            }
                            ReceiverTypeResolution::NoProfile => {
                                self.set_dynamic_send_reason(insn_id, SendNoProfiles);
                                self.push_insn_id(block, insn_id);
                                continue;
                            }
                        } };
                        let ci = unsafe { (*cd).ci }; // info about the call site

                        let mut flags = unsafe { rb_vm_ci_flag(ci) };
                        let mut mid = unsafe { vm_ci_mid(ci) };

                        // A `send`/`__send__` call site that HIR build guarded to one method name
                        // is compiled as a call to that method: the name argument is dropped and,
                        // like the interpreter's vm_call_opt_send, private and protected methods
                        // are callable. Only plain positional call sites get an override (see
                        // send_method_names), so every other property of `ci` — the keyword
                        // table, the splat and block-arg flags — is unchanged by the rewrite and
                        // stays valid for the resolved call.
                        let mut send_mid_override = self.send_mid_overrides.get(&insn_id).copied();
                        if let Some(target_mid) = send_mid_override {
                            // Only rewrite while `send` really is BasicObject#send. If it has
                            // been replaced, call it like any other method.
                            let send_cme = unsafe { rb_callable_method_entry(klass, mid) };
                            let is_opt_send = !send_cme.is_null()
                                && unsafe { get_cme_def_type(send_cme) } == VM_METHOD_TYPE_OPTIMIZED
                                && unsafe { get_cme_def_body_optimized_type(send_cme) } == OPTIMIZED_METHOD_TYPE_SEND;
                            if is_opt_send {
                                self.push_insn(block, Insn::PatchPoint { invariant: Invariant::MethodRedefined { klass, method: mid, cme: send_cme }, state });
                                mid = target_mid;
                                flags |= VM_CALL_FCALL;
                            } else {
                                send_mid_override = None;
                            }
                        }
                        let flags = flags;
                        let mid = mid;

                        // Do method lookup
                        let mut cme = unsafe { rb_callable_method_entry(klass, mid) };
                        if cme.is_null() {
                            self.set_dynamic_send_reason(insn_id, SendNotOptimizedMethodType(MethodType::Null));
                            self.push_insn_id(block, insn_id); continue;
                        }
                        // Load an overloaded cme if applicable. See vm_search_cc().
                        // It allows you to use a faster ISEQ if possible.
                        cme = unsafe { rb_check_overloaded_cme(cme, ci) };
                        let visibility = unsafe { METHOD_ENTRY_VISI(cme) };
                        match (visibility, flags & VM_CALL_FCALL != 0) {
                            (METHOD_VISI_PUBLIC, _) => {}
                            (METHOD_VISI_PRIVATE, true) => {}
                            (METHOD_VISI_PROTECTED, true) => {}
                            _ => {
                                self.set_dynamic_send_reason(insn_id, SendNotOptimizedNeedPermission);
                                self.push_insn_id(block, insn_id); continue;
                            }
                        }
                        let mut def_type = unsafe { get_cme_def_type(cme) };
                        while def_type == VM_METHOD_TYPE_ALIAS {
                            cme = unsafe { rb_aliased_callable_method_entry(cme) };
                            def_type = unsafe { get_cme_def_type(cme) };
                        }

                        // Check if we can optimize `foo(&block)` where block is nil to a send without block.
                        // `state` keeps referring to the pre-send frame state (block arg still on the
                        // stack). Any guard that side-exits before the call re-executes the `send` in
                        // the interpreter, so it must reconstruct the stack with the block arg present.
                        // Only the direct-send frame setup uses `send_frame_state`, which has the nil
                        // block arg stripped from the stack.
                        let mut send_block = send_block;
                        let mut send_frame_state = state;
                        let mut args = match resolved.insn(self) {
                            Insn::Send { args, .. } => args.to_vec(),
                            _ => panic!("Expected Send instruction"),
                        };
                        if send_mid_override.is_some() {
                            // Drop the method-name argument, as vm_call_opt_send does. The
                            // pre-send `state` is kept for guards so that a side exit re-runs
                            // `send` in the interpreter with the name still on the stack;
                            // `send_frame_state` describes the callee frame without it.
                            args.remove(0);
                            let new_state = self.frame_state(state).with_replaced_args(&args, args.len() + 1);
                            send_frame_state = self.push_insn(block, Insn::Snapshot { state: Box::new(new_state) });
                        }
                        let mut stripped_block_arg = false;
                        let mut send_block_arg = None;
                        // A C method's frame carries the block handler in its specval just like an
                        // ISEQ frame's, so the same reduction applies; the difference is that the
                        // block argument keeps its VM stack slot, which the C frame setup accounts
                        // for. Nothing else reads `args` positionally for a C call.
                        if send_block == Some(BlockHandler::BlockArg)
                            && matches!(def_type, VM_METHOD_TYPE_ISEQ | VM_METHOD_TYPE_CFUNC) {
                            // The block arg is the last element in args
                            if let Some(&block_arg) = args.last() {
                                let statically_nil = self.is_a(block_arg, types::NilClass);
                                let profiled_nil = self.profiled_type_of_at(block_arg, state)
                                    .map_or(false, |pt| pt.is_nil());
                                let is_block_param_proxy = !statically_nil
                                    && self.type_of(block_arg).ruby_object() == Some(unsafe { rb_block_param_proxy });
                                let proc_type = if statically_nil || profiled_nil || is_block_param_proxy { None } else { self.proc_block_arg_type(block_arg, state) };
                                if statically_nil || profiled_nil {
                                    if !statically_nil {
                                        // Guard needed when relying on profiled type. Uses the original
                                        // `state` so a side exit re-executes the send with the block
                                        // arg still on the VM stack.
                                        //
                                        // Recompile on exit so a site that starts seeing non-nil
                                        // blocks re-profiles the block arg and drops this speculation
                                        // (falling back to a dynamic send) instead of paying the guard
                                        // side exit repeatedly. This matches the receiver GuardType
                                        // below and the getblockparamproxy BlockParamProxyNotNil guard.
                                        self.push_insn(block, Insn::GuardBitEquals {
                                            val: block_arg,
                                            expected: Const::Value(Qnil),
                                            reason: Box::new(SideExitReason::BlockArgNotNil),
                                            state,
                                            recompile: Some(Recompile),
                                        });
                                    }
                                    // Strip nil block arg and treat as no block
                                    args = args[..args.len() - 1].to_vec();
                                    send_block = None;
                                    has_block = false;
                                    stripped_block_arg = true;
                                    // Frame state for the direct send only: the block arg is removed
                                    // from the stack so the callee frame is laid out correctly.
                                    let new_state = self.frame_state(state).with_replaced_args(&args, args.len() + 1);
                                    send_frame_state = self.push_insn(block, Insn::Snapshot { state: Box::new(new_state) });
                                } else if is_block_param_proxy {
                                    // `vm_caller_setup_arg_block` answers the block param proxy with
                                    // `VM_CF_BLOCK_HANDLER(cfp)`, this frame's own block handler.
                                    // Load it out of the local EP and install it as the callee's,
                                    // which is the whole of what the interpreter would have done for
                                    // a `def foo(&blk) = bar(&blk)` forwarding site.
                                    let lep_level = get_lvar_level(self.frame_state(state).iseq);
                                    let lep = self.push_insn(block, Insn::GetEP { level: lep_level });
                                    let block_handler = self.load_ep_env_field(block, lep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, VM_ENV_DATA_INDEX_SPECVAL, types::BasicObject);
                                    send_block_arg = Some(block_handler);
                                    args = args[..args.len() - 1].to_vec();
                                    send_block = None;
                                    stripped_block_arg = true;
                                    let new_state = self.frame_state(state).with_replaced_args(&args, args.len() + 1);
                                    send_frame_state = self.push_insn(block, Insn::Snapshot { state: Box::new(new_state) });
                                } else if let Some(proc_type) = proc_type {
                                    // `foo(&blk)` with a Proc in `blk`: `vm_caller_setup_arg_block`
                                    // hands `vm_to_proc(blk)`, i.e. the Proc itself, to the callee as
                                    // its block handler. Guard the Proc so the frame setup can write
                                    // that handler into the callee's specval directly instead of
                                    // paying for a dynamic send. The guard uses the original `state`,
                                    // so a side exit re-runs the send with the block arg still on the
                                    // VM stack, and recompiles so a site that starts seeing other
                                    // block arguments re-profiles rather than exiting every call.
                                    let guarded = if self.is_a(block_arg, proc_type) {
                                        block_arg
                                    } else {
                                        let guarded = self.push_insn(block, Insn::GuardType {
                                            val: block_arg,
                                            guard_type: proc_type,
                                            state,
                                            recompile: Some(Recompile),
                                        });
                                        self.insn_types[guarded.to_usize()] = self.infer_type(guarded);
                                        guarded
                                    };
                                    send_block_arg = Some(guarded);
                                    // The block handler is not one of the callee's parameters, so it
                                    // comes off the argument list the same way a nil one does.
                                    args = args[..args.len() - 1].to_vec();
                                    send_block = None;
                                    stripped_block_arg = true;
                                    let new_state = self.frame_state(state).with_replaced_args(&args, args.len() + 1);
                                    send_frame_state = self.push_insn(block, Insn::Snapshot { state: Box::new(new_state) });
                                } else {
                                    // Can't prove block arg is nil
                                    self.set_dynamic_send_reason(insn_id, SendBlockArgNotNil);
                                    self.push_insn_id(block, insn_id); continue;
                                }
                            }
                        }

                        // If the call site info indicates that the `Function` has overly complex arguments, then do not optimize into a `SendDirect`.
                        // Optimized methods(`VM_METHOD_TYPE_OPTIMIZED`) and C methods handle their own argument constraints (e.g., kw_splat for Proc call).
                        // Mask out ARGS_BLOCKARG only if we've already handled the nil block arg case above.
                        let mut flags_for_check = if stripped_block_arg { flags & !VM_CALL_ARGS_BLOCKARG } else { flags };

                        // `foo(*args)`: if the splat array has always had the same length here, guard
                        // that length and read the elements out so the call can be a direct send.
                        // Only ISEQ callees benefit: the other method types read the argument count
                        // from the call info rather than from the HIR argument list.
                        const SPLAT_EXPANSION_BLOCKERS: u32 = VM_CALL_KW_SPLAT | VM_CALL_KWARG | VM_CALL_ARGS_BLOCKARG
                            | VM_CALL_FORWARDING | VM_CALL_TAILCALL | VM_CALL_OPT_SEND | VM_CALL_SUPER | VM_CALL_ZSUPER;
                        // Set when the splat was handed straight to the callee's rest parameter,
                        // which leaves `args` already in SendDirect shape.
                        let mut rest_prepacked = false;
                        if flags_for_check & VM_CALL_ARGS_SPLAT != 0
                            && flags_for_check & SPLAT_EXPANSION_BLOCKERS == 0
                            && matches!(def_type, VM_METHOD_TYPE_ISEQ | VM_METHOD_TYPE_BMETHOD)
                        {
                            // Expanding is the better shape when the length is predictable: the
                            // callee's parameters get the elements in registers with no Array to
                            // allocate. Forwarding is the fallback for a varying length.
                            let rewritten = self.try_expand_splat_args(block, &args, state)
                                .or_else(|| {
                                    // Only the ISEQ path below consumes prepacked arguments; a
                                    // bmethod dispatches through the proc's own parameter setup.
                                    if def_type != VM_METHOD_TYPE_ISEQ { return None; }
                                    let callee_iseq = unsafe { get_def_iseq_ptr((*cme).def) };
                                    let forwarded = self.try_forward_splat_to_rest(block, &args, callee_iseq, state);
                                    rest_prepacked = forwarded.is_some();
                                    forwarded
                                });
                            if let Some(rewritten) = rewritten {
                                let new_state = self.frame_state(send_frame_state).with_replaced_args(&rewritten, args.len());
                                send_frame_state = self.push_insn(block, Insn::Snapshot { state: Box::new(new_state) });
                                args = rewritten;
                                flags_for_check &= !VM_CALL_ARGS_SPLAT;
                            }
                        }
                        if def_type != VM_METHOD_TYPE_OPTIMIZED && def_type != VM_METHOD_TYPE_CFUNC && unspecializable_call_type(flags_for_check) {
                            self.count_complex_call_features(block, flags, state);
                            self.set_dynamic_send_reason(insn_id, ComplexArgPass);
                            self.push_insn_id(block, insn_id); continue;
                        }

                        if def_type == VM_METHOD_TYPE_ISEQ {
                            // TODO(max): Allow non-iseq; cache cme
                            // Only specialize positional-positional calls
                            // TODO(max): Handle other kinds of parameter passing
                            let iseq = unsafe { get_def_iseq_ptr((*cme).def) };
                            let Ok(call) = self.build_send_direct_args(&args, ci, iseq, has_block, send_block_arg.is_some(), rest_prepacked)
                                .inspect_err(|failure| failure.record(self, block, insn_id, SendDirectFallbackContext::Send)) else {
                                self.push_insn_id(block, insn_id); continue;
                            };

                            // Check singleton class assumption first, before emitting other patchpoints
                            if !self.assume_no_singleton_classes_for_send(block, klass, state, ancestor_class) {
                                self.set_dynamic_send_reason(insn_id, SingletonClassSeen);
                                self.push_insn_id(block, insn_id); continue;
                            }

                            // Add PatchPoint for method redefinition
                            self.assume_cme_for_send(block, klass, mid, cme, state, ancestor_class);

                            // Add GuardType for profiled receiver
                            if let Some(profiled_type) = profiled_type {
                                recv = self.push_insn(block, Insn::GuardType { val: recv, guard_type: Type::from_profiled_type(profiled_type), state, recompile: Some(Recompile) });
                                self.insn_types[recv.to_usize()] = self.infer_type(recv);
                            }

                            let SendDirectArgs { state: send_state, args: send_args, kw_bits, jit_entry_idx } =
                                self.emit_send_direct_args(block, call, &args, send_frame_state);
                            let replacement = self.push_insn(block, Insn::SendDirect(Box::new(SendDirectData { recv, cd, cme, iseq, args: send_args, kw_bits, jit_entry_idx, state: send_state, guard_state: state, block: send_block, block_arg: send_block_arg })));
                            self.make_equal_to(insn_id, replacement);
                        } else if !has_block && def_type == VM_METHOD_TYPE_BMETHOD {
                            let procv = unsafe { rb_get_def_bmethod_proc((*cme).def) };
                            let proc = unsafe { rb_jit_get_proc_ptr(procv) };
                            let proc_block = unsafe { &(*proc).block };
                            // Target ISEQ bmethods. Can't handle for example, `define_method(:foo, &:foo)`
                            // which makes a `block_type_symbol` bmethod.
                            if proc_block.type_ != block_type_iseq {
                                self.set_dynamic_send_reason(insn_id, BmethodNonIseqProc);
                                self.push_insn_id(block, insn_id); continue;
                            }
                            let capture = unsafe { proc_block.as_.captured.as_ref() };
                            let iseq = unsafe { *capture.code.iseq.as_ref() };

                            let Ok(call) = self.build_send_direct_args(&args, ci, iseq, has_block, send_block_arg.is_some(), rest_prepacked)
                                .inspect_err(|failure| failure.record(self, block, insn_id, SendDirectFallbackContext::Send)) else {
                                self.push_insn_id(block, insn_id); continue;
                            };

                            // Patch points:
                            // Check for "defined with an un-shareable Proc in a different Ractor"
                            if !procv.shareable_p() && !self.assume_single_ractor_mode(block, state) {
                                // TODO(alan): Turn this into a ractor belonging guard to work better in multi ractor mode.
                                self.set_dynamic_send_reason(insn_id, SingleRactorModeRequired);
                                self.push_insn_id(block, insn_id); continue;
                            }
                            // Check singleton class assumption first, before emitting other patchpoints
                            if !self.assume_no_singleton_classes_for_send(block, klass, state, ancestor_class) {
                                self.set_dynamic_send_reason(insn_id, SingletonClassSeen);
                                self.push_insn_id(block, insn_id); continue;
                            }
                            self.assume_cme_for_send(block, klass, mid, cme, state, ancestor_class);

                            if let Some(profiled_type) = profiled_type {
                                recv = self.guard_profiled_type(block, recv, profiled_type, state);
                            }

                            let SendDirectArgs { state: send_state, args: send_args, kw_bits, jit_entry_idx } =
                                self.emit_send_direct_args(block, call, &args, send_frame_state);
                            let replacement = self.push_insn(block, Insn::SendDirect(Box::new(SendDirectData { recv, cd, cme, iseq, args: send_args, kw_bits, jit_entry_idx, state: send_state, guard_state: state, block: None, block_arg: None })));
                            self.make_equal_to(insn_id, replacement);
                        } else if !has_block && def_type == VM_METHOD_TYPE_IVAR && args.is_empty() {
                            // Check if we're accessing ivars of a Class or Module object as they require single-ractor mode.
                            // We omit gen_prepare_non_leaf_call on gen_getivar, so it's unsafe to raise for multi-ractor mode.
                            if klass.is_metaclass() && !self.assume_single_ractor_mode(block, state) {
                                self.set_dynamic_send_reason(insn_id, SingleRactorModeRequired);
                                self.push_insn_id(block, insn_id); continue;
                            }
                            // Check singleton class assumption first, before emitting other patchpoints
                            if !self.assume_no_singleton_classes_for_send(block, klass, state, ancestor_class) {
                                self.set_dynamic_send_reason(insn_id, SingletonClassSeen);
                                self.push_insn_id(block, insn_id); continue;
                            }

                            self.assume_cme_for_send(block, klass, mid, cme, state, ancestor_class);

                            let id = unsafe { get_cme_def_body_attr_id(cme) };
                            if let Some(profiled_type) = profiled_type {
                                recv = self.guard_profiled_type(block, recv, profiled_type, state);

                                // A polymorphic-arm profile is not a prediction for this program
                                // point (see the attr_writer case below), so branch on its shape
                                // rather than guarding: a receiver of the right class but a
                                // different shape takes rb_ivar_get instead of side-exiting and
                                // recompiling the ISEQ around a shape the arm never promised.
                                let shape_miss = if profiled_type.flags().is_polymorphic_arm() {
                                    ShapeMiss::CallFallbackWithoutReprofile
                                } else {
                                    ShapeMiss::SideExit
                                };
                                let branch_on_shape = shape_miss.calls_fallback()
                                    && !profiled_type.shape().is_complex()
                                    && !profiled_type.flags().is_immediate();
                                let replacement = if branch_on_shape {
                                    let insn_idx = self.frame_state_insn_idx(state) as u32;
                                    let shapes = self.profiled_shape_variants(recv, state, profiled_type);
                                    let (join_block, result) = self.dispatch_getivar(&shapes, /* covers_profile */ false, block, insn_idx, recv, id, std::ptr::null(), state, shape_miss)
                                        .expect("dispatch_getivar with a profiled shape never side-exits unconditionally");
                                    block = join_block;
                                    result
                                } else {
                                    match self.try_emit_optimized_getivar(block, recv, id, profiled_type, state) {
                                        Ok(replacement) => replacement,
                                        // The final version of an ISEQ may not speculate with a
                                        // guard that side-exits, but it can still branch on the
                                        // shape the way getinstancevariable does, so the common
                                        // shape reads inline instead of calling rb_ivar_get every
                                        // time.
                                        Err(Counter::getivar_fallback_no_side_exits) => {
                                            let insn_idx = self.frame_state_insn_idx(state) as u32;
                                            let shapes = self.profiled_shape_variants(recv, state, profiled_type);
                                            let (join_block, result) = self.dispatch_getivar(&shapes, /* covers_profile */ false, block, insn_idx, recv, id, std::ptr::null(), state, shape_miss)
                                                .expect("dispatch_getivar only side-exits without a profiled shape");
                                            block = join_block;
                                            result
                                        }
                                        Err(counter) => {
                                            self.count(block, counter);
                                            self.push_insn(block, Insn::GetIvar { self_val: recv, id, ic: std::ptr::null(), state })
                                        }
                                    }
                                };
                                self.make_equal_to(insn_id, replacement);
                            } else {
                                // No shape information, just static class information
                                let resolution = self.resolve_receiver_type_from_profile(recv, state);
                                let counter = Self::getivar_fallback_reason(resolution, std::ptr::null());
                                self.count(block, counter);
                                let getivar = self.push_insn(block, Insn::GetIvar { self_val: recv, id, ic: std::ptr::null(), state });
                                self.make_equal_to(insn_id, getivar);
                            }
                        } else if let (false, VM_METHOD_TYPE_ATTRSET, &[val]) = (has_block, def_type, args.as_slice()) {
                            // Check if we're accessing ivars of a Class or Module object as they require single-ractor mode.
                            // We omit gen_prepare_non_leaf_call on gen_getivar, so it's unsafe to raise for multi-ractor mode.
                            if klass.is_metaclass() && !self.assume_single_ractor_mode(block, state) {
                                self.set_dynamic_send_reason(insn_id, SingleRactorModeRequired);
                                self.push_insn_id(block, insn_id); continue;
                            }

                            self.assume_cme_for_send(block, klass, mid, cme, state, ancestor_class);
                            let id = unsafe { get_cme_def_body_attr_id(cme) };
                            if let Some(profiled_type) = profiled_type {
                                recv = self.guard_profiled_type(block, recv, profiled_type, state);
                                // A polymorphic-arm profile is not a prediction for this program
                                // point: the arm's type test only pins the class, and the shape is
                                // whatever the profiler happened to see for that class at the
                                // unrefined call site. Guarding it with a side exit measurably
                                // pushes the call site megamorphic. Branch on the shape instead and
                                // let rb_ivar_set handle every other shape, which also gives the
                                // final version of an ISEQ a fast path it is otherwise denied.
                                let shape_miss = if profiled_type.flags().is_polymorphic_arm() {
                                    ShapeMiss::CallFallbackWithoutReprofile
                                } else {
                                    ShapeMiss::SideExit
                                };
                                match self.prepare_optimized_setivar(id, profiled_type) {
                                    Ok(spec) if shape_miss.calls_fallback() || self.policy.no_side_exits => {
                                        let insn_idx = self.frame_state_insn_idx(state) as u32;
                                        block = self.dispatch_setivar(&[spec], None, /* covers_profile */ false, block, insn_idx, recv, id, std::ptr::null(), val, state, shape_miss)
                                            .expect("dispatch_setivar with a spec never side-exits unconditionally");
                                    }
                                    Ok(spec) => {
                                        // TODO: attr_writer SetIvar has a null inline cache and may target a receiver
                                        // operand other than CFP self. Support it with a reprofile strategy that
                                        // profiles the receiver operand even after the send insn has finished profiling.
                                        let recv = self.guard_heap(block, recv, state);
                                        let shape = self.load_shape(block, recv);
                                        self.guard_shape(block, shape, profiled_type.shape(), state, None);
                                        self.emit_optimized_setivar(block, recv, id, val, spec);
                                    }
                                    Err(counter) => {
                                        self.count(block, counter);
                                        self.push_insn(block, Insn::SetIvar { self_val: recv, id, ic: std::ptr::null(), val, state });
                                    }
                                }
                            } else {
                                // No shape information, just static class information
                                self.push_insn(block, Insn::SetIvar { self_val: recv, id, ic: std::ptr::null(), val, state });
                            }
                            self.make_equal_to(insn_id, val);
                        } else if !has_block && def_type == VM_METHOD_TYPE_OPTIMIZED {
                            let opt_type: OptimizedMethodType = unsafe { get_cme_def_body_optimized_type(cme) }.into();
                            match (opt_type, args.as_slice()) {
                                (OptimizedMethodType::Call, _) => {
                                    if flags & (VM_CALL_ARGS_SPLAT | VM_CALL_KWARG) != 0 {
                                        self.count_complex_call_features(block, flags, state);
                                        self.set_dynamic_send_reason(insn_id, ComplexArgPass);
                                        self.push_insn_id(block, insn_id); continue;
                                    }
                                    // Check singleton class assumption first, before emitting other patchpoints
                                    if !self.assume_no_singleton_classes_for_send(block, klass, state, ancestor_class) {
                                        self.set_dynamic_send_reason(insn_id, SingletonClassSeen);
                                        self.push_insn_id(block, insn_id); continue;
                                    }
                                    self.assume_cme_for_send(block, klass, mid, cme, state, ancestor_class);
                                    if let Some(profiled_type) = profiled_type {
                                        recv = self.guard_profiled_type(block, recv, profiled_type, state);
                                    }
                                    let kw_splat = flags & VM_CALL_KW_SPLAT != 0;
                                    let invoke_proc = self.push_insn(block, Insn::InvokeProc { recv, args: args.clone(), state, kw_splat });
                                    self.make_equal_to(insn_id, invoke_proc);
                                }
                                (OptimizedMethodType::StructAref, &[]) | (OptimizedMethodType::StructAset, &[_]) => {
                                    if unspecializable_call_type(flags) {
                                        self.count_complex_call_features(block, flags, state);
                                        self.set_dynamic_send_reason(insn_id, ComplexArgPass);
                                        self.push_insn_id(block, insn_id); continue;
                                    }
                                    let index: i32 = unsafe { get_cme_def_body_optimized_index(cme) }
                                                    .try_into()
                                                    .unwrap();
                                    // We are going to use an encoding that takes a 4-byte immediate which
                                    // limits the offset to INT32_MAX.
                                    {
                                        let native_index = (index as i64) * (SIZEOF_VALUE as i64);
                                        if native_index > (i32::MAX as i64) {
                                            self.set_dynamic_send_reason(insn_id, OperandTooLarge);
                                            self.push_insn_id(block, insn_id); continue;
                                        }
                                    }
                                    // Get the profiled type to check if the fields is embedded or heap allocated.
                                    let Some(is_embedded) = self.profiled_type_of_at(recv, state).map(|t| t.flags().is_struct_embedded()) else {
                                        // No (monomorphic/skewed polymorphic) profile info
                                        self.set_dynamic_send_reason(insn_id, SendNoProfiles);
                                        self.push_insn_id(block, insn_id); continue;
                                    };
                                    // Check singleton class assumption first, before emitting other patchpoints
                                    if !self.assume_no_singleton_classes_for_send(block, klass, state, ancestor_class) {
                                        self.set_dynamic_send_reason(insn_id, SingletonClassSeen);
                                        self.push_insn_id(block, insn_id); continue;
                                    }
                                    self.assume_cme_for_send(block, klass, mid, cme, state, ancestor_class);
                                    if let Some(profiled_type) = profiled_type {
                                        recv = self.guard_profiled_type(block, recv, profiled_type, state);
                                    }
                                    // All structs from the same Struct class should have the same
                                    // length. So if our recv is embedded all runtime
                                    // structs of the same class should be as well, and the same is
                                    // true of the converse.
                                    //
                                    // No need for a GuardShape.
                                    if let OptimizedMethodType::StructAset = opt_type {
                                        self.guard_not_frozen(block, recv, state);
                                    }

                                    let (target, offset) = if is_embedded {
                                        let offset = RUBY_OFFSET_RSTRUCT_AS_ARY + (SIZEOF_VALUE_I32 * index);
                                        (recv, offset)
                                    } else {
                                        let as_heap = self.load_field(block, recv, FieldName::as_heap, RUBY_OFFSET_RSTRUCT_AS_HEAP_PTR, types::CPtr);
                                        let offset = SIZEOF_VALUE_I32 * index;
                                        (as_heap, offset)
                                    };

                                    let replacement = if let (OptimizedMethodType::StructAset, &[val]) = (opt_type, args.as_slice()) {
                                        self.push_insn(block, Insn::StoreField { recv: target, id: mid.into(), offset, val, num_bits: types::BasicObject.num_bits() });
                                        self.push_insn(block, Insn::WriteBarrier { recv, val });
                                        val
                                    } else { // StructAref
                                        self.load_field(block, target, mid.into(), offset, types::BasicObject)
                                    };
                                    self.make_equal_to(insn_id, replacement);
                                },
                                _ => {
                                    self.set_dynamic_send_reason(insn_id, SendNotOptimizedMethodTypeOptimized(OptimizedMethodType::from(opt_type)));
                                    self.push_insn_id(block, insn_id); continue;
                                },
                            };
                        } else if def_type == VM_METHOD_TYPE_CFUNC && !unsafe { rb_zjit_method_tracing_currently_enabled() } {
                            // Try to reduce a Send insn to a CCallWithFrame
                            fn reduce_send_to_ccall(
                                fun: &mut Function,
                                block: BlockId,
                                send_insn_id: InsnId,
                                mut recv: InsnId,
                                cd: *const rb_call_data,
                                send_block: Option<BlockHandler>,
                                args: Vec<InsnId>,
                                state: InsnId,
                                recv_class: VALUE,
                                profiled_type: Option<ProfiledType>,
                                cme: *const rb_callable_method_entry_struct,
                                method_id: ID,
                                argc: u32,
                                // The call site's flags with `VM_CALL_ARGS_BLOCKARG` cleared when
                                // `block_arg` already holds the handler the interpreter would have
                                // built from it.
                                ci_flags: u32,
                                block_arg: Option<InsnId>,
                            ) -> Result<(), ()> {
                                // Argument shapes the C frame setup cannot reproduce.
                                if unspecializable_c_call_type(ci_flags) {
                                    // Only count features NOT already counted in type_specialize.
                                    if !unspecializable_call_type(ci_flags) {
                                        fun.count_complex_call_features(block, ci_flags, state);
                                    }
                                    fun.set_dynamic_send_reason(send_insn_id, ComplexArgPass);
                                    return Err(());
                                }

                                let blockiseq = match send_block {
                                    Some(BlockHandler::BlockArg) => unreachable!("unsupported &block should have been filtered out"),
                                    Some(BlockHandler::BlockIseq(blockiseq)) => Some(blockiseq),
                                    None => None,
                                };
                                // A block reaches the callee either way, so neither the inline
                                // bodies nor the leaf fast path (which push no frame to carry the
                                // handler) can serve this call.
                                let passes_block = blockiseq.is_some() || block_arg.is_some();

                                let cfunc = unsafe { get_cme_def_body_cfunc(cme) };
                                // Find the `argc` (arity) of the C method, which describes the parameters it expects
                                let cfunc_argc = unsafe { get_mct_argc(cfunc) };
                                let cfunc_ptr = unsafe { get_mct_func(cfunc) }.cast();
                                let name = unsafe { (*cme).called_id };

                                // Look up annotations
                                let props = ZJITState::get_method_annotations().get_cfunc_properties(cme);
                                if props.is_none() && get_option!(stats) {
                                    fun.count_not_annotated_cfunc(block, cme);
                                }
                                let props = props.unwrap_or_default();
                                let return_type = props.return_type;
                                // Don't consider cfuncs with block arguments as elidable for now
                                let elidable = !passes_block && props.elidable;

                                match cfunc_argc {
                                    0.. => {
                                        // (self, arg0, arg1, ..., argc) form
                                        //
                                        // Bail on argc mismatch
                                        if argc != cfunc_argc as u32 {
                                            fun.set_dynamic_send_reason(send_insn_id, ArgcParamMismatch);
                                            return Err(());
                                        }

                                        // TODO: Support passing arguments on the stack in C calls
                                        // +1 for self
                                        if (argc as usize)+1 > C_ARG_OPNDS.len() {
                                            fun.set_dynamic_send_reason(send_insn_id, TooManyArgsForLir);
                                            return Err(());
                                        }

                                        // Check singleton class assumption first, before emitting other patchpoints
                                        if !fun.assume_no_singleton_classes(block, recv_class, state) {
                                            fun.set_dynamic_send_reason(send_insn_id, SingletonClassSeen);
                                            return Err(());
                                        }

                                        // Commit to the replacement. Put PatchPoint.
                                        fun.gen_patch_points_for_optimized_ccall(block, recv_class, method_id, cme, state);

                                        if let Some(profiled_type) = profiled_type {
                                            // Guard receiver class
                                            recv = fun.guard_profiled_type(block, recv, profiled_type, state);
                                        }

                                        // Try inlining the cfunc into HIR. Only inline if we don't have a block argument
                                        if !passes_block {
                                            let tmp_block = fun.new_block(u32::MAX);
                                            if let Some(replacement) = (props.inline)(fun, tmp_block, recv, &args, state) {
                                                // Copy contents of tmp_block to block
                                                assert_ne!(block, tmp_block);
                                                let insns = std::mem::take(&mut fun.blocks[tmp_block.to_usize()].insns);
                                                fun.blocks[block.to_usize()].insns.extend(insns);
                                                fun.count(block, Counter::inline_cfunc_optimized_send_count);
                                                fun.make_equal_to(send_insn_id, replacement);
                                                fun.infer_inlined_type(replacement);
                                                fun.remove_block(tmp_block);
                                                return Ok(());
                                            }

                                            // Only allow leaf calls if we don't have a block argument
                                            if props.leaf && props.no_gc {
                                                fun.count(block, Counter::inline_cfunc_optimized_send_count);
                                                let owner = unsafe { (*cme).owner };
                                                let ccall = fun.push_insn(block, Insn::CCall { cfunc: cfunc_ptr, recv, args, name, owner, return_type, elidable });
                                                fun.insn_types[ccall.to_usize()] = fun.infer_type(ccall);
                                                fun.make_equal_to(send_insn_id, ccall);
                                                return Ok(());
                                            }
                                        }

                                        // Emit a call
                                        if get_option!(stats) {
                                            fun.count_not_inlined_cfunc(block, cme);
                                        }
                                        let ccall = fun.push_insn(block, Insn::CCallWithFrame(Box::new(CCallWithFrameData {
                                            cd,
                                            cfunc: cfunc_ptr,
                                            recv,
                                            args,
                                            cme,
                                            name,
                                            state,
                                            return_type,
                                            elidable,
                                            block: blockiseq.map(BlockHandler::BlockIseq),
                                            block_arg,
                                        })));
                                        fun.insn_types[ccall.to_usize()] = fun.infer_type(ccall);
                                        fun.make_equal_to(send_insn_id, ccall);
                                        Ok(())
                                    }
                                    // Variadic method
                                    -1 => {
                                        // The method gets a pointer to the first argument
                                        // func(int argc, VALUE *argv, VALUE recv)

                                        // Check singleton class assumption first, before emitting other patchpoints
                                        if !fun.assume_no_singleton_classes(block, recv_class, state) {
                                            fun.set_dynamic_send_reason(send_insn_id, SingletonClassSeen);
                                            return Err(());
                                        }

                                        fun.gen_patch_points_for_optimized_ccall(block, recv_class, method_id, cme, state);

                                        if let Some(profiled_type) = profiled_type {
                                            // Guard receiver class
                                            recv = fun.guard_profiled_type(block, recv, profiled_type, state);
                                        }

                                        // Try inlining the cfunc into HIR. Only inline if we don't have a block argument
                                        if !passes_block {
                                            let tmp_block = fun.new_block(u32::MAX);
                                            if let Some(replacement) = (props.inline)(fun, tmp_block, recv, &args, state) {
                                                // Copy contents of tmp_block to block
                                                assert_ne!(block, tmp_block);
                                                let insns = std::mem::take(&mut fun.blocks[tmp_block.to_usize()].insns);
                                                fun.blocks[block.to_usize()].insns.extend(insns);
                                                fun.count(block, Counter::inline_cfunc_optimized_send_count);
                                                fun.make_equal_to(send_insn_id, replacement);
                                                fun.infer_inlined_type(replacement);
                                                fun.remove_block(tmp_block);
                                                return Ok(());
                                            }

                                            // Only allow inline calls if they are leaf, don't allocate, and don't have a block argument
                                            if props.leaf && props.no_gc {
                                                fun.count(block, Counter::inline_cfunc_optimized_send_count);
                                                let owner = unsafe { (*cme).owner };
                                                let ccall = fun.push_insn(block, Insn::CCall { cfunc: cfunc_ptr, recv, args, name, owner, return_type, elidable });
                                                fun.insn_types[ccall.to_usize()] = fun.infer_type(ccall);
                                                fun.make_equal_to(send_insn_id, ccall);
                                                return Ok(());
                                            }
                                        }

                                        // No inlining; emit a call
                                        if get_option!(stats) {
                                            fun.count_not_inlined_cfunc(block, cme);
                                        }

                                        let ccall = fun.push_insn(block, Insn::CCallVariadic(Box::new(CCallVariadicData {
                                            cfunc: cfunc_ptr,
                                            recv,
                                            args,
                                            cme,
                                            name: method_id,
                                            state,
                                            return_type,
                                            elidable,
                                            block: blockiseq.map(BlockHandler::BlockIseq),
                                            block_arg,
                                        })));
                                        fun.insn_types[ccall.to_usize()] = fun.infer_type(ccall);
                                        fun.make_equal_to(send_insn_id, ccall);
                                        Ok(())
                                    }
                                    -2 => {
                                        // (self, args_ruby_array)
                                        fun.set_dynamic_send_reason(send_insn_id, SendCfuncArrayVariadic);
                                        Err(())
                                    }
                                    _ => unreachable!("unknown cfunc kind: cfunc_argc={cfunc_argc}")
                                }
                            }

                            let ccall_argc = if send_mid_override.is_some() { args.len() as u32 } else { unsafe { vm_ci_argc(ci) } };
                            if reduce_send_to_ccall(self, block, insn_id, recv, cd, send_block, args, state, klass, profiled_type, cme, mid, ccall_argc, flags_for_check, send_block_arg).is_ok() {
                                continue;
                            }

                            self.push_insn_id(block, insn_id);
                        } else {
                            self.set_dynamic_send_reason(insn_id, SendNotOptimizedMethodType(MethodType::from(def_type)));
                            self.push_insn_id(block, insn_id); continue;
                        }
                    }
                    &Insn::IsMethodCfunc { val, cd, cfunc, state } if self.type_of(val).ruby_object_known() => {
                        let class = self.type_of(val).ruby_object().unwrap();
                        let cd_owner = self.frame_state_iseq(state);
                        let cme = unsafe { rb_zjit_vm_search_method(cd_owner.into(), cd as *mut rb_call_data, class) };
                        let is_expected_cfunc = unsafe { rb_zjit_cme_is_cfunc(cme, cfunc as *const c_void) };
                        let method = unsafe { rb_vm_ci_mid((*cd).ci) };
                        self.push_insn(block, Insn::PatchPoint { invariant: Invariant::MethodRedefined { klass: class, method, cme }, state });
                        let replacement = self.push_insn(block, Insn::Const { val: Const::CBool(is_expected_cfunc) });
                        self.insn_types[replacement.to_usize()] = self.infer_type(replacement);
                        self.make_equal_to(insn_id, replacement);
                    }
                    &Insn::ObjectAlloc { val, state } => {
                        if let Some(replacement) = self.try_inline_object_alloc(block, val, state) {
                            self.insn_types[replacement.to_usize()] = self.infer_type(replacement);
                            self.make_equal_to(insn_id, replacement);
                        } else {
                            self.push_insn_id(block, insn_id);
                        }
                    }
                    &Insn::NewRange { low, high, flag, state } => {
                        let low_is_fix  = self.is_a(low,  types::Fixnum);
                        let high_is_fix = self.is_a(high, types::Fixnum);

                        if low_is_fix || high_is_fix {
                            let low_fix = self.coerce_to(block, low, types::Fixnum, state);
                            let high_fix = self.coerce_to(block, high, types::Fixnum, state);
                            let replacement = self.push_insn(block, Insn::NewRangeFixnum { low: low_fix, high: high_fix, flag, state });
                            self.make_equal_to(insn_id, replacement);
                            self.insn_types[replacement.to_usize()] = self.infer_type(replacement);
                        } else {
                            self.push_insn_id(block, insn_id);
                        };
                    }
                    &Insn::InvokeSuper { recv, cd, blockiseq, state, .. } => {
                        // Helper to emit common guards for super call optimization.
                        fn emit_super_call_guards(
                            fun: &mut Function,
                            block: BlockId,
                            super_cme: *const rb_callable_method_entry_t,
                            current_cme: *const rb_callable_method_entry_t,
                            mid: ID,
                            state: InsnId,
                            local_iseq: IseqPtr,
                        ) {
                            fun.push_insn(block, Insn::PatchPoint {
                                invariant: Invariant::MethodRedefined {
                                    klass: unsafe { (*super_cme).defined_class },
                                    method: mid,
                                    cme: super_cme
                                },
                                state
                            });

                            // Get the EP of the ISeq of the containing method, or "local level", skipping over block-level EPs.
                            // Equivalent of GET_LEP() macro. The iseq is the FrameState's, not the
                            // outer compilation's, so that an inlined super call walks from the
                            // callee's CFP rather than the caller's.
                            let level = get_lvar_level(local_iseq);
                            let lep = fun.get_ep(block, level);
                            // Load ep[VM_ENV_DATA_INDEX_ME_CREF]
                            let me_cref = fun.load_field(block, lep, FieldName::VM_ENV_DATA_INDEX_ME_CREF, SIZEOF_VALUE_I32 * VM_ENV_DATA_INDEX_ME_CREF, types::RubyValue);
                            // The slot holds an imemo_svar wrapping the method entry once the
                            // frame has touched a special variable, so read through it the same
                            // way rb_vm_frame_method_entry (and so the profile) does.
                            let method_entry = fun.push_insn(block, Insn::UnwrapSvar { val: me_cref });
                            // Guard that it matches the expected CME. Recompile on a miss: the
                            // profiled CME is not always the one the frame runs with, and exiting
                            // on every call is far worse than dispatching super dynamically.
                            fun.push_insn(block, Insn::GuardBitEquals { val: method_entry, expected: Const::Value(current_cme.into()), reason: Box::new(SideExitReason::GuardSuperMethodEntry), state, recompile: Some(Recompile) });

                            let block_handler = fun.load_field(block, lep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, SIZEOF_VALUE_I32 * VM_ENV_DATA_INDEX_SPECVAL, types::RubyValue);
                            fun.push_insn(block, Insn::GuardBitEquals {
                                val: block_handler,
                                expected: Const::Value(VALUE(VM_BLOCK_HANDLER_NONE as usize)),
                                reason: Box::new(SideExitReason::UnhandledBlockArg),
                                state,
                                recompile: Some(Recompile),
                            });
                        }

                        // Don't handle calls with literal blocks (e.g., super { ... })
                        if !blockiseq.is_null() {
                            self.push_insn_id(block, insn_id);
                            self.set_dynamic_send_reason(insn_id, SuperCallWithBlock);
                            continue;
                        }

                        // Specializing `super` requires guarding the frame's method entry against
                        // one CME. When a previous version's guard kept missing and we have run
                        // out of versions, stop guessing and dispatch `super` dynamically: staying
                        // in JIT code costs far less than side-exiting on every call.
                        if self.policy.no_side_exits {
                            self.push_insn_id(block, insn_id);
                            self.set_dynamic_send_reason(insn_id, SuperMethodEntryUnstable);
                            continue;
                        }

                        let (frame_state_iseq, frame_state_insn_idx) = {
                            let frame_state = self.frame_state_ref(state);
                            (frame_state.iseq, frame_state.insn_idx)
                        };

                        // Don't handle super in a block since that needs a loop to find the running CME.
                        if frame_state_iseq != unsafe { rb_get_iseq_body_local_iseq(frame_state_iseq) } {
                            self.push_insn_id(block, insn_id);
                            self.set_dynamic_send_reason(insn_id, SuperFromBlock);
                            continue;
                        }

                        let ci = unsafe { (*cd).ci };
                        let flags = unsafe { rb_vm_ci_flag(ci) };
                        assert!(flags & VM_CALL_FCALL != 0);

                        // Reject calls with complex argument handling.
                        if unspecializable_c_call_type(flags) {
                            self.push_insn_id(block, insn_id);
                            self.set_dynamic_send_reason(insn_id, SuperComplexArgsPass);
                            continue;
                        }

                        // Get the profiled CME from the current method.
                        if self.profiles.is_none() {
                            self.push_insn_id(block, insn_id);
                            self.set_dynamic_send_reason(insn_id, SuperNoProfiles);
                            continue;
                        }

                        // Use frame_state_iseq so that an inlined super call looks up its
                        // profiled CME against the callee's payload rather than the outer
                        // compilation's. The runtime guard walks from the live CFP, which is
                        // the callee's CFP for inlined code, so the profile lookup must agree.
                        let local_payload = get_or_create_iseq_payload(frame_state_iseq);
                        let Some(current_cme) = local_payload.profile.get_super_method_entry(frame_state_insn_idx) else {
                            self.push_insn_id(block, insn_id);

                            // The absence of the super CME could be due to a missing profile, but
                            // if we've made it this far the value would have been deleted, indicating
                            // that the call is at least polymorphic and possibly megamorphic.
                            self.set_dynamic_send_reason(insn_id, SuperPolymorphic);
                            continue;
                        };

                        // Get defined_class and method ID from the profiled CME.
                        let current_defined_class = unsafe { (*current_cme).defined_class };
                        let mid = unsafe { get_def_original_id((*current_cme).def) };

                        // Compute superclass: RCLASS_SUPER(RCLASS_ORIGIN(defined_class))
                        let superclass = unsafe { rb_class_get_superclass(RCLASS_ORIGIN(current_defined_class)) };
                        if superclass.nil_p() {
                            self.push_insn_id(block, insn_id);
                            self.set_dynamic_send_reason(insn_id, SuperClassNotFound);
                            continue;
                        }

                        // Look up the super method.
                        let mut super_cme = unsafe { rb_callable_method_entry(superclass, mid) };
                        if super_cme.is_null() {
                            self.push_insn_id(block, insn_id);
                            self.set_dynamic_send_reason(insn_id, SuperTargetNotFound);
                            continue;
                        }

                        let mut def_type = unsafe { get_cme_def_type(super_cme) };
                        while def_type == VM_METHOD_TYPE_ALIAS {
                            super_cme = unsafe { rb_aliased_callable_method_entry(super_cme) };
                            def_type = unsafe { get_cme_def_type(super_cme) };
                        }

                        let args = match resolved.insn(self) {
                            Insn::InvokeSuper { args, .. } => args.to_vec(),
                            _ => unreachable!("expected InvokeSuper insn"),
                        };

                        if def_type == VM_METHOD_TYPE_ISEQ {
                            // Check if the super method's parameters support direct send.
                            // If not, we can't do direct dispatch.
                            let super_iseq = unsafe { get_def_iseq_ptr((*super_cme).def) };
                            // TODO: pass Option<blockiseq> to build_send_direct_args when we start specializing `super { ... }`.
                            let Ok(call) = self.build_send_direct_args(&args, ci, super_iseq, false, false, false)
                                .inspect_err(|failure| failure.record(self, block, insn_id, SendDirectFallbackContext::Super)) else {
                                self.push_insn_id(block, insn_id); continue;
                            };

                            emit_super_call_guards(self, block, super_cme, current_cme, mid, state, frame_state_iseq);

                            let SendDirectArgs { state: send_state, args: send_args, kw_bits, jit_entry_idx } =
                                self.emit_send_direct_args(block, call, &args, state);
                            // Use SendDirect with the super method's CME and ISEQ.
                            let replacement = self.push_insn(block, Insn::SendDirect(Box::new(SendDirectData {
                                recv,
                                cd,
                                cme: super_cme,
                                iseq: super_iseq,
                                args: send_args,
                                kw_bits,
                                jit_entry_idx,
                                state: send_state,
                                guard_state: state,
                                block: None,
                                block_arg: None,
                            })));
                            self.make_equal_to(insn_id, replacement);

                        } else if def_type == VM_METHOD_TYPE_CFUNC {
                            let cfunc = unsafe { get_cme_def_body_cfunc(super_cme) };
                            let cfunc_argc = unsafe { get_mct_argc(cfunc) };
                            let cfunc_ptr = unsafe { get_mct_func(cfunc) }.cast();

                            let props = ZJITState::get_method_annotations().get_cfunc_properties(super_cme);
                            if props.is_none() && get_option!(stats) {
                                self.count_not_annotated_cfunc(block, super_cme);
                            }
                            let props = props.unwrap_or_default();

                            match cfunc_argc {
                                // C function with fixed argument count.
                                0.. => {
                                    // Check argc matches
                                    if args.len() != cfunc_argc as usize {
                                        self.push_insn_id(block, insn_id);
                                        self.set_dynamic_send_reason(insn_id, ArgcParamMismatch);
                                        continue;
                                    }
                                    // TODO: Support passing arguments on the stack in C calls
                                    // +1 for self
                                    if args.len()+1 > C_ARG_OPNDS.len() {
                                        self.push_insn_id(block, insn_id);
                                        self.set_dynamic_send_reason(insn_id, TooManyArgsForLir);
                                        continue;
                                    }

                                    emit_super_call_guards(self, block, super_cme, current_cme, mid, state, frame_state_iseq);

                                    // Try inlining the cfunc into HIR
                                    let tmp_block = self.new_block(u32::MAX);
                                    if let Some(replacement) = (props.inline)(self, tmp_block, recv, &args, state) {
                                        // Copy contents of tmp_block to block
                                        assert_ne!(block, tmp_block);
                                        let insns = std::mem::take(&mut self.blocks[tmp_block.to_usize()].insns);
                                        self.blocks[block.to_usize()].insns.extend(insns);
                                        self.count(block, Counter::inline_cfunc_optimized_send_count);
                                        self.make_equal_to(insn_id, replacement);
                                        self.infer_inlined_type(replacement);
                                        self.remove_block(tmp_block);
                                        continue;
                                    }

                                    // Use CCallWithFrame for the C function.
                                    let name = unsafe { (*super_cme).called_id };
                                    let owner = unsafe { (*super_cme).owner };
                                    let return_type = props.return_type;
                                    let elidable = props.elidable;
                                    // Filter for a leaf and GC free function
                                    let ccall = if props.leaf && props.no_gc {
                                        self.count(block, Counter::inline_cfunc_optimized_send_count);
                                        self.push_insn(block, Insn::CCall { cfunc: cfunc_ptr, recv, args, name, owner, return_type, elidable })
                                    } else {
                                        if get_option!(stats) {
                                            self.count_not_inlined_cfunc(block, super_cme);
                                        }
                                        self.push_insn(block, Insn::CCallWithFrame(Box::new(CCallWithFrameData {
                                            cd,
                                            cfunc: cfunc_ptr,
                                            recv,
                                            args,
                                            cme: super_cme,
                                            name,
                                            state,
                                            return_type,
                                            elidable,
                                            block: None,
                                            block_arg: None,
                                        })))
                                    };
                                    self.make_equal_to(insn_id, ccall);
                                }

                                // Variadic C function: func(int argc, VALUE *argv, VALUE recv)
                                -1 => {
                                    emit_super_call_guards(self, block, super_cme, current_cme, mid, state, frame_state_iseq);

                                    // Try inlining the cfunc into HIR
                                    let tmp_block = self.new_block(u32::MAX);
                                    if let Some(replacement) = (props.inline)(self, tmp_block, recv, &args, state) {
                                        // Copy contents of tmp_block to block
                                        assert_ne!(block, tmp_block);
                                        let insns = std::mem::take(&mut self.blocks[tmp_block.to_usize()].insns);
                                        self.blocks[block.to_usize()].insns.extend(insns);
                                        self.count(block, Counter::inline_cfunc_optimized_send_count);
                                        self.make_equal_to(insn_id, replacement);
                                        self.infer_inlined_type(replacement);
                                        self.remove_block(tmp_block);
                                        continue;
                                    }

                                    // Use CCallVariadic for the variadic C function.
                                    let name = unsafe { (*super_cme).called_id };
                                    let owner = unsafe { (*super_cme).owner };
                                    let return_type = props.return_type;
                                    let elidable = props.elidable;
                                    // Filter for a leaf and GC free function
                                    let ccall = if props.leaf && props.no_gc {
                                        self.count(block, Counter::inline_cfunc_optimized_send_count);
                                        self.push_insn(block, Insn::CCall { cfunc: cfunc_ptr, recv, args, name, owner, return_type, elidable })
                                    } else {
                                        if get_option!(stats) {
                                            self.count_not_inlined_cfunc(block, super_cme);
                                        }
                                        self.push_insn(block, Insn::CCallVariadic(Box::new(CCallVariadicData {
                                            cfunc: cfunc_ptr,
                                            recv,
                                            args,
                                            cme: super_cme,
                                            name,
                                            state,
                                            return_type,
                                            elidable,
                                            block: None,
                                            block_arg: None,
                                        })))
                                    };
                                    self.make_equal_to(insn_id, ccall);
                                }

                                // Array-variadic: (self, args_ruby_array).
                                -2 => {
                                    self.push_insn_id(block, insn_id);
                                    self.set_dynamic_send_reason(insn_id, SuperNotOptimizedMethodType(MethodType::Cfunc));
                                    continue;
                                }
                                _ => unreachable!("unknown cfunc argc: {}", cfunc_argc)
                            }
                        } else {
                            // Other method types (not ISEQ or CFUNC)
                            self.push_insn_id(block, insn_id);
                            self.set_dynamic_send_reason(insn_id, SuperNotOptimizedMethodType(MethodType::from(def_type)));
                            continue;
                        }
                    }
                    &Insn::InvokeBuiltin { bf, recv, ref args, state, .. } => {
                        // Builtins reached through inline_methods are translated to HIR
                        // before their operand types are inferred, so their
                        // annotation-based inlining may have failed. Retry now that
                        // types are known.
                        let args = args.to_vec();
                        match self.try_inline_builtin_body(block, bf, recv, &args, state) {
                            Some(replacement) => { self.make_equal_to(insn_id, replacement); }
                            None => { self.push_insn_id(block, insn_id); }
                        }
                    }
                    _ => { self.push_insn_id(block, insn_id); }
                }
            }
        }
        crate::stats::trace_compile_phase("infer_types", || self.infer_types());
    }

    /// Check whether a callee ISEQ can be inlined.
    fn can_inline(callee_iseq: IseqPtr) -> bool {
        // Inline callees with required, optional, post-required positional, keyword, and
        // block parameters, including callees that dispatch to a passed block with `yield`.
        // Double-splat (kwrest) and forwardable params stay out of the general
        // inliner for now. SendDirect argument emission normalizes rest params to a
        // packed Array, so the inliner can map that Array to the rest local.
        let params = unsafe { callee_iseq.params() };
        if params.flags.forwardable() != 0
            || params.flags.has_kwrest() != 0
        {
            incr_counter!(inline_reject_complex_params);
            return false;
        }

        // Reject callees whose environment pointer can escape (e.g., via binding).
        // TODO (nirvdrum 2026-04-15) The interaction between inlined frames and EP escape hasn't been verified.
        if iseq_ep_starts_escaped(callee_iseq) || iseq_seen_ep_escape(callee_iseq) {
            incr_counter!(inline_reject_ep_escapes);
            return false;
        }

        true
    }

    /// True if inlining `callee_iseq` is what stands between a `yield` inside it and
    /// [`inline_block_at_yield`], which turns the literal block's non-local `return` into
    /// a plain return of this function.
    ///
    /// [`inline_block_at_yield`] only fires when the yielding frame is itself inlined into
    /// the frame the `return` escapes to, so an iterator that stays out of line leaves the
    /// block on the JIT-to-JIT dispatch whose only exit is `throw TAG_RETURN`. That throw
    /// longjmps out of every native JIT frame, and every frame between the catch frame and
    /// the interpreter's `vm_exec` runs interpreted from there on -- far more expensive than
    /// the extra code an oversized iterator costs us. `ary.each { ... return x ... }` is the
    /// common shape: `Array#each` is 41 instructions, well past the ordinary threshold.
    ///
    /// This only relaxes the *size* limit. Every soundness condition still has to hold at the
    /// yield site itself, which [`block_return_inlinable`] rechecks there.
    fn inlining_unlocks_block_return(&self, callee_iseq: IseqPtr, blockiseq: Option<IseqPtr>) -> bool {
        let Some(blockiseq) = blockiseq else { return false };
        // Only worth the extra size if the callee actually yields to the block.
        if !iseq_contains_invokeblock(callee_iseq) {
            return false;
        }
        block_return_inlinable(blockiseq, callee_iseq, self.iseq())
    }

    /// True if inlining `callee_iseq` is what lets a `yield` inside it dispatch straight to
    /// the literal block this call site passes.
    ///
    /// A `yield` only reaches the guard-free single-ISEQ dispatch when the yielding frame was
    /// itself inlined and the block is that frame's literal block; see `inlined_known_block`
    /// in [`add_iseq_to_hir`]. Left out of line, a shared iterator's `invokeblock` sees a
    /// different block on every call, so the monomorphic dispatch's profile never settles and
    /// the polymorphic chain's coverage check rejects the unbounded handler sets that
    /// process-wide `each`/`map`/`times` sites collect. The site then calls
    /// `rb_vm_invokeblock()` forever, which is the largest remaining dynamic-dispatch item on
    /// liquid, rack and erubi.
    ///
    /// This is strictly broader than [`Self::inlining_unlocks_block_return`], which only fires
    /// when the block also has a non-local `return` to erase and the yielding frame lands at
    /// inlining depth 1. The direct dispatch needs neither: any depth will do, and a block
    /// that just runs and falls off the end benefits as much.
    ///
    /// Kept narrow on purpose, because the whole point of a targeted relaxation is not to hand
    /// every oversized callee the bigger budget: the call site has to pass a literal block, the
    /// callee has to contain a `yield`, and the block has to be one the direct dispatch can
    /// actually take. Only the *size* limit is relaxed; the yield site rechecks every
    /// condition itself, and a miss costs nothing beyond the code the inlined body took.
    fn inlining_unlocks_direct_yield(&self, callee_iseq: IseqPtr, blockiseq: Option<IseqPtr>) -> bool {
        let Some(blockiseq) = blockiseq else { return false };
        if !iseq_contains_invokeblock(callee_iseq) {
            return false;
        }
        // `direct_invoke_block_adapt` is what the yield site tests, and it needs the arity the
        // `yield` passes, which is not known here. Test the block at its own parameter count:
        // that is the arity a matching `yield` passes, and the one a lone yielded Array
        // auto-splats into. A `yield` with some other argument count just doesn't take the
        // dispatch, which the site sorts out on its own.
        if !unsafe { rb_simple_iseq_p(blockiseq) } {
            return false;
        }
        let lead_num = unsafe { rb_get_iseq_body_param_lead_num(blockiseq) } as usize;
        if direct_invoke_block_adapt(blockiseq, lead_num).is_err() {
            return false;
        }
        // A block that can `throw` is the one case where inlining the iterator can cost more
        // than the dispatch saves. The throw longjmps past every native frame, and the frame it
        // unwinds into resumes at a mid-ISEQ PC that the compiled exception entries have to
        // cover for execution to get back into JIT code -- with an inlined iterator body in
        // the way there are more such PCs and the dispatch misses more of them, so the frame
        // and its callers run interpreted from there on. On liquid-render, which throws half a
        // million times per run, extending the relaxation to these blocks traded 940K
        // `rb_vm_invokeblock()` calls for 99K failed re-entries and came out 3% behind.
        //
        // [`Self::inlining_unlocks_block_return`] is the right answer for those blocks: it
        // takes the ones whose `throw` inlining can *erase*, so no unwind happens at all.
        !crate::codegen::block_iseq_may_throw(blockiseq)
    }

    /// Decide whether an inlinable callee ISEQ is worth inlining into this
    /// function based on heuristics. `blockiseq` is the literal block the call site passes,
    /// if any.
    fn should_inline(&mut self, callee_iseq: IseqPtr, cme: *const rb_callable_method_entry_t, blockiseq: Option<IseqPtr>) -> bool {
        let threshold = get_option!(inline_threshold);
        if threshold == 0 {
            return false;
        }

        // User-supplied denylist of qualified method names (e.g. `User#name` or
        // `Foo.bar`). Lets us isolate the inliner's contribution from individual
        // problem methods without committing to a heuristic. Anonymous code paths
        // (blocks, procs without a stable method binding) can't be expressed in this
        // format, so they aren't matched and fall through to the rest of the checks.
        // Unsafe deref of `cme` is safe here because `inline_methods` only calls
        // `should_inline` for `SendDirect` instructions, which carry a non-null cme.
        if !cme.is_null() {
            let deny = unsafe { crate::options::OPTIONS.as_ref() }.map(|o| &o.inline_deny);
            if deny.is_some_and(|d| !d.is_empty()) {
                let owner = unsafe { (*cme).owner };
                let method_id = unsafe { get_def_original_id((*cme).def) };
                let qualified = qualified_method_name(owner, method_id);
                if deny.unwrap().contains(&qualified) {
                    incr_counter!(inline_reject_denied);
                    return false;
                }
            }
        }

        // An iterator gets a larger budget when inlining it is the only thing keeping the
        // `yield` inside it off the direct block dispatch: leaving it out of line costs an
        // `rb_vm_invokeblock()` per iteration, and an interpreted unwind per call on top of
        // that when the block has a non-local `return`.
        let unlocks_yield = self.inlining_unlocks_direct_yield(callee_iseq, blockiseq)
            || self.inlining_unlocks_block_return(callee_iseq, blockiseq);

        // Per-caller cumulative budget. Once that count crosses the budget, every further callee
        // is rejected and the optimization fixed-point loop reaches its terminal iteration. See
        // `Options::inline_budget` for the full unit/semantics caveat.
        let budget = get_option!(inline_budget);
        let over_budget = self.inline_budget_exhausted
            || (budget != INLINE_BUDGET_UNLIMITED && self.num_instructions > budget);
        let needs_bonus = over_budget && unlocks_yield;
        if over_budget && (!unlocks_yield || self.yield_inline_bonuses == MAX_YIELD_INLINE_BONUSES) {
            incr_counter!(inline_reject_budget_exceeded);
            return false;
        }

        // Check callee bytecode size against threshold.
        let threshold = if unlocks_yield {
            threshold.saturating_mul(YIELD_INLINE_THRESHOLD_FACTOR)
        } else {
            threshold
        };
        let callee_size = unsafe { get_iseq_encoded_size(callee_iseq) } as usize;
        if callee_size > threshold {
            incr_counter!(inline_reject_too_large);
            return false;
        }

        if needs_bonus {
            self.yield_inline_bonuses += 1;
            incr_counter!(inline_yield_bonus_count);
        }
        true
    }

    /// Inline method calls by replacing eligible SendDirect instructions with the
    /// callee's HIR body. Returns true if any inlining occurred.
    fn inline_methods(&mut self) -> bool {
        // Bail early if inlining is disabled.
        if get_option!(inline_threshold) == 0 {
            return false;
        }

        // Fail fast if inlining is enabled but we've exhausted our inlining budget.
        // Otherwise, `can_inline` and `should_inline` will make local inlining decisions.
        // A function this big only has yield-unlocking callees left to consider, and only
        // while it still has bonuses for them. `should_inline` rejects everything else, so
        // once the bonuses are gone there is nothing to find and the scan is wasted work.
        let budget = get_option!(inline_budget);
        self.inline_budget_exhausted = budget != INLINE_BUDGET_UNLIMITED && self.insns.len() > budget;
        if self.inline_budget_exhausted && self.yield_inline_bonuses == MAX_YIELD_INLINE_BONUSES {
            incr_counter!(inline_reject_budget_exceeded);
            return false;
        }

        let mut did_inline = false;

        // Worklist of blocks left to scan for inlinable SendDirects. Seeded with
        // the function's current RPO so we visit every existing block, and
        // extended with each continuation block we create below. Inlining a SendDirect
        // splits its block: pre-Send instructions stay in `block` and the post-Send
        // tail moves to a fresh `continuation`. That tail may contain further
        // SendDirects that were present before this call started, so queueing
        // `continuation` ensures they get a chance to inline as well.
        //
        // The callee body blocks emitted by add_iseq_to_hir are deliberately
        // NOT enqueued: any Sends they contain are next-level work that only becomes
        // inlinable after a later type_specialize pass promotes them to SendDirect.
        let mut worklist: VecDeque<BlockId> = self.reverse_post_order().into_iter().collect();

        while let Some(block) = worklist.pop_front() {
            // Walk this block looking for an inlinable SendDirect. Under the
            // basic-block invariant, the terminator is the last instruction in
            // the block and SendDirect is never a terminator (it has an
            // output), so every SendDirect lives in the block body and its
            // position can be found by a linear scan. We commit at most one
            // inline per block visit; the post-Send tail that moves to the
            // continuation is rescanned when that continuation comes off the
            // worklist. The cursor below advances past SendDirects we reject
            // (denylist, compile failure, no-return callee) so that a later,
            // inlinable SendDirect in the same block still gets a chance.
            let mut search_start = 0;
            loop {
                let Some(offset) = self.blocks[block.to_usize()].insns[search_start..].iter()
                    .position(|&id| self.is_send_direct(id))
                else {
                    break;
                };
                let send_pos = search_start + offset;

                let send_insn_id = self.blocks[block.to_usize()].insns[send_pos];
                let send = self.resolve(send_insn_id);
                let Insn::SendDirect(data) = send.insn(self)
                else {
                    unreachable!("position {send_insn_id} is not a SendDirect");
                };
                let SendDirectData { recv, cme, iseq, kw_bits, jit_entry_idx, block: call_block, block_arg, state, guard_state, .. } = **data;
                let args_len = data.args.len();
                // A `&blk` block handler lives in the callee frame's specval, which inlining
                // never pushes: the callee's `yield` would read the caller's block instead.
                if block_arg.is_some() {
                    search_start = send_pos + 1;
                    continue;
                }
                // SendDirect invariant: block is either None or BlockIseq.
                // BlockArg is rejected upstream during type specialization.
                // TODO(max): If we accept BlockArg here, we need to change the folding of Defined
                // in HIR construction for the defined opcode to check the send flags of the method
                // being inlined, too.
                let blockiseq: Option<IseqPtr> = call_block.map(|bh| match bh {
                    BlockHandler::BlockIseq(bi) => bi,
                    BlockHandler::BlockArg => unreachable!("BlockArg in SendDirect"),
                });

                // Apply the cheap optimization heuristics (size, budget, denylist)
                // before can_inline's more expensive elibility checks. This allows
                // oversized callees to bail out early. Both guards must pass.
                if !self.should_inline(iseq, cme, blockiseq) || !Self::can_inline(iseq) {
                    search_start = send_pos + 1;
                    continue;
                }

                // Snapshot the caller's HIR length so we can roll back if compiling
                // the callee fails or its body has no return paths. add_iseq_to_hir
                // appends to the caller in place, so on rejection we truncate the
                // instruction, type, and block tables back to these lengths,
                // discarding the partial translation. Union-find needs no snapshot:
                // HIR construction only appends instructions and never calls
                // make_equal_to or find, so the forwarding table is untouched between
                // here and the rejection points below. The original block isn't
                // touched until we commit to the inline below, so rejection paths
                // only need to advance the search cursor without restoring it.
                let pre_insns_len = self.insns.len();
                let pre_insn_types_len = self.insn_types.len();
                let pre_blocks_len = self.blocks.len();

                // Pick the callee body entry matching how many optional parameters
                // the caller actually filled. Entry index `k` is where execution
                // begins when `lead_num + k + post_num + kw_num` arguments are
                // passed: it runs the default-init code for the remaining
                // `opt_num - k` optionals (if any) before falling through into the
                // post-default body. SendDirect argument planning records the matching
                // `jit_entry_idx` before packing the caller args, so do not recover the optional
                // positional count from args.len() here.
                let callee_params = unsafe { iseq.params() };
                let lead_num = callee_params.lead_num as usize;
                let opt_num = callee_params.opt_num as usize;
                let post_num = callee_params.post_num as usize;
                let kw_num = callee_kw_num(iseq);
                let rest_slots = usize::from(callee_params.flags.has_rest() != 0);
                let passed_opt_num = jit_entry_idx as usize;

                // Create the continuation block before translating the callee so it
                // can serve as the return_block argument; the callee's leaves become
                // Jumps to this block directly during translation. The continuation
                // is included in the pre-state snapshot above so a rollback also
                // discards it. Execution resumes in the caller at the instruction
                // following the call, so label the block with that index rather than
                // the enclosing block's start.
                let call_state = self.frame_state(state);
                let continuation = self.new_block(call_state.insn_idx() as u32 + insn_len(call_state.get_opcode() as usize));

                // Inlining works top-down: a method's own calls only become inlinable in a
                // later fixed-point iteration. So, the call site's depth is known here. The
                // callee sits one level deeper (caller_depth + 1).
                let caller_depth = self.frame_depth(state);

                // The callee's perspective of the stack is with the receiver and arguments popped off.
                let caller_stack_size = call_state.stack_size() - args_len - 1; // -1 for receiver
                let post_send_caller = self.new_insn(Insn::Snapshot { state: Box::new(call_state.with_stack_size(caller_stack_size)) });
                let mode = AddIseqMode::Inlined {
                    return_block: continuation,
                    caller: post_send_caller,
                    depth: caller_depth + 1,
                    jit_entry_idx: passed_opt_num,
                    blockiseq,
                    block_return_pops: None,
                };
                let add_result = match add_iseq_to_hir(self, iseq, mode) {
                    Ok(r) => r,
                    Err(_) => {
                        self.insns.truncate(pre_insns_len);
                        self.insn_types.truncate(pre_insn_types_len);
                        self.blocks.truncate(pre_blocks_len);
                        incr_counter!(inline_reject_compile_failure);
                        search_start = send_pos + 1;
                        continue;
                    }
                };

                // Bump the rough estimate of new instructions added by inlining this function
                self.num_instructions += self.insns.len() - pre_insns_len;

                // Past the point of no return: commit the inlining.
                incr_counter!(inline_method_count);
                did_inline = true;

                let args = match send.insn(self) {
                    Insn::SendDirect(data) => data.args.to_vec(),
                    _ => unreachable!("position {send_insn_id} is not a SendDirect"),
                };

                // Split the original block at the SendDirect's position. Pre-Send
                // instructions stay in `block`; the SendDirect itself is consumed
                // (we alias its uses to the continuation's return-value Param
                // below); everything after, including the original terminator,
                // moves onto the continuation. We split before adding new
                // instructions to either block so that the param-initialization
                // constants land in `block` at the correct position (after the
                // pre-Send body, before the PushLightweightFrame and Jump we add
                // last).
                let tail = self.blocks[block.to_usize()].insns.split_off(send_pos);
                debug_assert!(self.is_send_direct(tail[0]));

                let omitted_opt_num = opt_num - passed_opt_num;
                let positional_kw_end = lead_num + opt_num + rest_slots + post_num + kw_num;
                let kw_bits_local_idx = callee_kw_bits_local_idx(iseq);
                let callee_entry_body_block = add_result.body_entry_block
                    .expect("inlined compilation always produces a body entry block");

                // Map callee body entry params to caller values:
                //
                // The callee's body entry block has params: [self, local0, local1, ..., stack0, stack1, ...]
                // The first param is self (recv); the rest follow the callee's local
                // table order (lead, opt, post, kw, then any hidden kw_bits slot, then
                // non-parameter locals). The caller pushes args without gaps, so we map
                // locals to args by category:
                //
                //   * lead locals (indices 0..lead_num) and filled optional locals
                //     (lead_num..lead_num + passed_opt_num) take args in order.
                //   * unfilled optional locals (lead_num + passed_opt_num..lead_num + opt_num)
                //     are nil-initialized; the body's default-init code overwrites them.
                //   * post-required and keyword locals
                //     (lead_num + opt_num..lead_num + opt_num + post_num + kw_num) take the
                //     trailing args, but their position in the local table leaves a gap
                //     of (opt_num - passed_opt_num) above the args' compact layout, so we
                //     shift the arg index down by that amount. Keyword args follow this
                //     same arithmetic because SendDirect argument planning already
                //     reordered them into callee table order with defaults filled in.
                //   * the hidden kw_bits storage local (when the callee has keywords) is
                //     aliased to the SendDirect's compile-time kw_bits value as a fixnum
                //     constant. `checkkeyword` lowers to `FixnumBitCheck` against this
                //     constant, and FrameState materialization on a side exit writes the
                //     same value back to the runtime frame so a resuming interpreter sees
                //     the correct bitmask.
                //   * any remaining non-parameter locals are nil-initialized.
                let callee_body_params: Vec<InsnId> = self.blocks[callee_entry_body_block.to_usize()].params.clone();

                // First param is self.
                if !callee_body_params.is_empty() {
                    self.make_equal_to(callee_body_params[0], recv);
                }

                // Next params are locals.
                let num_locals = callee_body_params.len() - 1; // -1 for self
                for (i, &param_id) in callee_body_params[1..].iter().enumerate() {
                    if i < lead_num + passed_opt_num {
                        // Lead local or filled optional: arg index matches local index.
                        self.make_equal_to(param_id, args[i]);
                    } else if i < lead_num + opt_num {
                        // Unfilled optional: nil-initialized; default-init code will overwrite.
                        let nil = self.push_insn(block, Insn::Const { val: Const::Value(Qnil) });
                        self.make_equal_to(param_id, nil);
                    } else if i < positional_kw_end {
                        // Post-required or keyword local: the arg sits compactly after
                        // the filled optionals, so shift the arg index down by the gap
                        // of unfilled optionals between the optional and post regions.
                        self.make_equal_to(param_id, args[i - omitted_opt_num]);
                    } else if Some(i) == kw_bits_local_idx {
                        // Hidden kw_bits slot: alias to the SendDirect's compile-time
                        // value as a fixnum, the same encoding the interpreter uses for
                        // this hidden local. checkkeyword's FixnumBitCheck will read
                        // this constant directly inside the inlined body.
                        let bits_const = self.push_insn(block, Insn::Const {
                            val: Const::Value(VALUE::fixnum_from_usize(kw_bits as usize)),
                        });
                        self.make_equal_to(param_id, bits_const);
                    } else if i < num_locals {
                        // Non-parameter local: nil-initialized.
                        let nil = self.push_insn(block, Insn::Const { val: Const::Value(Qnil) });
                        self.make_equal_to(param_id, nil);
                    }
                }

                // Clear the callee body entry block's params since we've aliased
                // them via make_equal_to rather than passing them as branch
                // arguments. This keeps validation happy (the Jump passes 0 args).
                self.blocks[callee_entry_body_block.to_usize()].params.clear();

                // Set up the continuation block: a single Param merges all return
                // values jumped in from the callee's leaves, then PopLightweightFrame
                // unwinds the inlined frame before any post-call code runs. The
                // tail collected above (post-Send instructions plus the original
                // terminator) is grafted on after, skipping the SendDirect at
                // tail[0] which is being consumed.
                let return_val_param = self.push_insn(continuation, Insn::Param);
                self.push_insn(continuation, Insn::PopInlineFrame {
                    iseq,
                    argc: args.len(),
                    state,
                });

                // Start at 1 to skip over the SendDirect at position 0.
                for &id in &tail[1..] {
                    self.push_insn_id(continuation, id);
                }

                // The original SendDirect result is now the continuation's return value param.
                self.make_equal_to(send_insn_id, return_val_param);

                // Keep the caller FrameState that inlined callee Snapshots point at
                // as a distinct Snapshot instead of rewriting the consumed SendDirect.
                self.push_insn_id(block, post_send_caller);

                // Insert PushLightweightFrame and jump to callee body entry.
                self.push_insn(block, Insn::PushInlineFrame {
                    iseq, cme, recv, num_args: args.len().try_into().unwrap(), blockiseq, captured: None, state, guard_state,
                });
                self.count(block, Counter::inline_iseq_optimized_send_count);
                self.push_insn(block, Insn::Jump(BranchEdge {
                    target: callee_entry_body_block,
                    args: vec![],
                }));

                // Append the callee's profile entries. The callee body was emitted directly into
                // this Function, so its Snapshot and operand InsnIds already live in caller space.
                if let Some(caller_profiles) = self.profiles.as_mut() {
                    caller_profiles.append(&add_result.profiles);
                } else {
                    self.profiles = Some(add_result.profiles);
                }

                // The post-Send tail now lives in `continuation` and may itself
                // contain further inlinable SendDirects. Queue it for scanning
                // so we handle every SendDirect at the current level in this
                // single inline_methods call.
                worklist.push_back(continuation);

                // Done with this block: the rest of it is in `continuation`.
                break;
            }
        }

        if did_inline {
            self.infer_types();
        }

        did_inline
    }

    fn load_shape(&mut self, block: BlockId, recv: InsnId) -> InsnId {
        self.load_field(block, recv, FieldName::shape_id, unsafe { rb_shape_id_offset() } as i32, types::CShape)
    }

    fn guard_shape(&mut self, block: BlockId, val: InsnId, expected: ShapeId, state: InsnId, recompile: Option<Recompile>) -> InsnId {
        self.push_insn(block, Insn::GuardBitEquals {
            val,
            expected: Const::CShape(expected),
            reason: Box::new(SideExitReason::GuardShape(expected)),
            state,
            recompile,
        })
    }

    fn load_ivar_c_call(&mut self, block: BlockId, recv: InsnId, ivar_index: attr_index_t) -> InsnId {
        // NOTE: it's fine to use rb_ivar_get_at_no_ractor_check because
        // getinstancevariable does assume_single_ractor_mode()
        let ivar_index_insn = self.push_insn(block, Insn::Const { val: Const::CAttrIndex(ivar_index) });
        self.push_insn(block, Insn::CCall {
            cfunc: rb_ivar_get_at_no_ractor_check as *const u8,
            recv,
            args: vec![ivar_index_insn],
            name: ID!(rb_ivar_get_at_no_ractor_check),
            owner: Qnil,
            return_type: types::BasicObject,
            elidable: true })
    }

    fn load_ivar_embedded(&mut self, block: BlockId, recv: InsnId, id: ID, ivar_index: attr_index_t) -> InsnId {
        // See ROBJECT_FIELDS() from include/ruby/internal/core/robject.h
        let offset = ROBJECT_OFFSET_AS_ARY
            + (SIZEOF_VALUE * ivar_index.to_usize()) as i32;
        self.load_field(block, recv, id.into(), offset, types::BasicObject)
    }

    /// Guard that `recv` is a heap allocated object
    fn guard_heap(&mut self, block: BlockId, recv: InsnId, state: InsnId) -> InsnId {
        self.push_insn(block, Insn::GuardType { val: recv, guard_type: types::HeapBasicObject, state, recompile: None })
    }

    fn load_ivar(&mut self, block: BlockId, self_val: InsnId, recv_type: ProfiledType, id: ID) -> InsnId {
        // Too-complex shapes use hash tables; rb_shape_get_iv_index doesn't support them.
        // Callers must filter these out before calling load_ivar.
        assert!(!recv_type.shape().is_complex(), "load_ivar called with too-complex shape");
        let mut ivar_index: attr_index_t = 0;
        if ! unsafe { rb_shape_get_iv_index(recv_type.shape().0, id, &mut ivar_index) } {
            // If there is no IVAR index, then the ivar was undefined when we
            // entered the compiler.  That means we can just return nil for this
            // shape + iv name
            return self.push_insn(block, Insn::Const { val: Const::Value(Qnil) });
        }

        let layout = recv_type.shape().layout();

        match layout {
            ShapeLayout::RClass | ShapeLayout::Extended => {
                let offset = if layout == ShapeLayout::RClass {
                    RCLASS_OFFSET_PRIME_FIELDS_OBJ
                } else {
                    TDATA_OFFSET_FIELDS_OBJ
                };

                let fields_obj = self.load_field(block, self_val, FieldName::fields_obj, offset, types::IMemo);
                // All fields objects are embedded
                self.load_ivar_embedded(block, fields_obj, id, ivar_index)
            },
            ShapeLayout::RObject => {
                self.load_ivar_embedded(block, self_val, id, ivar_index)
            },
            ShapeLayout::Other => {
                // Non-T_OBJECT, non-class/module, non-typed-data: fall back to C call
                // NOTE: it's fine to use rb_ivar_get_at_no_ractor_check because
                // getinstancevariable does assume_single_ractor_mode()
                self.load_ivar_c_call(block, self_val, ivar_index)
            }
        }
    }

    fn getivar_fallback_reason(resolution: ReceiverTypeResolution, ic: *const iseq_inline_iv_cache_entry) -> Counter {
        match resolution {
            ReceiverTypeResolution::Megamorphic => Counter::getivar_fallback_megamorphic,
            ReceiverTypeResolution::SkewedMegamorphic { .. } => Counter::getivar_fallback_skewed_megamorphic,
            ReceiverTypeResolution::Polymorphic => Counter::getivar_fallback_polymorphic,
            ReceiverTypeResolution::NoProfile if ic.is_null() => Counter::getivar_fallback_no_profile_missing_ic,
            ReceiverTypeResolution::NoProfile => Counter::getivar_fallback_no_profile,
            _ => Counter::getivar_fallback_not_monomorphic,
        }
    }

    fn try_emit_optimized_getivar(&mut self, block: BlockId, self_val: InsnId, id: ID, profiled_type: ProfiledType, state: InsnId) -> Result<InsnId, Counter> {
        if profiled_type.flags().is_immediate() {
            // Instance variable lookups on immediate values are always nil
            return Err(Counter::getivar_fallback_immediate);
        }
        assert!(profiled_type.shape().is_valid());
        if profiled_type.shape().is_complex() {
            // too-complex shapes can't use index access
            return Err(Counter::getivar_fallback_complex);
        }
        if self.policy.no_side_exits {
            // On the final version, skip GetIvar shape specialization.
            // iseq_to_hir already generates polymorphic branches with a
            // GetIvar C call fallback for getinstancevariable, so we don't
            // need to wrap it again here.
            return Err(Counter::getivar_fallback_no_side_exits);
        }
        let self_val = self.guard_heap(block, self_val, state);
        let shape = self.load_shape(block, self_val);
        self.guard_shape(block, shape, profiled_type.shape(), state, Some(Recompile));
        Ok(self.load_ivar(block, self_val, profiled_type, id))
    }

    fn prepare_optimized_setivar(&mut self, id: ID, profiled_type: ProfiledType) -> Result<SetIvarSpec, Counter> {
        if profiled_type.flags().is_immediate() {
            // Instance variable writes on immediate values raise.
            return Err(Counter::setivar_fallback_immediate);
        }

        let extended_robject = match profiled_type.shape().layout() {
            ShapeLayout::RObject => false,
            ShapeLayout::Extended if profiled_type.flags().is_t_object() => true,
            _ => return Err(Counter::setivar_fallback_not_t_object),
        };

        assert!(profiled_type.shape().is_valid());
        if profiled_type.shape().is_frozen() {
            // Can't set ivars on frozen objects
            return Err(Counter::setivar_fallback_frozen);
        }
        if profiled_type.shape().is_complex() {
            // too-complex shapes can't use index access
            return Err(Counter::setivar_fallback_complex);
        }
        let mut ivar_index: attr_index_t = 0;
        let mut next_shape = profiled_type.shape();
        if !unsafe { rb_shape_get_iv_index(profiled_type.shape().0, id, &mut ivar_index) } {
            // Updating the fields object's shape requires preserving its private layout and
            // capacity bits, which can differ from the owning RObject's. Existing ivars do not
            // change either shape, so they can still use the fast path.
            if extended_robject {
                return Err(Counter::setivar_fallback_shape_transition);
            }

            // Current shape does not contain this ivar; do a shape transition.
            let current_shape_id = profiled_type.shape();
            let class = profiled_type.class();
            // We're only looking at T_OBJECT so ignore all of the imemo stuff.
            assert!(profiled_type.flags().is_t_object());
            next_shape = ShapeId(unsafe { rb_shape_transition_add_ivar_no_warnings(current_shape_id.0, id, class) });
            // If the VM ran out of shapes, or this class generated too many leaf,
            // it may be de-optimized into OBJ_COMPLEX_SHAPE (hash-table).
            let new_shape_complex = unsafe { rb_jit_shape_complex_p(next_shape.0) };
            // TODO(max): Is it OK to bail out here after making a shape transition?
            if new_shape_complex {
                return Err(Counter::setivar_fallback_new_shape_complex);
            }
            let ivar_result = unsafe { rb_shape_get_iv_index(next_shape.0, id, &mut ivar_index) };
            assert!(ivar_result, "New shape must have the ivar index");
            let current_capacity = unsafe { rb_jit_shape_capacity(current_shape_id.0) };
            let next_capacity = unsafe { rb_jit_shape_capacity(next_shape.0) };
            // If the new shape has a different capacity, or is COMPLEX, we'll have to
            // reallocate it.
            let needs_extension = next_capacity != current_capacity;
            if needs_extension {
                return Err(Counter::setivar_fallback_new_shape_needs_extension);
            }
            // Fall through to preparing the ivar write.
        }

        Ok(SetIvarSpec { profiled_type, ivar_index, next_shape })
    }

    fn emit_optimized_setivar(&mut self, block: BlockId, self_val: InsnId, id: ID, val: InsnId, spec: SetIvarSpec) {
        let offset = ROBJECT_OFFSET_AS_ARY + (SIZEOF_VALUE * spec.ivar_index.to_usize()) as i32;

        // See ROBJECT_FIELDS() from include/ruby/internal/core/robject.h
        let (ivar_storage, embedded) = match spec.profiled_type.shape().layout() {
            ShapeLayout::RObject => { // AKA embedded
                (self_val, true)
            },
            ShapeLayout::Extended => {
                let fields = self.load_field(block, self_val, FieldName::as_heap, ROBJECT_OFFSET_AS_HEAP_FIELDS, types::IMemo);
                (fields, false)
            },
            ShapeLayout::Other | ShapeLayout::RClass => {
                panic!("This is a T_OBJECT only path (for now)")
            }
        };

        self.push_insn(block, Insn::StoreField { recv: ivar_storage, id: id.into(), offset, val, num_bits: types::BasicObject.num_bits() });
        self.push_insn(block, Insn::WriteBarrier { recv: ivar_storage, val });
        if spec.next_shape != spec.profiled_type.shape() {
            // Write the new shape ID
            let shape_id = self.push_insn(block, Insn::Const { val: Const::CShape(spec.next_shape) });
            let shape_id_offset = unsafe { rb_shape_id_offset() };

            if !embedded {
                // FIXME: We need to strip the SHAPE_ID_FL_PRIVATE_MASK from the shape for `ivar_storage`.
                // see `RBASIC_SET_SHAPE_ID`.
                // This path is currently dead code, see the FIXME in `prepare_optimized_setivar`
                self.push_insn(block, Insn::StoreField { recv: ivar_storage, id: FieldName::shape_id, offset: shape_id_offset, val: shape_id, num_bits: types::CShape.num_bits() });
            }
            self.push_insn(block, Insn::StoreField { recv: self_val, id: FieldName::shape_id, offset: shape_id_offset, val: shape_id, num_bits: types::CShape.num_bits() });
        }
    }

    fn gen_patch_points_for_optimized_ccall(&mut self, block: BlockId, recv_class: VALUE, method_id: ID, cme: *const rb_callable_method_entry_struct, state: InsnId) {
        self.push_insn(block, Insn::PatchPoint { invariant: Invariant::NoTracePoint, state });
        self.push_insn(block, Insn::PatchPoint { invariant: Invariant::MethodRedefined { klass: recv_class, method: method_id, cme }, state });
    }

    /// Side exit back to the state after a block-backed send.
    /// Using the pre-send snapshot would re-execute the send in the interpreter.
    fn gen_post_send_no_ep_escape_patch_point(&mut self, block: BlockId, state: &FrameState, insn_idx: u32) {
        let iseq = state.iseq;
        let mut reload_state = state.clone();
        reload_state.insn_idx = insn_idx as usize;
        reload_state.pc = unsafe { rb_iseq_pc_at_idx(iseq, insn_idx) };
        let reload_exit_id = self.push_insn(block, Insn::Snapshot { state: Box::new(reload_state.without_locals()) });
        self.push_insn(block, Insn::PatchPoint { invariant: Invariant::NoEPEscape(iseq), state: reload_exit_id });
    }

    /// After a call that takes a block iseq, reload the locals that the block (or any iseq nested
    /// within it) may have written. This covers syntactically visible local writes where the
    /// environment does not escape. Exordinary modifications through `Binding` and debug.h APIs are
    /// handled via patchpoints.
    fn reload_locals_modified_by_block(
        &mut self,
        block: BlockId,
        iseq: IseqPtr,
        blockiseq: IseqPtr,
        state: &mut FrameState,
        ep_escaped: bool,
    ) {
        let to_reload: &mut dyn Iterator<Item = usize> = if ep_escaped {
            // Reload everything when working with an escaped environment
            &mut (0..state.locals.len())
        } else {
            // When not escaped, only reload syntactically visible local modifications
            let params = unsafe { iseq.params() };
            let block_param_local_idx: Option<usize> = if params.flags.has_block() != 0 {
                params.block_start.try_into().ok()
            } else {
                None
            };
            let outer_variables = unsafe { blockiseq.outer_variables() };
            &mut (0..state.locals.len()).filter(move |&local_idx| {
                let id = unsafe { rb_zjit_local_id(iseq, local_idx.try_into().unwrap()) };
                let access = outer_variables.local_access(id);
                if block_param_local_idx == Some(local_idx) {
                    // The block param slot is special: `getblockparam` come from a syntactic read,
                    // but operationally can write to the local slot. So, reload it whenever the
                    // block references it at all (read or write), not just on a setlocal.
                    access.is_some()
                } else {
                    access == Some(OuterLocalAccess::ReadWrite)
                }
            })
        };

        let mut base: Option<InsnId> = None;
        for local_idx in to_reload {
            let ep_offset = local_idx_to_ep_offset(iseq, local_idx);
            let ep_offset_u32 = u32::try_from(ep_offset)
                .unwrap_or_else(|_| panic!("Could not convert ep_offset {ep_offset} to u32"));
            let recv = *base.get_or_insert_with(|| {
                let base_insn = if !ep_escaped { Insn::LoadSP } else { Insn::GetEP { level: 0 } };
                self.push_insn(block, base_insn)
            });
            let val = if !ep_escaped {
                self.get_local_from_sp(block, iseq, recv, ep_offset_u32, types::BasicObject)
            } else {
                self.get_local_from_ep(block, iseq, recv, ep_offset_u32, 0, types::BasicObject)
            };
            state.setlocal(ep_offset_u32, val);
        }
    }

    fn count_not_inlined_cfunc(&mut self, block: BlockId, cme: *const rb_callable_method_entry_t) {
        let owner = unsafe { (*cme).owner };
        let called_id = unsafe { (*cme).called_id };
        let qualified_method_name = qualified_method_name(owner, called_id);
        let not_inlined_cfunc_counter_pointers = ZJITState::get_not_inlined_cfunc_counter_pointers();
        let counter_ptr = not_inlined_cfunc_counter_pointers.entry(qualified_method_name.clone()).or_insert_with(|| Box::new(0));
        let counter_ptr = &mut **counter_ptr as *mut u64;

        self.push_insn(block, Insn::IncrCounterPtr { counter_ptr });
    }

    fn count_iseq_calls(&mut self, block: BlockId) {
        let iseq_name = iseq_get_location(self.iseq, 0);
        let access_counter_ptrs = crate::state::ZJITState::get_iseq_calls_count_pointers();
        let counter_ptr = access_counter_ptrs.entry(iseq_name.to_string()).or_insert_with(|| Box::new(0));
        let counter_ptr: &mut u64 = counter_ptr.as_mut();

        self.push_insn(block, Insn::IncrCounterPtr { counter_ptr });
    }

    fn count_not_annotated_cfunc(&mut self, block: BlockId, cme: *const rb_callable_method_entry_t) {
        let owner = unsafe { (*cme).owner };
        let called_id = unsafe { (*cme).called_id };
        let qualified_method_name = qualified_method_name(owner, called_id);
        let not_annotated_cfunc_counter_pointers = ZJITState::get_not_annotated_cfunc_counter_pointers();
        let counter_ptr = not_annotated_cfunc_counter_pointers.entry(qualified_method_name.clone()).or_insert_with(|| Box::new(0));
        let counter_ptr = &mut **counter_ptr as *mut u64;

        self.push_insn(block, Insn::IncrCounterPtr { counter_ptr });
    }

    /// Convert `Send` instructions with no profile data into `SideExit` with recompile info.
    /// This runs after strength reduction passes (type_specialize, inline) so
    /// that sends that can be optimized without profiling (e.g. known CFUNCs) are already handled.
    /// The remaining no-profile sends are turned into side exits that trigger recompilation with
    /// fresh profile data.
    fn convert_no_profile_sends(&mut self) {
        // On the final version, recompilation is not possible, so converting sends to
        // SideExits would just add overhead (the exit fires every time without benefit).
        // Keep them as Send fallbacks so the interpreter handles them directly.
        let payload = get_or_create_iseq_payload(self.iseq);
        if payload.versions.len() + 1 >= payload.version_limit() {
            return;
        }
        for block in self.reverse_post_order() {
            let old_insns = std::mem::take(&mut self.blocks[block.to_usize()].insns);
            assert!(self.blocks[block.to_usize()].insns.is_empty());
            for insn_id in old_insns {
                match self.resolve(insn_id).insn(self) {
                    &Insn::Send { state, reason: SendFallbackReason::SendNoProfiles, .. } => {
                        self.push_insn(block, Insn::SideExit { state, reason: Box::new(SideExitReason::NoProfileSend), recompile: Some(Recompile) });
                        // SideExit is a terminator; don't add remaining instructions
                        break;
                    }
                    _ => {
                        self.push_insn_id(block, insn_id);
                    }
                }
            }
        }
    }

    /// ZJIT uses block parameters in HIR SSA representation.
    /// Sometimes, we can prove that a block param is only called with a single value.
    /// This pass identifies such trivial block params and replaces them with the concretized value.
    /// This produces a minimal SSA representation amenable to further optimizations.
    /// The implementation is inspired from algorithm 2 in <https://c9x.me/compile/bib/braun13cc.pdf>.
    fn remove_trivial_block_params(&mut self) {
        // Each block param is lifted to an abstract domain of ParamValues.
        // The lattice is simple. None is Bottom, Multiple is Top, and One is between both.
        // During analysis, all block params start with None.
        // New values passed to the block transition up the lattice.
        // Trivial block params have one unique value. This is the case we optimize away.
        // Lattice structure taken from cranelift: <https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/codegen/src/remove_constant_phis.rs>
        #[derive(Clone, Copy)]
        enum ParamValue {
            None,
            One(InsnId),
            Many
        }

        impl ParamValue {
            fn update(&mut self, value: InsnId) {
                *self = match *self {
                    ParamValue::None => ParamValue::One(value),
                    ParamValue::One(original) if original != value => ParamValue::Many,
                    other => other
                };
            }
        }

        // Helper function to remove selected indices from a vec in place
        fn prune_vec_by_indices<T>(v: &mut Vec<T>, indices: &[usize]) {
            let mut i: usize = 0;
            v.retain(|_| {
                let valid_id = !indices.contains(&i);
                i += 1;
                valid_id
            })
        }

        fn block_terminator(fun: &Function, block_id: BlockId) -> InsnId {
            *fun.blocks[block_id.to_usize() as usize].insns().last().unwrap()
        }

        macro_rules! edges_of {
            ($insn:expr) => {
                match $insn {
                    Insn::Jump(edge) => [Some(edge), None],
                    Insn::CondBranch { if_true, if_false, .. } => [Some(if_true), Some(if_false)],
                    _ => [None, None],
                }.into_iter().flatten()
            };
        }

        fn outgoing_edges(fun: &Function, block_id: BlockId) -> impl Iterator<Item = &BranchEdge> {
            let insn_id = block_terminator(fun, block_id);
            edges_of!(&fun.insns[insn_id.to_usize()])
        }

        fn outgoing_edges_mut(fun: &mut Function, block_id: BlockId) -> impl Iterator<Item = &mut BranchEdge> {
            let insn_id = block_terminator(fun, block_id);
            edges_of!(&mut fun.insns[insn_id.to_usize()])
        }

        // Instantiate the domain for abstract interpretation.
        // We store possible param values for each block
        let mut param_values: Vec<Vec<ParamValue>> = vec![Vec::new(); self.blocks.len()];

        let blocks = self.reverse_post_order();

        // Collect blocks that terminate with Jump or CondBranch instructions that pass at least one block param along.
        let blocks_sending_params: Vec<BlockId> = blocks.iter().copied()
            .filter(|&block_id|
                outgoing_edges(self, block_id).any(|edge| !edge.args.is_empty()))
            .collect();

        // We only need to update blocks that have params. (Blocks without params cannot be improved)
        let blocks_receiving_params: Vec<BlockId> = blocks.iter().copied()
            .filter(|&block_id|
                self.blocks[block_id.to_usize()].params().len() != 0)
            .collect();

        // Create a vec to represent trivial indices
        let max_params = blocks.iter().copied().map(|id| self.blocks[id.to_usize()].params.len()).max().unwrap_or(0);
        let mut trivial_indices: Vec<usize> = Vec::with_capacity(max_params);

        let mut changed = true;

        while changed {
            changed = false;

            for (row, block) in param_values.iter_mut().zip(&self.blocks) {
                row.resize(block.params.len(), ParamValue::None);
            }

            // Scan through each jump, collecting edges with params to analyze from CondBranch and Jump insns.
            for block_id in &blocks_sending_params {
                // Use the results of abstract interpretation to update the states
                // Perform abstract interpretation
                for BranchEdge { target: block_id, args: params } in outgoing_edges(self, *block_id) {
                    for (i, param) in params.iter().enumerate() {
                        let param = self.find_id(*param);
                        // If the param is the same as passed into the block, it is a self loop and provides no new predecessor information.
                        if param == self.find_id(self.blocks[block_id.to_usize()].params[i]) {
                            continue
                        }
                        param_values[block_id.to_usize()][i].update(param);
                    }
                }
            }

            // Remove the trivial block params and fix up our SSA representation
            // This is done by as follows.
            // 1. Replace uses of the trivial params with the concretized value
            // 2. Remove trivial params from the basic block definition
            // 3. Remove trivial params from each CondBranch and Jump that targets the basic block that was just updated
            for block_id in &blocks_receiving_params {
                let block_preds = &param_values[block_id.to_usize()];
                trivial_indices.clear();
                for (idx, state) in block_preds.iter().enumerate() {
                    if let ParamValue::One(_) = state {
                        trivial_indices.push(idx);
                    }
                }

                // Replace uses of the trivial params with the concretized value
                for param_index in &trivial_indices {
                    if let ParamValue::One(insn_id) = block_preds[*param_index] {
                        self.make_equal_to(self.blocks[block_id.to_usize()].params[*param_index], insn_id);
                        changed = true;
                    }
                }

                // Update the block
                prune_vec_by_indices(&mut self.blocks[block_id.to_usize()].params, &trivial_indices);

                // Update the terminators (basic blocks can only branch at the terminator. This is where block params are passed)
                for jump_block_id in &blocks_sending_params {
                    for edge in outgoing_edges_mut(self, *jump_block_id) {
                        if edge.target == *block_id {
                            prune_vec_by_indices(&mut edge.args, &trivial_indices);
                        }
                    }
                }
            }
        }
    }


    fn optimize_load_store(&mut self) {
        for block in self.reverse_post_order() {
            let mut compile_time_heap: HashMap<(InsnId, i32), InsnId>  = HashMap::default();
            let old_insns = std::mem::take(&mut self.blocks[block.to_usize()].insns);
            let mut new_insns = Vec::with_capacity(old_insns.len());
            for insn_id in old_insns {
                let replacement_insn: InsnId = match self.resolve(insn_id).insn(self) {
                    &Insn::StoreField { recv, offset, val, .. } => {
                        let key = (self.chase_insn(recv), offset);
                        let heap_entry = compile_time_heap.get(&key).copied();
                        // TODO(Jacob): Switch from actual to partial equality
                        if Some(val) == heap_entry {
                            // If the value is already stored, short circuit and don't add an instruction to the block
                            continue
                        }
                        // TODO(Jacob): Add TBAA to avoid removing so many entries
                        compile_time_heap.retain(|(_, off), _| *off != offset);
                        compile_time_heap.insert(key, val);
                        insn_id
                    },
                    &Insn::LoadField { recv, offset, return_type, .. } => {
                        let key = (self.chase_insn(recv), offset);
                        match compile_time_heap.entry(key) {
                            std::collections::hash_map::Entry::Occupied(entry) => {
                                let cached_insn = *entry.get();

                                // TODO (nirvdrum 2026-06-04): Remove the return type guard and supporting code when the type checker becomes more accurate.
                                // If there's an an embedded<=>heap shape storage transition, it's possible for this `LoadField` to have a different return
                                // type than the cached entry (`CPtr` vs `BasicObject`). While the loaded value would be the same in either case, the
                                // difference in associated type causes type checking to fail. Consequently, we conservatively retain the duplicate `LoadField`.
                                // The `optimize_load_store_does_not_alias_loads_with_incompatible_return_types` test checks the problematic case.
                                let can_forward_cached_insn = match self.resolve(cached_insn).insn(self) {
                                    Insn::LoadField { return_type : cached_return_type,.. } => cached_return_type.is_subtype(return_type),
                                    _ => true
                                };

                                if can_forward_cached_insn {
                                    // If the value is stored already, we should short circuit.
                                    // However, we need to replace insn_id with its representative in the SSA union.
                                    self.make_equal_to(insn_id, cached_insn);
                                    continue
                                }
                            }
                            std::collections::hash_map::Entry::Vacant(_) => {
                                // If the value has not been accessed, cache a copy to optimize future loads or stores.
                                compile_time_heap.insert(key, insn_id);
                            }
                        }
                        insn_id
                    }
                    &Insn::WriteBarrier { .. } => {
                        // Currently, WriteBarrier write effects are Allocator and Memory when we'd really like them to be flags.
                        // We don't use LoadField for mark bits so we can ignore them for now.
                        // But flags does not exist in our effects abstract heap modeling and we don't want to add special casing to effects.
                        // This special casing in this pass here should be removed once we refine our effects system to provide greater granularity for WriteBarrier.
                        // TODO: use TBAA
                        let offset = RUBY_OFFSET_RBASIC_FLAGS;
                        compile_time_heap.retain(|(_, off), _| *off != offset);
                        insn_id
                    },
                    insn => {
                        // If an instruction affects memory and we haven't modeled it, the compile_time_heap is invalidated
                        if insn.effects_of().includes(Effect::write(abstract_heaps::Memory)) {
                            compile_time_heap.clear();
                        }
                        insn_id
                    }
                };
                new_insns.push(replacement_insn);
            }
            self.blocks[block.to_usize()].insns = new_insns;
        }
    }

    /// Fold a binary operator on fixnums.
    fn fold_fixnum_bop(&mut self, insn_id: InsnId, left: InsnId, right: InsnId, f: impl FnOnce(Option<i64>, Option<i64>) -> Option<i64>) -> InsnId {
        f(self.type_of(left).fixnum_value(), self.type_of(right).fixnum_value())
            .filter(|&n| n >= (RUBY_FIXNUM_MIN as i64) && n <= RUBY_FIXNUM_MAX as i64)
            .map(|n| self.new_insn(Insn::Const { val: Const::Value(VALUE::fixnum_from_isize(n as isize)) }))
            .unwrap_or(insn_id)
    }

    /// Fold a binary predicate on fixnums.
    fn fold_fixnum_pred(&mut self, insn_id: InsnId, left: InsnId, right: InsnId, f: impl FnOnce(Option<i64>, Option<i64>) -> Option<bool>) -> InsnId {
        f(self.type_of(left).fixnum_value(), self.type_of(right).fixnum_value())
            .map(|b| if b { Qtrue } else { Qfalse })
            .map(|b| self.new_insn(Insn::Const { val: Const::Value(b) }))
            .unwrap_or(insn_id)
    }

    /// Canonicalize: rewrite each operand through union-find and a map of the most recent `Guard*`
    /// for that value in the dominator tree. Forwards guarded values into branch-edge args (so
    /// `infer_types` narrows merge-block parameters and `fold_constants` drops redundant CFG-join
    /// guards) and ordinary in-block uses.
    ///
    /// `Guard*` substitutions are unconditional for dominated uses: a guard's side-exit semantics
    /// guarantee the substituted value type holds for every dominated use.
    ///
    /// `RefineType` is intentionally skipped: as constructed in HIR build right now, its narrowing
    /// is only valid on one branch arm, which would require dropping refine-derived rewrites at
    /// each `IfTrue`/`IfFalse`. Cross-arm refine forwarding is left for a follow-up
    /// dominator-scoped pass.
    ///
    /// Inspired by Cranelift's aegraph canonicalize step
    /// (<https://cfallin.org/blog/2026/04/09/aegraph/>).
    fn canonicalize(&mut self) {
        // TODO(max): Don't make so many maps. Instead, use either undo-redo or dominator numbering
        // information for dominator tree.
        let mut rewrite_maps: Vec<Option<HashMap<InsnId, InsnId>>> = vec![None; self.blocks.len()];
        let dominators = Dominators::new(self);
        for &block in dominators.cfi.reverse_post_order() {
            let mut rewrite_map = rewrite_maps[dominators.idom(block).to_usize()].clone().unwrap_or_else(|| HashMap::default());
            for i in 0..self.blocks[block.to_usize()].insns.len() {
                let insn_id = self.blocks[block.to_usize()].insns[i];
                let canonical_id = self.union_find.find_const(insn_id);

                let union_find = &self.union_find;
                self.insns[canonical_id.to_usize()].for_each_operand_mut(|operand| {
                    let canon = union_find.find_const(*operand);
                    *operand = rewrite_map.get(&canon).copied().unwrap_or(canon);
                });

                // For the binary guards only `left` is registered because their infer_type is
                // type_of(left).
                match &self.insns[canonical_id.to_usize()] {
                    Insn::GuardType      { val:  src, .. }
                    | Insn::GuardBitEquals { val:  src, .. }
                    | Insn::GuardAnyBitSet { val:  src, .. }
                    | Insn::GuardNoBitsSet { val:  src, .. }
                    | Insn::GuardNotRuby2KeywordsHash { val: src, .. }
                    | Insn::GuardGreaterEq { left: src, .. }
                    | Insn::GuardLess      { left: src, .. } => {
                        rewrite_map.insert(*src, canonical_id);
                    }
                    _ => {}
                }
            }
            rewrite_maps[block.to_usize()] = Some(rewrite_map);
        }

        crate::stats::trace_compile_phase("infer_types", || self.infer_types());
    }

    /// Use type information left by `infer_types` to fold away operations that can be evaluated at compile-time.
    ///
    /// It can fold fixnum math, truthiness tests, and branches with constant conditionals.
    fn fold_constants(&mut self) {
        fn is_power_of_two(d: i64) -> bool {
            d > 0 && (d & (d - 1)) == 0
        }
        // TODO(max): Determine if it's worth it for us to reflow types after each branch
        // simplification. This means that we can have nice cascading optimizations if what used to
        // be a union of two different basic block arguments now has a single value.
        //
        // This would require 1) fixpointing, 2) worklist, or 3) (slightly less powerful) calling a
        // function-level infer_types after each pruned branch.
        for block in self.reverse_post_order() {
            let old_insns = std::mem::take(&mut self.blocks[block.to_usize()].insns);
            let mut new_insns = Vec::with_capacity(old_insns.len());
            for insn_id in old_insns {
                let replacement_id = match self.resolve(insn_id).insn(self) {
                    &Insn::GuardType { val, guard_type, .. } if self.is_a(val, guard_type) => {
                        self.make_equal_to(insn_id, val);
                        // Don't bother re-inferring the type of val; we already know it.
                        continue;
                    }
                    &Insn::RefineType { val, new_type, .. } if self.is_a(val, new_type) => {
                        self.make_equal_to(insn_id, val);
                        // Don't bother re-inferring the type of val; we already know it.
                        continue;
                    }
                    &Insn::LoadField { recv, offset, return_type, .. } if return_type.is_subtype(types::BasicObject) &&
                            u32::try_from(offset).is_ok() => {
                        let offset = (offset as u32).to_usize();
                        let recv_type = self.type_of(recv);
                        match recv_type.ruby_object() {
                            Some(recv_obj) if recv_obj.is_frozen() => {
                                let recv_ptr = recv_obj.as_ptr() as *const VALUE;
                                let val = unsafe { recv_ptr.byte_add(offset).read() };
                                self.new_insn(Insn::Const { val: Const::Value(val) })
                            }
                            _ => insn_id,
                        }
                    }
                    &Insn::LoadField { recv, offset, return_type, .. } if return_type.is_subtype(types::CShape) &&
                            u32::try_from(offset).is_ok() => {
                        let offset = (offset as u32).to_usize();
                        let recv_type = self.type_of(recv);
                        match recv_type.ruby_object() {
                            Some(recv_obj) if recv_obj.is_frozen() => {
                                let recv_ptr = recv_obj.as_ptr() as *const u32;
                                let val = unsafe { recv_ptr.byte_add(offset).read() };
                                self.new_insn(Insn::Const { val: Const::CShape(ShapeId(val)) })
                            }
                            _ => insn_id,
                        }
                    }
                    &Insn::ArrayLength { array } => {
                        match self.type_of(array).ruby_object() {
                            Some(array_obj) if array_obj.is_frozen() => {
                                let length = unsafe { rb_jit_array_len(array_obj) };
                                self.new_insn(Insn::Const { val: Const::CInt64(length) })
                            }
                            _ => insn_id,
                        }
                    }
                    &Insn::UnboxFixnum { val } => {
                        let recv_type = self.type_of(val);
                        match recv_type.fixnum_value() {
                            Some(val) => self.new_insn(Insn::Const { val: Const::CInt64(val) }),
                            _ => insn_id,
                        }
                    },
                    &Insn::StringCoderangeOrScan { cached, .. } => {
                        // A known coderange other than UNKNOWN needs no scan.
                        match self.type_of(cached).cint64_value() {
                            Some(coderange) if coderange != RUBY_ENC_CODERANGE_UNKNOWN.into() => {
                                self.make_equal_to(insn_id, cached);
                                continue;
                            }
                            _ => insn_id,
                        }
                    },
                    &Insn::ArrayAsetOrStore { array, index, length, val, .. } => {
                        match (self.type_of(index).cint64_value(), self.type_of(length).cint64_value()) {
                            // Statically in range and nonnegative: the store can't grow the array,
                            // so drop the bounds check and the rb_ary_store fallback.
                            (Some(index_num), Some(length_num)) if index_num >= 0 && index_num < length_num =>
                                self.new_insn(Insn::ArrayAset { array, index, val }),
                            _ => insn_id,
                        }
                    },
                    &Insn::GuardGreaterEq { left, right, state, ref reason, recompile } => {
                        let left_num = self.type_of(left).cint64_value();
                        let right_num = self.type_of(right).cint64_value();
                        match (left_num, right_num) {
                            (Some(l), Some(r)) if l >= r => {
                                self.make_equal_to(insn_id, left);
                                continue
                            },
                            (Some(_), Some(_)) => self.new_insn(Insn::SideExit { state, reason: reason.clone(), recompile }),
                            _ => insn_id,
                        }
                    },
                    &Insn::GuardLess { left, right, state, ref reason } => {
                        let left_num = self.type_of(left).cint64_value();
                        let right_num = self.type_of(right).cint64_value();
                        match (left_num, right_num) {
                            (Some(l), Some(r)) if l < r => {
                                self.make_equal_to(insn_id, left);
                                continue
                            },
                            (Some(_), Some(_)) => self.new_insn(Insn::SideExit { state, reason: reason.clone(), recompile: None }),
                            _ => insn_id,
                        }
                    },
                    &Insn::GuardBitEquals { val, expected, .. } => {
                        let recv_type = self.type_of(val);
                        if recv_type.has_value(expected) {
                            self.make_equal_to(insn_id, val);
                            continue;
                        } else {
                            insn_id
                        }
                    }
                    &Insn::IsA { val, class } => 'is_a: {
                        let class_type = self.type_of(class);
                        if !class_type.is_subtype(types::Class) {
                            break 'is_a insn_id;
                        }
                        let Some(class_value) = class_type.ruby_object() else {
                            break 'is_a insn_id;
                        };
                        let val_type = self.type_of(val);
                        let the_class = Type::from_class_inexact(class_value);
                        if val_type.is_subtype(the_class) {
                            self.new_insn(Insn::Const { val: Const::Value(Qtrue) })
                        } else if !val_type.could_be(the_class) {
                            self.new_insn(Insn::Const { val: Const::Value(Qfalse) })
                        } else {
                            insn_id
                        }
                    }
                    &Insn::StringEqual { left, right } => {
                        let left = self.chase_insn(left);
                        let right = self.chase_insn(right);
                        // If both operands resolve to the same SSA value,
                        // String#== is guaranteed to be true.
                        if left == right {
                            self.new_insn(Insn::Const { val: Const::Value(Qtrue) })
                        } else {
                            let left_type = self.type_of(left);
                            let right_type = self.type_of(right);
                            match (left_type.ruby_object(), right_type.ruby_object()) {
                                (Some(left_obj), Some(right_obj))
                                    if left_obj.is_frozen() && right_obj.is_frozen() =>
                                {
                                    // For known frozen objects, evaluate String#== at compile time.
                                    let val = unsafe { rb_yarv_str_eql_internal(left_obj, right_obj) };
                                    self.new_insn(Insn::Const { val: Const::Value(val) })
                                }
                                _ => insn_id,
                            }
                        }
                    }
                    &Insn::FixnumAdd { left, right, .. } => {
                        match (self.type_of(left).fixnum_value(), self.type_of(right).fixnum_value()) {
                            (Some(0), _) => { self.make_equal_to(insn_id, right); continue; }
                            (_, Some(0)) => { self.make_equal_to(insn_id, left); continue; }
                            _ => {}
                        }
                        self.fold_fixnum_bop(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => l.checked_add(r),
                            _ => None,
                        })
                    }
                    &Insn::FixnumSub { left, right, .. } => {
                        match (self.type_of(left).fixnum_value(), self.type_of(right).fixnum_value()) {
                            (_, Some(0)) => { self.make_equal_to(insn_id, left); continue; }
                            _ => {}
                        }
                        self.fold_fixnum_bop(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => l.checked_sub(r),
                            _ => None,
                        })
                    }
                    &Insn::FixnumMult { left, right, .. } => {
                        match (self.type_of(left).fixnum_value(), self.type_of(right).fixnum_value()) {
                            (Some(1), _) => { self.make_equal_to(insn_id, right); continue; }
                            (_, Some(1)) => { self.make_equal_to(insn_id, left); continue; }
                            _ => {}
                        }
                        self.fold_fixnum_bop(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => l.checked_mul(r),
                            (Some(0), _) | (_, Some(0)) => Some(0),
                            _ => None,
                        })
                    }
                    &Insn::FixnumDiv { left, right, .. } => {
                        match (self.type_of(left).fixnum_value(), self.type_of(right).fixnum_value()) {
                            (_, Some(1)) => { self.make_equal_to(insn_id, left); continue; }
                            // Strength-reduce division by a power of two to an arithmetic right
                            // shift. Both Ruby's Integer#/ and a sign-extending shift round the
                            // quotient towards negative infinity, so this holds for all fixnums.
                            (None, Some(d)) if is_power_of_two(d) => {
                                let shift = self.new_insn(Insn::Const { val: Const::Value(VALUE::fixnum_from_isize(d.trailing_zeros() as isize)) });
                                self.insn_types[shift.to_usize()] = self.infer_type(shift);
                                new_insns.push(shift);
                                let replacement = self.new_insn(Insn::FixnumRShift { left, right: shift });
                                self.make_equal_to(insn_id, replacement);
                                self.insn_types[replacement.to_usize()] = self.infer_type(replacement);
                                new_insns.push(replacement);
                                continue;
                            }
                            _ => {}
                        }
                        self.fold_fixnum_bop(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) if l == (RUBY_FIXNUM_MIN as i64) && r == -1 => None, // Avoid Fixnum overflow
                            (Some(_l), Some(r)) if r == 0 => None, // Avoid Divide by zero.
                            (Some(l), Some(r)) => {
                                let l_obj = VALUE::fixnum_from_isize(l as isize);
                                let r_obj = VALUE::fixnum_from_isize(r as isize);
                                Some(unsafe { rb_jit_fix_div_fix(l_obj, r_obj) }.as_fixnum())
                            },
                            _ => None,
                        })
                    }
                    &Insn::FixnumMod { left, right, .. } => {
                        match (self.type_of(left).fixnum_value(), self.type_of(right).fixnum_value()) {
                            // Strength-reduce modulo by a power of two to a bitwise AND. The sign
                            // of Ruby's Integer#% follows the (positive) divisor, so the result is
                            // in [0, d), which matches two's complement AND for all fixnums.
                            (None, Some(d)) if is_power_of_two(d) => {
                                let mask = self.new_insn(Insn::Const { val: Const::Value(VALUE::fixnum_from_isize((d - 1) as isize)) });
                                self.insn_types[mask.to_usize()] = self.infer_type(mask);
                                new_insns.push(mask);
                                let replacement = self.new_insn(Insn::FixnumAnd { left, right: mask });
                                self.make_equal_to(insn_id, replacement);
                                self.insn_types[replacement.to_usize()] = self.infer_type(replacement);
                                new_insns.push(replacement);
                                continue;
                            }
                            _ => {}
                        }
                        self.fold_fixnum_bop(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) if r != 0 => {
                                let l_obj = VALUE::fixnum_from_isize(l as isize);
                                let r_obj = VALUE::fixnum_from_isize(r as isize);
                                Some(unsafe { rb_jit_fix_mod_fix(l_obj, r_obj) }.as_fixnum())
                            },
                            _ => None,
                        })
                    }
                    &Insn::FixnumXor { left, right, .. } => {
                        self.fold_fixnum_bop(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => Some(l ^ r),
                            _ => None,
                        })
                    }
                    &Insn::FixnumAnd { left, right, .. } => {
                        self.fold_fixnum_bop(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => Some(l & r),
                            _ => None,
                        })
                    }
                    &Insn::FixnumOr { left, right, .. } => {
                        self.fold_fixnum_bop(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => Some(l | r),
                            _ => None,
                        })
                    }
                    &Insn::FixnumEq { left, right, .. } => {
                        self.fold_fixnum_pred(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => Some(l == r),
                            _ => None,
                        })
                    }
                    &Insn::FixnumNeq { left, right, .. } => {
                        self.fold_fixnum_pred(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => Some(l != r),
                            _ => None,
                        })
                    }
                    &Insn::FixnumLt { left, right, .. } => {
                        self.fold_fixnum_pred(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => Some(l < r),
                            _ => None,
                        })
                    }
                    &Insn::FixnumLe { left, right, .. } => {
                        self.fold_fixnum_pred(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => Some(l <= r),
                            _ => None,
                        })
                    }
                    &Insn::FixnumGt { left, right, .. } => {
                        self.fold_fixnum_pred(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => Some(l > r),
                            _ => None,
                        })
                    }
                    &Insn::FixnumGe { left, right, .. } => {
                        self.fold_fixnum_pred(insn_id, left, right, |l, r| match (l, r) {
                            (Some(l), Some(r)) => Some(l >= r),
                            _ => None,
                        })
                    }
                    &Insn::ArrayArefOrNil { array, index, .. }
                        if self.type_of(array).ruby_object_known()
                            && self.type_of(index).is_subtype(types::CInt64) => {
                        let array_obj = self.type_of(array).ruby_object().unwrap();
                        match (array_obj.is_frozen(), self.type_of(index).cint64_value()) {
                            (true, Some(index)) => {
                                // rb_yarv_ary_entry_internal returns nil out of bounds, which is
                                // exactly what ArrayArefOrNil does.
                                let val = unsafe { rb_yarv_ary_entry_internal(array_obj, index) };
                                self.new_insn(Insn::Const { val: Const::Value(val) })
                            }
                            _ => insn_id,
                        }
                    }
                    &Insn::ArrayAref { array, index }
                        if self.type_of(array).ruby_object_known()
                            && self.type_of(index).is_subtype(types::CInt64) => {
                        let array_obj = self.type_of(array).ruby_object().unwrap();
                        match (array_obj.is_frozen(), self.type_of(index).cint64_value()) {
                            (true, Some(index)) => {
                                let val = unsafe { rb_yarv_ary_entry_internal(array_obj, index) };
                                self.new_insn(Insn::Const { val: Const::Value(val) })
                            }
                            _ => insn_id,
                        }
                    }
                    &Insn::AdjustBounds { index, .. } => {
                        // If index is known nonnegative, then we don't need to adjust bounds.
                        if self.type_of(index).known_nonnegative() {
                            self.make_equal_to(insn_id, index);
                            // Don't bother re-inferring the type of index; we already know it.
                            continue;
                        } else {
                            insn_id
                        }
                    }
                    &Insn::Test { val } if self.type_of(val).is_known_falsy() => {
                        self.new_insn(Insn::Const { val: Const::CBool(false) })
                    }
                    &Insn::Test { val } if self.type_of(val).is_known_truthy() => {
                        self.new_insn(Insn::Const { val: Const::CBool(true) })
                    }
                    &Insn::Test { val: test_val } => {
                        if let &Insn::BoxBool { val: bool_val } = self.resolve(test_val).insn(self) {
                            self.make_equal_to(insn_id, bool_val);
                            continue;
                        } else {
                            insn_id
                        }
                    }
                    &Insn::CondBranch { val, ref if_true, .. } if self.is_a(val, Type::from_cbool(true)) => {
                        self.new_insn(Insn::Jump(if_true.clone()))
                    }
                    &Insn::CondBranch { val, ref if_false, .. } if self.is_a(val, Type::from_cbool(false)) => {
                        self.new_insn(Insn::Jump(if_false.clone()))
                    }
                    _ => insn_id,
                };
                // If we're adding a new instruction, mark the two equivalent in the union-find and
                // do an incremental flow typing of the new instruction.
                if insn_id != replacement_id && self.insns[replacement_id.to_usize()].has_output() {
                    self.make_equal_to(insn_id, replacement_id);
                    self.insn_types[replacement_id.to_usize()] = self.infer_type(replacement_id);
                }
                new_insns.push(replacement_id);
                // If we've just folded an IfTrue into a Jump, for example, don't bother copying
                // over unreachable instructions afterward.
                if self.insns[replacement_id.to_usize()].is_terminator() {
                    break;
                }
            }
            self.blocks[block.to_usize()].insns = new_insns;
        }
    }

    /// Remove instructions that do not have side effects and are not referenced by any other
    /// instruction.
    fn eliminate_dead_code(&mut self) {
        let rpo = self.reverse_post_order();
        let mut worklist = VecDeque::new();
        // Find all of the instructions that have side effects, are control instructions, or are
        // otherwise necessary to keep around
        for block_id in &rpo {
            for insn_id in &self.blocks[block_id.to_usize()].insns {
                if !&self.insns[insn_id.to_usize()].is_elidable() {
                    worklist.push_back(*insn_id);
                }
            }
        }
        let mut necessary = InsnSet::with_capacity(self.insns.len());
        // Now recursively traverse their data dependencies and mark those as necessary
        while let Some(insn_id) = worklist.pop_front() {
            if necessary.get(insn_id) { continue; }
            necessary.insert(insn_id);
            let insn_id = self.union_find.find_const(insn_id);
            self.insns[insn_id.to_usize()].for_each_operand(|operand| {
                worklist.push_back(self.union_find.find_const(operand));
            });
        }
        // Now remove all unnecessary instructions
        for block_id in &rpo {
            self.blocks[block_id.to_usize()].insns.retain(|&insn_id| necessary.get(insn_id));
        }
    }

    fn absorb_dst_block(&mut self, num_in_edges: &[u32], block: BlockId) -> bool {
        let Some(&terminator_id) = self.blocks[block.to_usize()].insns.last()
            else { return false };
        let &mut Insn::Jump(ref mut edge) = self.resolve(terminator_id).insn_mut(self)
            else { return false };
        if edge.target == block {
            // Can't absorb self
            return false;
        }
        if num_in_edges[edge.target.to_usize()] != 1 {
            // Can't absorb block if it's the target of more than one branch
            return false;
        }
        // Link up params with block args
        let args = std::mem::take(&mut edge.args);
        let target = edge.target;
        // Drop the borrow of edge, which drops the borrow of self, which allows us to mutate self
        // again.
        let _ = edge;
        let params = std::mem::take(&mut self.blocks[target.to_usize()].params);
        assert_eq!(args.len(), params.len());
        for (arg, param) in args.iter().zip(params) {
            self.make_equal_to(param, *arg);
        }
        // Remove branch instruction
        self.blocks[block.to_usize()].insns.pop();
        // Move target instructions into block
        let target_insns = std::mem::take(&mut self.blocks[target.to_usize()].insns);
        self.blocks[block.to_usize()].insns.extend(target_insns);
        true
    }

    /// Replace block parameters that can only ever hold one value with that constant, and
    /// drop the parameter from the block and from every branch that targets it.
    fn reduce_block_params(&mut self) {
        // Parameter indices to drop for each block.
        let mut dropped: Vec<Vec<usize>> = vec![vec![]; self.blocks.len()];
        for block in self.reverse_post_order() {
            // Entry block parameters are the calling convention, not phis: codegen maps them
            // to argument registers and `copy_param_types` types them from the ISEQ.
            if block == self.entries_block || self.is_entry_block(block) {
                continue;
            }
            let params = self.blocks[block.to_usize()].params.clone();
            let mut consts = vec![];
            for (idx, &param) in params.iter().enumerate() {
                let Some(val) = self.type_of(param).exact_ruby_value() else {
                    continue
                };
                dropped[block.to_usize()].push(idx);
                consts.push((param, val));
            }
            // Insert the constants at the top of the block so that they dominate every
            // use of the parameter they replace.
            let mut materialized: HashMap<VALUE, InsnId> = HashMap::default();
            let mut prologue = vec![];
            for (param, val) in consts {
                let replacement = *materialized.entry(val).or_insert_with(|| {
                    let replacement = self.new_insn(Insn::Const { val: Const::Value(val) });
                    self.insn_types[replacement.to_usize()] = self.infer_type(replacement);
                    prologue.push(replacement);
                    replacement
                });
                self.make_equal_to(param, replacement);
            }
            self.blocks[block.to_usize()].insns.splice(0..0, prologue);
            // Keep the surviving parameters in order.
            let drop_set: HashSet<usize> = dropped[block.to_usize()].iter().copied().collect();
            self.blocks[block.to_usize()].params = params.into_iter().enumerate()
                .filter(|(idx, _)| !drop_set.contains(idx))
                .map(|(_, param)| param)
                .collect();
        }

        // If there's nothing to drop, finish the pass early.
        if dropped.iter().all(|indices| indices.is_empty()) {
            return;
        }

        // Drop the matching arguments from every branch, so each edge keeps the arity of
        // its target block.
        let retain_args = |edge: &mut BranchEdge, dropped: &[Vec<usize>]| {
            let drop_set = &dropped[edge.target.to_usize()];
            if drop_set.is_empty() {
                return;
            }
            let mut arg_idx = 0;
            edge.args.retain(|_| {
                let keep = !drop_set.contains(&arg_idx);
                arg_idx += 1;
                keep
            });
        };
        for block in self.reverse_post_order() {
            let Some(&terminator_id) = self.blocks[block.to_usize()].insns.last() else {
                continue
            };
            match &mut self.insns[terminator_id.to_usize()] {
                Insn::Jump(edge) => retain_args(edge, &dropped),
                Insn::CondBranch { if_true, if_false, .. } => {
                    retain_args(if_true, &dropped);
                    retain_args(if_false, &dropped);
                }
                _ => {}
            }
        }
    }

    /// Clean up linked lists of blocks A -> B -> C into A (with B's and C's instructions).
    fn clean_cfg(&mut self) {
        // num_in_edges is invariant throughout cleaning the CFG:
        // * we don't allocate new blocks
        // * blocks that get absorbed are not in RPO anymore
        // * blocks pointed to by blocks that get absorbed retain the same number of in-edges
        let mut num_in_edges = vec![0; self.blocks.len()];
        for block in self.reverse_post_order() {
            for target in self.successors(block) {
                num_in_edges[target.to_usize()] += 1;
            }
        }
        let mut changed = false;
        loop {
            let mut iter_changed = false;
            for block in self.reverse_post_order() {
                // Ignore transient empty blocks
                if self.blocks[block.to_usize()].insns.is_empty() { continue; }
                loop {
                    let absorbed = self.absorb_dst_block(&num_in_edges, block);
                    if !absorbed { break; }
                    iter_changed = true;
                }
            }
            if !iter_changed { break; }
            changed = true;
        }
        if changed {
            crate::stats::trace_compile_phase("infer_types", || self.infer_types());
        }
    }

    /// Remove duplicate PatchPoint instructions within each basic block.
    /// Two PatchPoints are redundant if they assert the same Invariant and no
    /// intervening instruction could invalidate it (i.e., writes to PatchPoint).
    fn remove_redundant_patch_points(&mut self) {
        for block_id in self.reverse_post_order() {
            let mut seen = HashSet::default();
            let insns = std::mem::take(&mut self.blocks[block_id.to_usize()].insns);
            let mut new_insns = Vec::with_capacity(insns.len());
            for insn_id in insns {
                // PatchPoint is never in union-find and it does not have operands, so fake a
                // ResolvedInsnId.
                let insn = ResolvedInsnId(insn_id).insn(self);
                if let Insn::PatchPoint { invariant, .. } = insn {
                    if !seen.insert(invariant) {
                        continue;
                    }
                } else if insn.effects_of().write_bits().overlaps(abstract_heaps::PatchPoint) {
                    seen.clear();
                }
                new_insns.push(insn_id);
            }
            self.blocks[block_id.to_usize()].insns = new_insns;
        }
    }

    /// Remove duplicate CheckInterrupts instructions within each basic block.
    /// Only the first CheckInterrupts in a block is needed unless an intervening
    /// instruction writes to InterruptFlag (e.g. a call), which resets tracking.
    fn remove_duplicate_check_interrupts(&mut self) {
        for block_id in self.reverse_post_order() {
            let mut seen = false;
            let insns = std::mem::take(&mut self.blocks[block_id.to_usize()].insns);
            let mut new_insns = Vec::with_capacity(insns.len());
            for insn_id in insns {
                let insn = &self.insns[insn_id.to_usize()];
                if matches!(insn, Insn::CheckInterrupts { .. }) {
                    if seen { continue; }
                    seen = true;
                } else if insn.effects_of().write_bits().overlaps(abstract_heaps::InterruptFlag) {
                    seen = false;
                }
                new_insns.push(insn_id);
            }
            self.blocks[block_id.to_usize()].insns = new_insns;
        }
    }

    /// Whether `insn` may stay between a PushInlineFrame/PopInlineFrame pair
    /// that gets elided, i.e. whether it can neither take a side exit nor
    /// observe the frame:
    /// * Its effects must be confined to the Stats heap, so it doesn't read
    ///   or write anything observable (in particular Frame and Control).
    ///   Stats counters only bump a global counter, so allowing them keeps
    ///   this pass enabled when --zjit-stats inserts IncrCounter between
    ///   every PushInlineFrame/PopInlineFrame.
    /// * It must not reference a FrameState `Snapshot` operand: a side exit
    ///   materializes the enclosing inlined frame, and effects don't model
    ///   deopt for otherwise pure instructions like `FixnumAdd`.
    /// * `LoadSP` reads the frame-dependent SP register despite having empty
    ///   effects, so it's excluded explicitly.
    fn can_elide_enclosing_frame(&self, insn: &Insn) -> bool {
        // TODO: Model LoadSP as reading from the control frame and drop this
        // special case.
        if matches!(insn, Insn::LoadSP) {
            return false;
        }
        // Snapshot is metadata that generates no code, so it never observes the
        // frame by itself. Whether a side exit can materialize the frame from it
        // is determined by the instructions that reference it, which this scan
        // checks individually; without this early return, the operand scan below
        // would reject every Snapshot whose `caller` chains to another Snapshot.
        if matches!(insn, Insn::Snapshot { .. }) {
            return true;
        }
        let effects = insn.effects_of();
        if !(abstract_heaps::Stats.includes(effects.read_bits()) && abstract_heaps::Stats.includes(effects.write_bits())) {
            return false;
        }
        // TODO: Model the possibility of taking a side exit as a subeffect of
        // Control so that the effect check above subsumes this operand scan.
        let mut references_snapshot = false;
        insn.for_each_operand(|opnd| {
            let opnd = self.union_find.find_const(opnd);
            if matches!(&self.insns[opnd.to_usize()], Insn::Snapshot { .. }) {
                references_snapshot = true;
            }
        });
        !references_snapshot
    }

    /// Remove PushInlineFrame/PopInlineFrame pairs whose inlined body has been
    /// optimized away entirely.
    ///
    /// When an inlined callee folds to a constant (e.g. a method body guarded by a
    /// constant that is false), earlier passes can leave a PushInlineFrame that is
    /// immediately followed by its matching PopInlineFrame, with only instructions
    /// that neither take a side exit nor observe the frame in between (see
    /// [`Function::can_elide_enclosing_frame`]). Such a frame is unobservable:
    /// nothing between the push and the pop can take a side exit, allocate, raise,
    /// or
    /// walk the frame chain, so pushing and popping the frame is pure overhead.
    /// This pass deletes both instructions of each such pair.
    ///
    /// CheckInterrupts gets special treatment: it can take a side exit, but it's
    /// also removed if it's the only such instruction between the push and the
    /// pop. Such a CheckInterrupts is the one emitted for the inlined callee's
    /// `leave`, and since the callee's body does nothing else, removing the check
    /// only delays interrupt delivery until the next CheckInterrupts the caller
    /// (or its caller) runs. The delay is bounded: CheckInterrupts on loop
    /// back-edges are never removed here because a loop body spans multiple
    /// blocks while this pass only matches pairs within a single block, so any
    /// cycle in execution still checks interrupts.
    ///
    /// NoTracePoint PatchPoints get the same treatment. Enabling a TracePoint
    /// invalidates all compiled code wholesale (see
    /// rb_zjit_tracing_invalidate_all); the patch points only make execution
    /// that is already inside compiled code side-exit as soon as possible.
    /// Removing the callee's NoTracePoint along with an otherwise-empty pair
    /// just delays that in-flight side exit until the next patch point the
    /// caller runs, and since the callee's body does nothing, no events would
    /// have fired from it anyway.
    ///
    /// A pair whose body is a single leaf InvokeBuiltin (e.g. an inlined
    /// opt_invokebuiltin_delegate_leave method) is also elided: a leaf builtin
    /// doesn't raise, call Ruby code, or otherwise observe the frame, so it can
    /// run against the caller's frame. Its FrameState is rewritten from the
    /// callee's to the PushInlineFrame's caller-side state, whose PC the
    /// builtin's code will store for GC. This reproduces how such methods used
    /// to call the builtin function without a frame push before inlining.
    ///
    /// Pairs are matched per basic block with a stack so that nested pairs are
    /// handled: an inner elided pair doesn't prevent the outer pair from being
    /// elided, while an inner pair that must be kept also keeps the outer one
    /// (the kept push/pop observe and modify the frame chain). A PushInlineFrame
    /// whose PopInlineFrame lives in another block is left untouched.
    fn eliminate_empty_inline_frames(&mut self) {
        /// A PushInlineFrame whose matching PopInlineFrame hasn't been seen yet.
        struct PendingPush {
            /// The PushInlineFrame's instruction ID.
            push_id: InsnId,
            /// Whether an instruction that may take a side exit or observe the
            /// frame has been seen since the push. If so, the pair must be kept.
            frame_observed: bool,
            /// CheckInterrupts and NoTracePoint PatchPoint instructions seen
            /// since the push. They don't set `frame_observed` on their own; if
            /// the pair is otherwise empty, they are removed together with the
            /// pair.
            removable_insns: Vec<InsnId>,
            /// A leaf InvokeBuiltin seen since the push. It doesn't set
            /// `frame_observed` on its own; if the pair is otherwise empty, the
            /// pair is elided and the InvokeBuiltin's FrameState is rewritten to
            /// the PushInlineFrame's caller-side state.
            leaf_builtin: Option<InsnId>,
        }

        for block_id in self.reverse_post_order() {
            // First, find the (PushInlineFrame, PopInlineFrame) pairs to elide.
            let mut elided_pairs: Vec<(InsnId, InsnId)> = Vec::new();
            let mut elided_removable_insns: Vec<InsnId> = Vec::new();
            // Leaf InvokeBuiltins in elided pairs, with their pair's PushInlineFrame.
            let mut rebased_leaf_builtins: Vec<(InsnId, InsnId)> = Vec::new();
            let mut pending_pushes: Vec<PendingPush> = Vec::new();
            for &insn_id in &self.blocks[block_id.to_usize()].insns {
                match self.find_ref(insn_id) {
                    Insn::PushInlineFrame { .. } => {
                        pending_pushes.push(PendingPush { push_id: insn_id, frame_observed: false, removable_insns: Vec::new(), leaf_builtin: None });
                    }
                    Insn::CheckInterrupts { .. } | Insn::PatchPoint { invariant: Invariant::NoTracePoint, .. } if !pending_pushes.is_empty() => {
                        pending_pushes.last_mut().unwrap().removable_insns.push(insn_id);
                    }
                    Insn::InvokeBuiltin { leaf: true, .. } if pending_pushes.last().is_some_and(|push| push.leaf_builtin.is_none()) => {
                        pending_pushes.last_mut().unwrap().leaf_builtin = Some(insn_id);
                    }
                    Insn::PopInlineFrame { .. } => {
                        match pending_pushes.pop() {
                            Some(PendingPush { push_id, frame_observed: false, removable_insns, leaf_builtin }) => {
                                // Empty pair: elide both the push and this pop, along
                                // with the callee's CheckInterrupts and NoTracePoint
                                // PatchPoints (if any).
                                elided_pairs.push((push_id, insn_id));
                                elided_removable_insns.extend(removable_insns);
                                if let Some(builtin_id) = leaf_builtin {
                                    rebased_leaf_builtins.push((builtin_id, push_id));
                                    // The InvokeBuiltin stays behind with a FrameState
                                    // that assumes the enclosing frame (if any) exists
                                    // at run-time, so the enclosing pair must be kept.
                                    if let Some(outer) = pending_pushes.last_mut() {
                                        outer.frame_observed = true;
                                    }
                                }
                            }
                            Some(PendingPush { frame_observed: true, .. }) => {
                                // Keep the pair. It observes the frame chain, so the
                                // enclosing pair (if any) must be kept too.
                                if let Some(outer) = pending_pushes.last_mut() {
                                    outer.frame_observed = true;
                                }
                            }
                            None => {
                                // The matching push is in another block; leave it alone.
                            }
                        }
                    }
                    insn => {
                        if !pending_pushes.is_empty() && !self.can_elide_enclosing_frame(insn) {
                            // The instruction may take a side exit or observe the frame,
                            // so the innermost pending pair must be kept. Enclosing pairs
                            // don't need to be marked here: a kept pair marks its
                            // enclosing pair when its PopInlineFrame is reached, so the
                            // flag propagates outward one pop at a time.
                            pending_pushes.last_mut().unwrap().frame_observed = true;
                        }
                    }
                }
            }
            if elided_pairs.is_empty() {
                continue;
            }

            // Elide each pair: drop the pop, and drop the push as well, except
            // that with --zjit-stats it's replaced with a counter of how many
            // times execution passes an elided pair at run-time.
            let mut rewrites: HashMap<InsnId, Option<InsnId>> = HashMap::default();
            for (push_id, pop_id) in elided_pairs {
                let replacement = get_option!(stats)
                    .then(|| self.new_insn(Insn::IncrCounter(Counter::empty_inline_frame_count)));
                rewrites.insert(push_id, replacement);
                rewrites.insert(pop_id, None);
            }
            for removable_insn_id in elided_removable_insns {
                rewrites.insert(removable_insn_id, None);
            }
            // Rewrite each elided pair's leaf InvokeBuiltin to use the PushInlineFrame's
            // call-site FrameState instead of the callee's. This is `guard_state`, not
            // `state`: the builtin now runs without the callee frame, so a side exit from
            // it re-runs the original call instruction and needs the stack the interpreter
            // had there, which `state` may have rewritten for the frame setup.
            for (builtin_id, push_id) in rebased_leaf_builtins {
                let &Insn::PushInlineFrame { guard_state: push_state, .. } = self.find_ref(push_id) else {
                    panic!("Expected PushInlineFrame instruction");
                };
                let &Insn::InvokeBuiltin { bf, recv, ref args, leaf, return_type, .. } = self.find_ref(builtin_id) else {
                    panic!("Expected InvokeBuiltin instruction");
                };
                let args = args.clone();
                let builtin_type = self.type_of(builtin_id);
                let replacement = self.new_insn(Insn::InvokeBuiltin { bf, recv, args, state: push_state, leaf, return_type });
                self.insn_types[replacement.to_usize()] = builtin_type;
                self.make_equal_to(builtin_id, replacement);
                rewrites.insert(builtin_id, Some(replacement));
            }
            self.blocks[block_id.to_usize()].insns.retain_mut(|insn_id| {
                match rewrites.get(insn_id) {
                    Some(Some(replacement)) => {
                        *insn_id = *replacement;
                        true
                    }
                    Some(None) => false,
                    None => true,
                }
            });
        }
    }

    /// Return a list that has entry_block and then jit_entry_blocks
    fn entry_blocks(&self) -> Vec<BlockId> {
        let mut entry_blocks = self.jit_entry_blocks.clone();
        entry_blocks.insert(0, self.entry_block);
        entry_blocks
    }

    pub fn is_entry_block(&self, block_id: BlockId) -> bool {
        self.entry_block == block_id || self.jit_entry_blocks.contains(&block_id)
    }

    /// Populate the entries superblock with an Entries instruction targeting all entry blocks.
    /// Must be called after all entry blocks have been created.
    fn seal_entries(&mut self) {
        let targets = self.entry_blocks();
        self.push_insn(self.entries_block, Insn::Entries { targets });
    }

    /// Return a traversal of the `Function`'s `BlockId`s in reverse post-order.
    pub fn reverse_post_order(&self) -> Vec<BlockId> {
        let mut result = self.post_order_from(self.entries_block);
        result.reverse();
        result
    }

    fn post_order_from(&self, start: BlockId) -> Vec<BlockId> {
        #[derive(PartialEq)]
        enum Action {
            VisitEdges,
            VisitSelf,
        }
        // Both vectors are bounded by the block count, and every pass in the pipeline
        // asks for the reverse post order, so growing them by doublings was a
        // measurable share of the allocator traffic all by itself.
        let num_blocks = self.blocks.len();
        let mut result = Vec::with_capacity(num_blocks);
        let mut seen = BlockSet::with_capacity(num_blocks);
        let mut stack = Vec::with_capacity(num_blocks * 2);
        stack.push((start, Action::VisitEdges));
        while let Some((block, action)) = stack.pop() {
            if action == Action::VisitSelf {
                result.push(block);
                continue;
            }
            if !seen.insert(block) { continue; }
            stack.push((block, Action::VisitSelf));
            for target in self.successors(block) {
                stack.push((target, Action::VisitEdges));
            }
        }
        result
    }

    fn assert_validates(&self) {
        if let Err(err) = self.validate() {
            eprintln!("Function failed validation.");
            eprintln!("Err: {err:?}");
            eprintln!("{}", FunctionPrinter::with_snapshot(self));
            panic!("Aborting...");
        }
    }

    /// Helper function to make an Iongraph JSON "instruction".
    /// `uses`, `memInputs` and `attributes` are left empty for now, but may be populated
    /// in the future.
    fn make_iongraph_instr(id: InsnId, inputs: Vec<Json>, opcode: &str, ty: &str) -> Json {
        Json::object()
            // Add an offset of 0x1000 to avoid the `ptr` being 0x0, which iongraph rejects.
            .insert("ptr", id.0 + 0x1000)
            .insert("id", id.0)
            .insert("opcode", opcode)
            .insert("attributes", Json::empty_array())
            .insert("inputs", Json::Array(inputs))
            .insert("uses", Json::empty_array())
            .insert("memInputs", Json::empty_array())
            .insert("type", ty)
            .build()
    }

    /// Helper function to make an Iongraph JSON "block".
    fn make_iongraph_block(id: BlockId, predecessors: &[BlockId], successors: &[BlockId], instructions: Vec<Json>, attributes: Vec<&str>, loop_depth: u32) -> Json {
        Json::object()
            // Add an offset of 0x1000 to avoid the `ptr` being 0x0, which iongraph rejects.
            .insert("ptr", id.0 + 0x1000)
            .insert("id", id.0)
            .insert("loopDepth", loop_depth)
            .insert("attributes", Json::array(attributes))
            .insert("predecessors", Json::array(predecessors.iter().map(|x| x.to_usize()).collect::<Vec<usize>>()))
            .insert("successors", Json::array(successors.iter().map(|x| x.to_usize()).collect::<Vec<usize>>()))
            .insert("instructions", Json::array(instructions))
            .build()
    }

    /// Helper function to make an Iongraph JSON "function".
    /// Note that `lir` is unpopulated right now as ZJIT doesn't use its functionality.
    fn make_iongraph_function(pass_name: &str, hir_blocks: Vec<Json>) -> Json {
        Json::object()
            .insert("name", pass_name)
            .insert("mir", Json::object()
                .insert("blocks", Json::array(hir_blocks))
                .build()
            )
            .insert("lir", Json::object()
                .insert("blocks", Json::empty_array())
                .build()
            )
            .build()
    }

    /// Generate an iongraph JSON pass representation for this function.
    pub fn to_iongraph_pass(&self, pass_name: &str) -> Json {
        let mut ptr_map = PtrPrintMap::identity();
        if cfg!(test) {
            ptr_map.map_ptrs = true;
        }

        let mut hir_blocks = Vec::new();
        let dominators = Dominators::new(self);
        let cfi = &dominators.cfi;
        let loop_info = LoopInfo::new(&dominators);

        // Push each block from the iteration in reverse post order to `hir_blocks`.
        for block_id in self.reverse_post_order() {
            // Create the block with instructions.
            let block = &self.blocks[block_id.to_usize()];
            let predecessors = cfi.predecessors(block_id);
            let successors = cfi.successors(block_id);
            let mut instructions = Vec::new();

            // Process all instructions (parameters and body instructions).
            // Parameters are currently guaranteed to be Parameter instructions, but in the future
            // they might be refined to other instruction kinds by the optimizer.
            for insn_id in block.params.iter().chain(block.insns.iter()) {
                let insn_id = self.union_find.find_const(*insn_id);
                let insn = self.find(insn_id);

                // Snapshots are not serialized, so skip them.
                if matches!(insn, Insn::Snapshot {..}) {
                    continue;
                }

                // Instructions with no output or an empty type should have an empty type field.
                let type_str = if insn.has_output() {
                    let insn_type = self.type_of(insn_id);
                    if insn_type.is_subtype(types::Empty) {
                        String::new()
                    } else {
                        insn_type.print(&ptr_map).to_string()
                    }
                } else {
                    String::new()
                };


                let opcode = insn.print(&ptr_map, Some(self)).to_string();

                // Collect inputs for a given instruction.
                let mut inputs = Vec::new();
                insn.for_each_operand(|id| inputs.push(id.0.into()));
                let inputs: Vec<Json> = inputs;

                instructions.push(
                    Self::make_iongraph_instr(
                        insn_id,
                        inputs,
                        &opcode,
                        &type_str
                    )
                );
            }

            let mut attributes = vec![];
            if loop_info.is_back_edge_source(block_id) {
                attributes.push("backedge");
            }
            if loop_info.is_loop_header(block_id) {
                attributes.push("loopheader");
            }
            let loop_depth = loop_info.loop_depth(block_id);

            hir_blocks.push(Self::make_iongraph_block(
                block_id,
                predecessors,
                successors,
                instructions,
                attributes,
                loop_depth,
            ));
        }

        Self::make_iongraph_function(pass_name, hir_blocks)
    }

    /// Run all the optimization passes we have.
    pub fn optimize(&mut self) {
        let mut passes: Vec<Json> = Vec::new();
        let should_dump = get_option!(dump_hir_iongraph);

        macro_rules! counter_for {
            // Bucket all strength reduction together
            (type_specialize) => { Counter::compile_hir_strength_reduce_time_ns };
            (convert_no_profile_sends) => { Counter::compile_hir_strength_reduce_time_ns };
            // End strength reduction bucket
            (inline_methods) => { Counter::compile_hir_inline_methods_time_ns };
            (remove_trivial_block_params) => { Counter::compile_hir_remove_trivial_block_params_time_ns };
            (optimize_load_store) => { Counter::compile_hir_optimize_load_store_time_ns };
            (canonicalize) => { Counter::compile_hir_canonicalize_time_ns };
            (fold_constants) => { Counter::compile_hir_fold_constants_time_ns };
            (clean_cfg) => { Counter::compile_hir_clean_cfg_time_ns };
            (reduce_block_params) => { Counter::compile_hir_reduce_block_params_time_ns };
            (remove_redundant_patch_points) => { Counter::compile_hir_remove_redundant_patch_points_time_ns };
            (remove_duplicate_check_interrupts) => { Counter::compile_hir_remove_duplicate_check_interrupts_time_ns };
            (eliminate_empty_inline_frames) => { Counter::compile_hir_eliminate_empty_inline_frames_time_ns };
            (eliminate_dead_code) => { Counter::compile_hir_eliminate_dead_code_time_ns };
            ($name:ident) => { unimplemented!("Counter for pass {}", stringify!($name)) };
        }

        macro_rules! run_pass {
            ($name:ident) => {{
                let counter = counter_for!($name);
                let result = crate::stats::trace_compile_phase(stringify!($name), ||
                    crate::stats::with_time_stat(counter, || self.$name())
                );
                #[cfg(debug_assertions)] crate::stats::trace_compile_phase("validate", || self.assert_validates());
                if should_dump {
                    passes.push(
                        self.to_iongraph_pass(stringify!($name))
                    );
                }

                result
            }};
        }

        if should_dump {
            passes.push(self.to_iongraph_pass("unoptimized"));
        }

        // The optimization pipeline runs in a fixed-point loop so that inlining and
        // type specialization can feed each other: an iteration inlines direct calls and
        // the next one specializes the freshly inlined code, which in turn can expose
        // calls that only became monomorphic after that specialization. Inlining naturally
        // stops when it reaches a fixed point, while inline_max_iterations sets an upper bound
        // on inlining passes. If we reach the max, we run the loop one more time with inlining
        // disabled in order to optimize the results of the last inlining operation.
        let inline_max_iterations = get_option!(inline_max_iterations);
        for iteration in 0..=inline_max_iterations {
            // Function is assumed to have types inferred already
            run_pass!(type_specialize);
            // Cap inlining at inline_max_iterations passes; the trailing iteration (see above)
            // runs the rest of the pipeline with inlining off.
            let did_inline = if iteration < inline_max_iterations {
                run_pass!(inline_methods)
            } else {
                false
            };
            run_pass!(remove_trivial_block_params);
            run_pass!(convert_no_profile_sends);
            run_pass!(optimize_load_store);
            run_pass!(canonicalize);
            run_pass!(fold_constants);
            run_pass!(reduce_block_params);
            run_pass!(clean_cfg);
            run_pass!(remove_redundant_patch_points);
            run_pass!(remove_duplicate_check_interrupts);
            run_pass!(eliminate_empty_inline_frames);
            run_pass!(eliminate_dead_code);

            if !did_inline {
                break;
            }
        }

        if should_dump {
            let iseq_name = iseq_get_location(self.iseq, 0);
            self.dump_iongraph(&iseq_name, passes);
        }
    }

    /// Dump HIR passed to codegen if specified by options.
    pub fn dump_hir(&self) {
        // Dump HIR after optimization
        match get_option!(dump_hir_opt) {
            Some(DumpHIR::WithoutSnapshot) => println!("Optimized HIR:\n{}", FunctionPrinter::without_snapshot(self)),
            Some(DumpHIR::All) => println!("Optimized HIR:\n{}", FunctionPrinter::with_snapshot(self)),
            Some(DumpHIR::Debug) => println!("Optimized HIR:\n{:#?}", &self),
            None => {},
        }
    }

    pub fn dump_iongraph(&self, function_name: &str, passes: Vec<Json>) {
        fn sanitize_for_filename(name: &str) -> String {
            name.chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect()
        }

        use std::io::Write;
        let dir = format!("/tmp/zjit-iongraph-{}", std::process::id());
        std::fs::create_dir_all(&dir).expect("Unable to create directory.");
        let sanitized = sanitize_for_filename(function_name);
        let path = format!("{dir}/func_{sanitized}.json");
        let mut file = std::fs::File::create(path).unwrap();
        let json = Json::object()
            .insert("name", function_name)
            .insert("passes", passes)
            .build();
        writeln!(file, "{json}").unwrap();
    }

    /// Validates the following:
    /// 1. Basic block jump args match parameter arity.
    /// 2. Every terminator must be in the last position.
    /// 3. Every block must have a terminator.
    fn validate_block_terminators_and_jumps(&self) -> Result<(), ValidationError> {
        let check_edge = |block_id: BlockId, edge: &BranchEdge| -> Result<(), ValidationError> {
            let target_len = self.blocks[edge.target.to_usize()].params.len();
            let args_len = edge.args.len();
            if target_len != args_len {
                return Err(ValidationError::MismatchedBlockArity(block_id, target_len, args_len));
            }
            Ok(())
        };

        for block_id in self.reverse_post_order() {
            let insns = &self.blocks[block_id.to_usize()].insns;
            for (idx, insn_id) in insns.iter().enumerate() {
                // No need for resolve(): we only look at edge targets/arity and terminators.
                let insn = self.find_ref(*insn_id);
                // Validate arity for all branch edges
                match insn {
                    Insn::Jump(edge) => {
                        check_edge(block_id, edge)?;
                    }
                    Insn::CondBranch { if_true, if_false, .. } => {
                        check_edge(block_id, if_true)?;
                        check_edge(block_id, if_false)?;
                    }
                    _ => {}
                }

                if insn.is_terminator() {
                    // Blow up if we have a terminator that isn't at the end
                    // of the block.
                    if idx != insns.len() - 1 {
                        return Err(ValidationError::TerminatorNotAtEnd(block_id, *insn_id, idx))
                    }
                }
                // If the last instruction isn't a terminator, return an error
                if idx == insns.len() - 1 {
                    if !insn.is_terminator() {
                        return Err(ValidationError::BlockHasNoTerminator(block_id));
                    }
                }
            }
        }
        Ok(())
    }

    // This performs a dataflow def-analysis over the entire CFG to detect any
    // possibly undefined instruction operands.
    fn validate_definite_assignment(&self) -> Result<(), ValidationError> {
        // Map of block ID -> InsnSet
        // Initialize with all missing values at first, to catch if a jump target points to a
        // missing location.
        // One flat matrix rather than one whole-function-width bitset per block:
        // validate() runs this dataflow on every compile, and a `Vec<BitSet>` costs an
        // allocation per block plus a full clone of one on every worklist visit.
        let mut assigned_in = BitMatrix::<InsnId>::new(self.num_blocks(), self.insns.len());
        let mut in_rpo = BlockSet::with_capacity(self.num_blocks());
        let rpo = self.reverse_post_order();
        // Begin with every block having every variable defined, except for entries_block, which
        // starts with nothing defined. Blocks outside the RPO are tracked separately so a jump
        // into one is still reported.
        for &block in &rpo {
            in_rpo.insert(block);
            if block != self.entries_block {
                assigned_in.insert_all_row(block.to_usize());
            }
        }
        let mut worklist = VecDeque::with_capacity(self.num_blocks());
        worklist.push_back(self.entries_block);
        let mut assigned = InsnSet::with_capacity(self.insns.len());
        while let Some(block) = worklist.pop_front() {
            assigned.copy_from_row(assigned_in.row(block.to_usize()));
            for &param in &self.blocks[block.to_usize()].params {
                assigned.insert(param);
            }
            for &insn_id in &self.blocks[block.to_usize()].insns {
                let insn_id = self.union_find.find_const(insn_id);
                // No need for resolve(): we only look at jump targets here, and the
                // operand check below resolves each operand itself.
                let insn = self.find_ref(insn_id);
                let mut propagate = |target: BlockId| -> Result<(), ValidationError> {
                    if !in_rpo.get(target) {
                        return Err(ValidationError::JumpTargetNotInRPO(target));
                    }
                    if assigned_in.intersect_row_with(target.to_usize(), &assigned) {
                        worklist.push_back(target);
                    }
                    Ok(())
                };
                match insn {
                    Insn::Jump(edge) => propagate(edge.target)?,
                    Insn::CondBranch { if_true, if_false, .. } => {
                        propagate(if_true.target)?;
                        propagate(if_false.target)?;
                    }
                    Insn::Entries { targets } => {
                        for &target in targets {
                            propagate(target)?;
                        }
                    }
                    insn if insn.has_output() => {
                        assigned.insert(insn_id);
                    }
                    _ => {}
                }
            }
        }
        // Check that each instruction's operands are assigned
        for &block in &rpo {
            assigned.copy_from_row(assigned_in.row(block.to_usize()));
            for &param in &self.blocks[block.to_usize()].params {
                assigned.insert(param);
            }
            for &insn_id in &self.blocks[block.to_usize()].insns {
                let insn_id = self.union_find.find_const(insn_id);
                self.insns[insn_id.to_usize()].try_for_each_operand(|operand| {
                    let operand = self.union_find.find_const(operand);
                    if !assigned.get(operand) {
                        return Err(ValidationError::OperandNotDefined(block, insn_id, operand));
                    }
                    Ok(())
                })?;
                if self.insns[insn_id.to_usize()].has_output() {
                    assigned.insert(insn_id);
                }
            }
        }
        Ok(())
    }

    /// Checks that each instruction('s representative) appears only once in the CFG.
    fn validate_insn_uniqueness(&self) -> Result<(), ValidationError> {
        let mut seen = InsnSet::with_capacity(self.insns.len());
        for block_id in self.reverse_post_order() {
            for &insn_id in &self.blocks[block_id.to_usize()].insns {
                let insn_id = self.union_find.find_const(insn_id);
                if !seen.insert(insn_id) {
                    return Err(ValidationError::DuplicateInstruction(block_id, insn_id));
                }
            }
        }
        Ok(())
    }

    fn assert_subtype(&self, user: InsnId, operand: InsnId, expected: Type) -> Result<(), ValidationError> {
        let actual = self.type_of(operand);
        if !actual.is_subtype(expected) {
            return Err(ValidationError::MismatchedOperandType(user, operand, format!("{expected}"), format!("{actual}")));
        }
        Ok(())
    }

    fn validate_insn_type(&self, insn_id: InsnId) -> Result<(), ValidationError> {
        let insn_id = self.union_find.find_const(insn_id);
        // No need for resolve(): type_of() resolves operands for us.
        match *self.find_ref(insn_id) {
            // Instructions with no InsnId operands (except state) or nothing to assert
            Insn::Const { .. }
            | Insn::IvarReprofile { .. }
            | Insn::Comment { .. }
            | Insn::Param
            | Insn::LoadArg { .. }
            | Insn::PutSpecialObject { .. }
            | Insn::LoadField { .. }
            | Insn::UnwrapSvar { .. }
            | Insn::GetConstantPath { .. }
            | Insn::IsBlockGiven { .. }
            | Insn::GetGlobal { .. }
            | Insn::LoadPC
            | Insn::LoadSP
            | Insn::LoadEC
            | Insn::GetEP { .. }
            | Insn::BreakPoint | Insn::Unreachable
            | Insn::LoadSelf
            | Insn::Snapshot { .. }
            | Insn::Jump { .. }
            | Insn::Entries { .. }
            | Insn::EntryPoint { .. }
            | Insn::PatchPoint { .. }
            | Insn::SideExit { .. }
            | Insn::IncrCounter { .. }
            | Insn::IncrCounterPtr { .. }
            | Insn::CheckInterrupts { .. }
            | Insn::GetClassVar { .. }
            | Insn::GetSpecialNumber { .. }
            | Insn::GetSpecialSymbol { .. }
            | Insn::GetBlockParam { .. }
            | Insn::Once { .. }
            | Insn::StoreField { .. } => {
                Ok(())
            }
            // Instructions with 1 Ruby object operand
            Insn::Test { val }
            | Insn::IsMethodCfunc { val, .. }
            | Insn::SetGlobal { val, .. }
            | Insn::SetLocal { val, .. }
            | Insn::SetClassVar { val, .. }
            | Insn::Return { val, .. }
            | Insn::Throw { val, .. }
            | Insn::GuardType { val, .. }
            | Insn::GuardNotRuby2KeywordsHash { val, .. }
            | Insn::ToArray { val, .. }
            | Insn::ToHash { val, .. }
            | Insn::CheckArrayType { val, .. }
            | Insn::ToAryForExpand { val, .. }
            | Insn::ToNewArray { val, .. }
            | Insn::Defined { v: val, .. }
            | Insn::ObjectAlloc { val, .. }
            | Insn::DupArrayInclude { target: val, .. }
            | Insn::GetIvar { self_val: val, .. }
            | Insn::CCall { recv: val, .. }
            | Insn::FixnumBitCheck { val, .. } // TODO (https://github.com/Shopify/ruby/issues/859) this should check Fixnum, but then test_checkkeyword_tests_fixnum_bit fails
            | Insn::DefinedIvar { self_val: val, .. } => {
                self.assert_subtype(insn_id, val, types::BasicObject)
            }
            // Instructions with 2 Ruby object operands
            Insn::SetIvar { self_val: left, val: right, .. }
            | Insn::NewRange { low: left, high: right, .. }
            | Insn::CheckMatch { target: left, pattern: right, .. }
            | Insn::WriteBarrier { recv: left, val: right } => {
                self.assert_subtype(insn_id, left, types::RubyValue)?;
                self.assert_subtype(insn_id, right, types::RubyValue)
            }
            Insn::GetConstant { klass, allow_nil, .. } => {
                self.assert_subtype(insn_id, klass, types::BasicObject)?;
                self.assert_subtype(insn_id, allow_nil, types::BoolExact)
            }
            Insn::AnyToString { val, .. } => {
                self.assert_subtype(insn_id, val, types::BasicObject)
            }
            Insn::PushInlineFrame { recv, .. } => {
                self.assert_subtype(insn_id, recv, types::BasicObject)
            }
            // Instructions with recv and a Vec of Ruby objects
            | Insn::Send { recv, ref args, .. }
            | Insn::SendForward { recv, ref args, .. }
            | Insn::InvokeSuper { recv, ref args, .. }
            | Insn::InvokeSuperForward { recv, ref args, .. }
            | Insn::InvokeBuiltin { recv, ref args, .. }
            | Insn::InvokeProc { recv, ref args, .. }
            | Insn::ArrayInclude { target: recv, elements: ref args, .. } => {
                self.assert_subtype(insn_id, recv, types::BasicObject)?;
                for &arg in args {
                    self.assert_subtype(insn_id, arg, types::BasicObject)?;
                }
                Ok(())
            }
            Insn::SendDirect(ref insn) => {
                self.assert_subtype(insn_id, insn.recv, types::BasicObject)?;
                for &arg in &insn.args {
                    self.assert_subtype(insn_id, arg, types::BasicObject)?;
                }
                if let Some(block_arg) = insn.block_arg {
                    self.assert_subtype(insn_id, block_arg, types::BasicObject)?;
                }
                Ok(())
            }
            Insn::CCallWithFrame(ref insn) => {
                self.assert_subtype(insn_id, insn.recv, types::BasicObject)?;
                for &arg in &insn.args {
                    self.assert_subtype(insn_id, arg, types::BasicObject)?;
                }
                if let Some(block_arg) = insn.block_arg {
                    self.assert_subtype(insn_id, block_arg, types::BasicObject)?;
                }
                Ok(())
            }
            Insn::CCallVariadic(ref insn) => {
                self.assert_subtype(insn_id, insn.recv, types::BasicObject)?;
                for &arg in &insn.args {
                    self.assert_subtype(insn_id, arg, types::BasicObject)?;
                }
                if let Some(block_arg) = insn.block_arg {
                    self.assert_subtype(insn_id, block_arg, types::BasicObject)?;
                }
                Ok(())
            }
            Insn::ArrayPackBuffer { ref elements, fmt, buffer, .. } => {
                self.assert_subtype(insn_id, fmt, types::BasicObject)?;
                if let Some(buffer) = buffer {
                    self.assert_subtype(insn_id, buffer, types::BasicObject)?;
                }
                for &element in elements {
                    self.assert_subtype(insn_id, element, types::BasicObject)?;
                }
                Ok(())
            }
            // Instructions with a Vec of Ruby objects
            Insn::InvokeBlock { ref args, .. }
            | Insn::InvokeBlockIseqDirect { ref args, .. }
            | Insn::InvokeBlockIfunc { ref args, .. }
            | Insn::NewArray { elements: ref args, .. }
            | Insn::ArrayHash { elements: ref args, .. }
            | Insn::ArrayMin { elements: ref args, .. }
            | Insn::ArrayMax { elements: ref args, .. } => {
                for &arg in args {
                    self.assert_subtype(insn_id, arg, types::BasicObject)?;
                }
                Ok(())
            }
            Insn::NewHash { ref elements, .. } => {
                if elements.len() % 2 != 0 {
                    return Err(ValidationError::MiscValidationError(insn_id, "NewHash elements length is not even".to_string()));
                }
                for &element in elements {
                    self.assert_subtype(insn_id, element, types::BasicObject)?;
                }
                Ok(())
            }
            Insn::StringConcat { ref strings, .. }
            | Insn::ToRegexp { values: ref strings, .. } => {
                for &string in strings {
                    self.assert_subtype(insn_id, string, types::String)?;
                }
                Ok(())
            }
            // Instructions with String operands
            Insn::StringCopy { val, .. } => self.assert_subtype(insn_id, val, types::StringExact),
            Insn::StringIntern { val, .. } => self.assert_subtype(insn_id, val, types::StringExact),
            Insn::StringAppend { recv, other, .. } => {
                self.assert_subtype(insn_id, recv, types::StringExact)?;
                self.assert_subtype(insn_id, other, types::String)
            }
            Insn::StringAppendCodepoint { recv, other, .. } => {
                self.assert_subtype(insn_id, recv, types::StringExact)?;
                self.assert_subtype(insn_id, other, types::Fixnum)
            }
            Insn::StringEqual { left, right } => {
                self.assert_subtype(insn_id, left, types::String)?;
                self.assert_subtype(insn_id, right, types::String)
            }
            // Instructions with Array operands
            Insn::ArrayDup { val, .. } => self.assert_subtype(insn_id, val, types::ArrayExact),
            Insn::ArrayExtend { left, right, .. } => {
                // TODO(max): Do left and right need to be ArrayExact?
                self.assert_subtype(insn_id, left, types::Array)?;
                self.assert_subtype(insn_id, right, types::Array)
            }
            Insn::ArrayPush { array, .. }
            | Insn::ArrayPop { array, .. }
            | Insn::ArrayLength { array, .. } => {
                self.assert_subtype(insn_id, array, types::Array)
            }
            Insn::ArrayAref { array, index } => {
                self.assert_subtype(insn_id, array, types::Array)?;
                self.assert_subtype(insn_id, index, types::CInt64)
            }
            Insn::ArrayArefOrNil { array, index, length } => {
                self.assert_subtype(insn_id, array, types::Array)?;
                self.assert_subtype(insn_id, index, types::CInt64)?;
                self.assert_subtype(insn_id, length, types::CInt64)
            }
            Insn::ArrayAset { array, index, .. } => {
                self.assert_subtype(insn_id, array, types::ArrayExact)?;
                self.assert_subtype(insn_id, index, types::CInt64)
            }
            Insn::ArrayAsetOrStore { array, index, length, .. } => {
                self.assert_subtype(insn_id, array, types::ArrayExact)?;
                self.assert_subtype(insn_id, index, types::CInt64)?;
                self.assert_subtype(insn_id, length, types::CInt64)
            }
            Insn::AdjustBounds { index, length } => {
                self.assert_subtype(insn_id, index, types::CInt64)?;
                self.assert_subtype(insn_id, length, types::CInt64)
            }
            // Instructions with Hash operands
            Insn::HashAref { hash, .. }
            | Insn::HashAset { hash, .. } => self.assert_subtype(insn_id, hash, types::HashExact),
            Insn::HashDup { val, .. } => self.assert_subtype(insn_id, val, types::HashExact),
            // Other
            Insn::ObjectAllocClass { class, .. } => {
                if !class_has_leaf_allocator(class) {
                    return Err(ValidationError::MiscValidationError(insn_id, "ObjectAllocClass must have leaf allocator".to_string()));
                }
                Ok(())
            }
            Insn::IsBitEqual { left, right }
            | Insn::IsBitNotEqual { left, right } => {
                if self.is_a(left, types::CInt) && self.is_a(right, types::CInt) {
                    // TODO(max): Check that int sizes match
                    Ok(())
                } else if self.is_a(left, types::CPtr) && self.is_a(right, types::CPtr) {
                    Ok(())
                } else if self.is_a(left, types::RubyValue) && self.is_a(right, types::RubyValue) {
                    Ok(())
                } else {
                    Err(ValidationError::MiscValidationError(insn_id, "IsBitEqual can only compare CInt/CInt or RubyValue/RubyValue".to_string()))
                }
            }
            Insn::IntAnd { left, right }
            | Insn::IntOr { left, right } => {
                // TODO: Expand this to other matching C integer sizes when we need them.
                let left_type = self.type_of(left);
                if left_type.is_subtype(types::CInt64) {
                    self.assert_subtype(insn_id, right, types::CInt64)
                } else if left_type.is_subtype(types::CUInt64) {
                    self.assert_subtype(insn_id, right, types::CUInt64)
                } else {
                    let all_ints = types::CInt64.union(types::CUInt64);
                    self.assert_subtype(insn_id, left, all_ints)?;
                    self.assert_subtype(insn_id, right, all_ints)
                }
            }
            Insn::BoxBool { val }
            | Insn::CondBranch { val, .. } => {
                self.assert_subtype(insn_id, val, types::CBool)
            }
            Insn::BoxFixnum { val, .. } => self.assert_subtype(insn_id, val, types::CInt64),
            Insn::UnboxFixnum { val } => {
                self.assert_subtype(insn_id, val, types::Fixnum)
            }
            Insn::FixnumAref { recv, index } => {
                self.assert_subtype(insn_id, recv, types::Fixnum)?;
                self.assert_subtype(insn_id, index, types::Fixnum)
            }
            Insn::FixnumAdd { left, right, .. }
            | Insn::FixnumSub { left, right, .. }
            | Insn::FixnumMult { left, right, .. }
            | Insn::FixnumDiv { left, right, .. }
            | Insn::FixnumMod { left, right, .. }
            | Insn::FixnumEq { left, right }
            | Insn::FixnumNeq { left, right }
            | Insn::FixnumLt { left, right }
            | Insn::FixnumLe { left, right }
            | Insn::FixnumGt { left, right }
            | Insn::FixnumGe { left, right }
            | Insn::FixnumAnd { left, right }
            | Insn::FixnumOr { left, right }
            | Insn::FixnumXor { left, right }
            | Insn::NewRangeFixnum { low: left, high: right, .. }
            => {
                self.assert_subtype(insn_id, left, types::Fixnum)?;
                self.assert_subtype(insn_id, right, types::Fixnum)
            }
            Insn::FloatAdd { recv, other, .. }
            | Insn::FloatSub { recv, other, .. }
            | Insn::FloatMul { recv, other, .. }
            | Insn::FloatDiv { recv, other, .. }
            => {
                self.assert_subtype(insn_id, recv, types::Flonum)?;
                // other can be Flonum or Fixnum (rb_float_plus etc. handle both)
                self.assert_subtype(insn_id, other, types::Flonum.union(types::Fixnum))
            }
            Insn::FloatLt { left, right }
            | Insn::FloatLe { left, right }
            | Insn::FloatGt { left, right }
            | Insn::FloatGe { left, right }
            => {
                self.assert_subtype(insn_id, left, types::Flonum)?;
                // right can be Flonum or Fixnum (rb_float_lt etc. handle both)
                self.assert_subtype(insn_id, right, types::Flonum.union(types::Fixnum))
            }
            Insn::FloatToInt { recv, .. } => {
                self.assert_subtype(insn_id, recv, types::Flonum)
            }
            Insn::FixnumLShift { left, right, .. }
            | Insn::FixnumRShift { left, right, .. } => {
                self.assert_subtype(insn_id, left, types::Fixnum)?;
                self.assert_subtype(insn_id, right, types::Fixnum)?;
                let Some(obj) = self.type_of(right).fixnum_value() else {
                    return Err(ValidationError::MismatchedOperandType(insn_id, right, "<a compile-time constant>".into(), "<unknown>".into()));
                };
                if obj < 0 {
                    return Err(ValidationError::MismatchedOperandType(insn_id, right, "<positive>".into(), format!("{obj}")));
                }
                if obj > 63 {
                    return Err(ValidationError::MismatchedOperandType(insn_id, right, "<less than 64>".into(), format!("{obj}")));
                }
                Ok(())
            }
            Insn::GuardBitEquals { val, expected, .. } => {
                match expected {
                    Const::Value(_) => self.assert_subtype(insn_id, val, types::RubyValue),
                    Const::CInt8(_) => self.assert_subtype(insn_id, val, types::CInt8),
                    Const::CInt16(_) => self.assert_subtype(insn_id, val, types::CInt16),
                    Const::CInt32(_) => self.assert_subtype(insn_id, val, types::CInt32),
                    Const::CInt64(_) => self.assert_subtype(insn_id, val, types::CInt64),
                    Const::CUInt8(_) => self.assert_subtype(insn_id, val, types::CUInt8),
                    Const::CUInt16(_) => self.assert_subtype(insn_id, val, types::CUInt16),
                    Const::CUInt32(_) => self.assert_subtype(insn_id, val, types::CUInt32),
                    Const::CAttrIndex(_) => self.assert_subtype(insn_id, val, types::CAttrIndex),
                    Const::CShape(_) => self.assert_subtype(insn_id, val, types::CShape),
                    Const::CUInt64(_) => self.assert_subtype(insn_id, val, types::CUInt64),
                    Const::CBool(_) => self.assert_subtype(insn_id, val, types::CBool),
                    Const::CDouble(_) => self.assert_subtype(insn_id, val, types::CDouble),
                    Const::CPtr(_) => self.assert_subtype(insn_id, val, types::CPtr),
                }
            }
            Insn::GuardAnyBitSet { val, mask, .. }
            | Insn::GuardNoBitsSet { val, mask, .. } => {
                match mask {
                    Const::CUInt8(_) | Const::CUInt16(_) | Const::CUInt32(_) | Const::CUInt64(_)
                        if self.is_a(val, types::CInt) || self.is_a(val, types::RubyValue) => {
                        Ok(())
                    }
                    _ => {
                        Err(ValidationError::MiscValidationError(insn_id, "GuardAnyBitSet/GuardNoBitsSet can only compare RubyValue/CUInt or CInt/CUInt".to_string()))
                    }
                }
            }
            Insn::GuardLess { left, right, .. }
            | Insn::GuardGreaterEq { left, right, .. } => {
                self.assert_subtype(insn_id, left, types::CInt64)?;
                self.assert_subtype(insn_id, right, types::CInt64)
            },
            Insn::StringGetbyte { string, index } => {
                self.assert_subtype(insn_id, string, types::String)?;
                self.assert_subtype(insn_id, index, types::CInt64)
            },
            Insn::StringCoderangeOrScan { string, cached, .. } => {
                self.assert_subtype(insn_id, string, types::String)?;
                self.assert_subtype(insn_id, cached, types::CInt64)
            },
            Insn::StringSetbyteFixnum { string, index, value } => {
                self.assert_subtype(insn_id, string, types::String)?;
                self.assert_subtype(insn_id, index, types::CInt64)?;
                self.assert_subtype(insn_id, value, types::Fixnum)
            }
            Insn::IsA { val, class } => {
                self.assert_subtype(insn_id, val, types::BasicObject)?;
                self.assert_subtype(insn_id, class, types::Class)
            }
            Insn::RefineType { .. } => Ok(()),
            Insn::HasType { val, .. } => self.assert_subtype(insn_id, val, types::BasicObject),
            Insn::HasAncestor { val, .. } => self.assert_subtype(insn_id, val, types::BasicObject),
            Insn::IsBlockParamModified { flags } => self.assert_subtype(insn_id, flags, types::CUInt64),
            // Frame instructions have no output to validate; their operands
            // are validated by the recv+args group (PushLightweightFrame)
            // or the state-only group (PopLightweightFrame).
            Insn::PopInlineFrame { .. } => Ok(()),
        }
    }

    /// Check that insn types match the expected types for each instruction.
    fn validate_types(&self) -> Result<(), ValidationError> {
        for block_id in self.reverse_post_order() {
            for &insn_id in &self.blocks[block_id.to_usize()].insns {
                self.validate_insn_type(insn_id)?;
            }
        }
        Ok(())
    }

    /// Run all validation passes we have.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.validate_block_terminators_and_jumps()?;
        self.validate_definite_assignment()?;
        self.validate_insn_uniqueness()?;
        self.validate_types()?;
        Ok(())
    }

    /// Dispatch an ivar access to profiled shapes. Callbacks generate the optimized access and
    /// generic fallback, optionally returning a value to pass through the join block.
    /// Emit a call that records the shape of a receiver reaching a frozen ivar dispatch's
    /// fallback, unless the ISEQ has already spent its respecialization budget (in which case
    /// no recompile can follow and the call would just be overhead on a hot path).
    fn emit_ivar_reprofile(&mut self, block: BlockId, self_param: InsnId, state: InsnId) {
        if self.iseq.is_null() {
            return;
        }
        let payload = get_or_create_iseq_payload(self.iseq);
        if payload.ivar_respecializations >= crate::payload::MAX_IVAR_RESPECIALIZATIONS
            || payload.ivar_reprofile_giveup
        {
            return;
        }
        self.push_insn(block, Insn::IvarReprofile { self_val: self_param, state });
    }

    fn dispatch_ivar<T: Copy>(
        &mut self,
        profiles: &[T],
        covers_profile: bool,
        mut block: BlockId,
        insn_idx: u32,
        self_param: InsnId,
        exit_id: InsnId,
        no_profile_reason: SideExitReason,
        no_profile_counter: Counter,
        chain_miss_counter: Counter,
        has_result: bool,
        shape_miss: ShapeMiss,
        profile_shape: impl Fn(T) -> ShapeId,
        mut emit_optimized: impl FnMut(&mut Function, BlockId, T) -> Option<InsnId>,
        mut emit_fallback: impl FnMut(&mut Function, BlockId) -> Option<InsnId>,
    ) -> Option<(BlockId, Option<InsnId>)> {
        // The final version of an ISEQ may not speculate at all, whatever the caller asked for.
        let shape_miss = match shape_miss {
            ShapeMiss::SideExit if self.policy.no_side_exits => ShapeMiss::CallFallback,
            shape_miss => shape_miss,
        };
        // Whether a shape this dispatch predicts may be guarded with an exit. The policy check is
        // explicit because the conversion above only rewrites `SideExit`; a caller that asks for
        // `CallFallback` directly still must not exit in the final version of an ISEQ.
        let speculate = shape_miss.speculates_on_predicted_shape() && !self.policy.no_side_exits;
        // 0 profiles: Generate a recompile exit or a fallback. No need for new HIR blocks.
        if profiles.is_empty() {
            if !speculate {
                self.count(block, no_profile_counter);
                if shape_miss == ShapeMiss::CallFallback {
                    self.emit_ivar_reprofile(block, self_param, exit_id);
                }
                let result = emit_fallback(self, block);
                assert_eq!(has_result, result.is_some());
                return Some((block, result));
            } else {
                self.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(no_profile_reason), recompile: Some(Recompile) });
                return None;
            }
        }
        // 1 profile: Generate a monomorphic ivar access with a guard if allowed by policy. No need for new HIR blocks.
        // A monomorphic guard is the one shape guard whose failure is worth an exit: the site has
        // only ever seen one shape, so a miss means the profile is stale or too narrow, and the
        // recompile it triggers can widen the site into a shape chain. That only holds if the
        // profile accounts for every receiver: without `covers_profile` we already know at compile
        // time that some receivers cannot match, so the guard would exit by construction.
        if profiles.len() == 1 && covers_profile && speculate {
            let actual = self.load_shape(block, self_param);
            self.guard_shape(block, actual, profile_shape(profiles[0]), exit_id, Some(Recompile));
            let result = emit_optimized(self, block, profiles[0]);
            assert_eq!(has_result, result.is_some());
            return Some((block, result));
        }

        // Otherwise, make HIR blocks to handle different shapes or a fallback, and let them jump to join_block.
        let edge = |target: BlockId| BranchEdge { target, args: vec![] };
        let branch = |cond: InsnId, if_true: BlockId, if_false: BlockId| Insn::CondBranch { val: cond, if_true: edge(if_true), if_false: edge(if_false) };
        let result_edge = |target: BlockId, result: Option<InsnId>| {
            assert_eq!(has_result, result.is_some());
            BranchEdge { target, args: result.into_iter().collect() }
        };
        let actual = self.load_shape(block, self_param);
        let last_shape_index = profiles.len() - 1;
        let join_block = self.new_block(insn_idx);
        let result = has_result.then(|| self.push_insn(join_block, Insn::Param));
        for (i, &profile) in profiles.iter().enumerate() {
            let optimized_block = self.new_block(insn_idx);
            if i == last_shape_index {
                if shape_miss.calls_fallback() {
                    // Without a side exit available, make a fallback block and jump to it if the shape doesn't match.
                    let expected = self.push_insn(block, Insn::Const { val: Const::CShape(profile_shape(profile)) });
                    let matches = self.push_insn(block, Insn::IsBitEqual { left: actual, right: expected });
                    let fallback_block = self.new_block(insn_idx);
                    self.push_insn(block, branch(matches, optimized_block, fallback_block));
                    // A receiver that misses every arm of a shape chain gets its own counter: it
                    // is a different event from having no profile to dispatch on at all.
                    self.count(fallback_block, chain_miss_counter);
                    if shape_miss == ShapeMiss::CallFallback {
                        self.emit_ivar_reprofile(fallback_block, self_param, exit_id);
                    }
                    let fallback_result = emit_fallback(self, fallback_block);
                    self.push_insn(fallback_block, Insn::Jump(result_edge(join_block, fallback_result)));
                } else {
                    // Otherwise exit to the interpreter if the shape doesn't match.
                    self.guard_shape(block, actual, profile_shape(profile), exit_id, Some(Recompile));
                    // TODO(max): Don't make a new block in this case
                    self.push_insn(block, Insn::Jump(edge(optimized_block)));
                }
            } else {
                // If this is not the last profiled shape, let the guard jump to the next block.
                let expected = self.push_insn(block, Insn::Const { val: Const::CShape(profile_shape(profile)) });
                let matches = self.push_insn(block, Insn::IsBitEqual { left: actual, right: expected });
                let next_block = self.new_block(insn_idx);
                self.push_insn(block, branch(matches, optimized_block, next_block));
                block = next_block;
            }
            let optimized_result = emit_optimized(self, optimized_block, profile);
            self.push_insn(optimized_block, Insn::Jump(result_edge(join_block, optimized_result)));
        }
        Some((join_block, result))
    }

    /// Return Some(InsnId) if we generated any code to load an ivar and None if we only generated
    /// an unconditional SideExit (in which case we should end the block).
    fn dispatch_getivar(
        &mut self,
        profiled_types: &[ProfiledType],
        covers_profile: bool,
        block: BlockId,
        insn_idx: u32,
        self_param: InsnId,
        id: ID,
        ic: *const iseq_inline_iv_cache_entry,
        exit_id: InsnId,
        shape_miss: ShapeMiss,
    ) -> Option<(BlockId, InsnId)> {
        let (block, result) = self.dispatch_ivar(
            profiled_types,
            covers_profile,
            block,
            insn_idx,
            self_param,
            exit_id,
            SideExitReason::NoProfileGetIvar,
            Counter::getivar_fallback_no_side_exits,
            Counter::getivar_fallback_shape_chain_miss,
            true,
            shape_miss,
            |profiled_type| profiled_type.shape(),
            |fun, block, profiled_type| Some(fun.load_ivar(block, self_param, profiled_type, id)),
            |fun, block| Some(fun.push_insn(block, Insn::GetIvar { self_val: self_param, id, ic, state: exit_id })),
        )?;
        Some((block, result.unwrap()))
    }

    /// Return Some(BlockId) if we generated a setivar or None if we only generated an
    /// unconditional SideExit (in which case we should end the block).
    fn dispatch_setivar(
        &mut self,
        specs: &[SetIvarSpec],
        unoptimized_reason: Option<Counter>,
        covers_profile: bool,
        block: BlockId,
        insn_idx: u32,
        self_param: InsnId,
        id: ID,
        ic: *const iseq_inline_iv_cache_entry,
        val: InsnId,
        exit_id: InsnId,
        shape_miss: ShapeMiss,
    ) -> Option<BlockId> {
        if specs.is_empty() {
            if let Some(counter) = unoptimized_reason {
                self.count(block, counter);
                self.push_insn(block, Insn::SetIvar { self_val: self_param, id, ic, val, state: exit_id });
                return Some(block);
            }
        }
        let (block, result) = self.dispatch_ivar(
            specs,
            // A spec set that dropped a bucket for `unoptimized_reason` cannot account for every
            // receiver, whatever the profile summary said.
            covers_profile && unoptimized_reason.is_none(),
            block,
            insn_idx,
            self_param,
            exit_id,
            SideExitReason::NoProfileSetIvar,
            Counter::setivar_fallback_no_side_exits,
            Counter::setivar_fallback_shape_chain_miss,
            false,
            shape_miss,
            |spec| spec.profiled_type.shape(),
            |fun, block, spec| {
                fun.emit_optimized_setivar(block, self_param, id, val, spec);
                None
            },
            |fun, block| {
                fun.push_insn(block, Insn::SetIvar { self_val: self_param, id, ic, val, state: exit_id });
                None
            },
        )?;
        assert!(result.is_none());
        Some(block)
    }
}

impl<'a> std::fmt::Display for FunctionPrinter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let fun = &self.fun;
        // In tests, there may not be an iseq to get location from.
        let iseq_name = if fun.iseq.is_null() {
            String::from("<manual>")
        } else {
            iseq_get_location(fun.iseq, 0)
        };

        // In tests, strip the line number for builtin ISEQs to make tests stable across line changes
        let iseq_name = if cfg!(test) && iseq_name.contains("@<internal:") {
            iseq_name[..iseq_name.rfind(':').unwrap()].to_string()
        } else {
            iseq_name
        };
        writeln!(f, "fn {iseq_name}:")?;
        for block_id in fun.reverse_post_order() {
            if !self.display_snapshot_and_tp_patchpoints && block_id == fun.entries_block {
                // Unless we're doing --zjit-dump-hir=all, skip the entries superblock -- it's an
                // internal CFG artifact
                continue;
            }
            write!(f, "{block_id}(")?;
            if !fun.blocks[block_id.to_usize()].params.is_empty() {
                let mut sep = "";
                for param in &fun.blocks[block_id.to_usize()].params {
                    write!(f, "{sep}{param}")?;
                    let insn_type = fun.type_of(*param);
                    if !insn_type.is_subtype(types::Empty) {
                        write!(f, ":{}", insn_type.print(&self.ptr_map))?;
                    }
                    sep = ", ";
                }
            }
            writeln!(f, "):")?;
            for insn_id in &fun.blocks[block_id.to_usize()].insns {
                let insn = fun.find(*insn_id);
                if !self.display_snapshot_and_tp_patchpoints &&
                    matches!(insn, Insn::Snapshot {..} | Insn::PatchPoint { invariant: Invariant::NoTracePoint, .. }) {
                    continue;
                }
                write!(f, "  ")?;
                if insn.has_output() {
                    let insn_type = fun.type_of(*insn_id);
                    if insn_type.is_subtype(types::Empty) {
                        write!(f, "{insn_id} = ")?;
                    } else {
                        write!(f, "{insn_id}:{} = ", insn_type.print(&self.ptr_map))?;
                    }
                }
                writeln!(f, "{}", insn.print(&self.ptr_map, Some(fun)))?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct FrameState {
    pub iseq: IseqPtr,
    insn_idx: YarvInsnIdx,
    // Ruby bytecode instruction pointer
    pub pc: *const VALUE,

    stack: Vec<InsnId>,
    locals: Vec<InsnId>,

    /// `InsnId` of the caller's post-send `Snapshot` for inlined frames; `None`
    /// for non-inlined frames. Stored as an instruction reference rather than
    /// an owned `FrameState` so that value remapping in the caller's `Snapshot`
    /// propagates here automatically, and so the caller's state has a single
    /// source of truth in the IR.
    caller: Option<InsnId>,

    /// Inlining nesting depth of this frame. The top-level (non-inlined) frame
    /// is depth 0; each level of inlining increments it by one. Codegen uses
    /// this to pick a distinct JITFrame slot per active inlined frame so that
    /// `cfp->jit_return` values do not alias across the shared native stack frame.
    /// This value's upper bound is the `inline_max_iterations` value.
    pub depth: InlineDepth,
}

/// Hand-written so that `clone_from` reuses the destination's `stack` and
/// `locals` buffers. `#[derive(Clone)]` leaves `clone_from` at its default,
/// which throws the destination away and allocates two fresh vectors; HIR
/// construction snapshots the frame state once per YARV instruction, which made
/// that the largest single source of allocator traffic in the compiler.
impl Clone for FrameState {
    fn clone(&self) -> Self {
        Self {
            iseq: self.iseq,
            insn_idx: self.insn_idx,
            pc: self.pc,
            stack: self.stack.clone(),
            locals: self.locals.clone(),
            caller: self.caller,
            depth: self.depth,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.iseq = source.iseq;
        self.insn_idx = source.insn_idx;
        self.pc = source.pc;
        self.stack.clone_from(&source.stack);
        self.locals.clone_from(&source.locals);
        self.caller = source.caller;
        self.depth = source.depth;
    }
}

impl FrameState {
    /// Get the YARV instruction index for the current instruction
    pub fn insn_idx(&self) -> YarvInsnIdx {
        self.insn_idx
    }

    /// Return itself without locals. Useful for side-exiting without spilling locals.
    fn without_locals(&self) -> Self {
        let mut state = self.clone();
        state.locals.clear();
        state
    }

    /// Return itself without stack. Used by leaf calls with GC to reset SP to the base pointer.
    pub fn without_stack(&self) -> Self {
        let mut state = self.clone();
        state.stack.clear();
        state
    }

    /// Return itself with a truncated stack.
    pub fn with_stack_size(&self, stack_size: usize) -> Self {
        let mut state = self.clone();
        state.stack.truncate(stack_size);
        state
    }

    /// Return itself with send args replaced. Used when kwargs are reordered/synthesized for callee.
    /// `original_argc` is the number of args originally on the stack (before processing).
    fn with_replaced_args(&self, new_args: &[InsnId], original_argc: usize) -> Self {
        let mut state = self.clone();
        let args_start = state.stack.len() - original_argc;
        state.stack.truncate(args_start);
        state.stack.extend_from_slice(new_args);
        state
    }

    fn replace(&mut self, old: InsnId, new: InsnId) {
        for slot in &mut self.stack {
            if *slot == old {
                *slot = new;
            }
        }
        for slot in &mut self.locals {
            if *slot == old {
                *slot = new;
            }
        }
    }
}

/// Print adaptor for [`FrameState`]. See [`PtrPrintMap`].
pub struct FrameStatePrinter<'a> {
    inner: &'a FrameState,
    ptr_map: &'a PtrPrintMap,
}

impl FrameState {
    fn new(iseq: IseqPtr) -> FrameState {
        FrameState { iseq, pc: std::ptr::null::<VALUE>(), insn_idx: 0, stack: vec![], locals: vec![], caller: None, depth: 0 }
    }

    /// Construct a `FrameState` for an inlined callee. `caller` is the `InsnId`
    /// of the caller's post-send `Snapshot`; `depth` is this frame's inlining depth.
    fn inlined(iseq: IseqPtr, caller: InsnId, depth: InlineDepth) -> FrameState {
        FrameState { caller: Some(caller), depth, ..FrameState::new(iseq) }
    }

    /// Get the number of stack operands
    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }

    /// Iterate over all stack slots
    pub fn stack(&self) -> Iter<'_, InsnId> {
        self.stack.iter()
    }

    pub fn caller(&self) -> Option<InsnId> {
        self.caller
    }

    /// Iterate over all local variables
    pub fn locals(&self) -> Iter<'_, InsnId> {
        self.locals.iter()
    }

    /// Push a stack operand
    fn stack_push(&mut self, opnd: InsnId) {
        self.stack.push(opnd);
    }

    /// Pop a stack operand
    fn stack_pop(&mut self) -> Result<InsnId, ParseError> {
        self.stack.pop().ok_or_else(|| ParseError::StackUnderflow(self.clone()))
    }

    fn stack_pop_n(&mut self, count: usize) -> Result<Vec<InsnId>, ParseError> {
        // Check if we have enough values on the stack
        let stack_len = self.stack.len();
        if stack_len < count {
            return Err(ParseError::StackUnderflow(self.clone()));
        }

        Ok(self.stack.split_off(stack_len - count))
    }

    /// Get a stack-top operand
    fn stack_top(&self) -> Result<InsnId, ParseError> {
        self.stack.last().ok_or_else(|| ParseError::StackUnderflow(self.clone())).copied()
    }

    /// Set a stack operand at idx
    fn stack_setn(&mut self, idx: usize, opnd: InsnId) {
        let idx = self.stack.len() - idx - 1;
        self.stack[idx] = opnd;
    }

    /// Get a stack operand at idx
    fn stack_topn(&self, idx: usize) -> Result<InsnId, ParseError> {
        let Some(idx) = self.stack.len().checked_sub(idx + 1) else {
            return Err(ParseError::StackUnderflow(self.clone()));
        };
        self.stack.get(idx).ok_or_else(|| ParseError::StackUnderflow(self.clone())).copied()
    }

    fn setlocal(&mut self, ep_offset: u32, opnd: InsnId) {
        let idx = ep_offset_to_local_idx(self.iseq, ep_offset);
        self.locals[idx] = opnd;
    }

    fn getlocal(&mut self, ep_offset: u32) -> InsnId {
        let idx = ep_offset_to_local_idx(self.iseq, ep_offset);
        self.locals[idx]
    }

    fn as_args(&self, self_param: InsnId) -> Vec<InsnId> {
        // We're currently passing around the self parameter as a basic block
        // argument because the register allocator uses a fixed register based
        // on the basic block argument index, which would cause a conflict if
        // we reuse an argument from another basic block.
        // TODO: Modify the register allocator to allow reusing an argument
        // of another basic block.
        let mut args = vec![self_param];
        args.extend(self.locals.iter().chain(self.stack.iter()).copied());
        args
    }

    /// Get the opcode for the current instruction
    pub fn get_opcode(&self) -> i32 {
        unsafe { rb_iseq_opcode_at_pc(self.iseq, self.pc) }
    }

    pub fn print<'a>(&'a self, ptr_map: &'a PtrPrintMap) -> FrameStatePrinter<'a> {
        FrameStatePrinter { inner: self, ptr_map }
    }
}

impl Display for FrameStatePrinter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let inner = self.inner;
        write!(f, "FrameState {{ pc: {:?}, stack: ", self.ptr_map.map_ptr(inner.pc))?;
        write_vec(f, &inner.stack)?;
        write!(f, ", locals: [")?;
        for (idx, local) in inner.locals.iter().enumerate() {
            let name: ID = unsafe { rb_zjit_local_id(inner.iseq, idx.try_into().unwrap()) };
            let name = name.contents_lossy();
            if idx > 0 { write!(f, ", ")?; }
            write!(f, "{name}={local}")?;
        }
        write!(f, "]")?;
        if let Some(caller) = inner.caller {
            write!(f, ", caller: {caller}")?;
        }
        write!(f, " }}")
    }
}

/// Get YARV instruction argument
fn get_arg(pc: *const VALUE, arg_idx: isize) -> VALUE {
    unsafe { *(pc.offset(arg_idx + 1)) }
}

/// Compute YARV instruction index at relative offset
fn insn_idx_at_offset(idx: u32, offset: i64) -> u32 {
    ((idx as isize) + (offset as isize)) as u32
}

struct BytecodeInfo {
    jump_targets: Vec<u32>,
}

/// The largest `opt_case_dispatch` hash we are willing to turn into an inline
/// binary search. Bigger `case`/`when` statements keep the `===` chain.
const MAX_CASE_DISPATCH_ENTRIES: usize = 512;

/// Read the `(key, jump offset)` pairs out of an `opt_case_dispatch` hash, sorted by key.
/// Returns None unless every key is a Fixnum, which is what lets us compile the dispatch
/// as an integer comparison tree guarded by a single `Integer#===` redefinition check.
fn cdhash_fixnum_entries(cdhash: VALUE) -> Option<Vec<(i64, i64)>> {
    let mut buf = vec![0 as std::os::raw::c_long; MAX_CASE_DISPATCH_ENTRIES * 2];
    let size = unsafe {
        rb_zjit_cdhash_fixnum_entries(cdhash, buf.as_mut_ptr(), MAX_CASE_DISPATCH_ENTRIES as std::os::raw::c_long)
    };
    if size <= 0 {
        return None;
    }
    let mut entries: Vec<(i64, i64)> = buf[..(size as usize) * 2]
        .chunks_exact(2)
        .map(|pair| (pair[0] as i64, pair[1] as i64))
        .collect();
    entries.sort_unstable_by_key(|&(key, _)| key);
    Some(entries)
}

fn compute_bytecode_info(iseq: *const rb_iseq_t, opt_table: &[u32]) -> BytecodeInfo {
    let iseq_size = unsafe { get_iseq_encoded_size(iseq) };
    let mut insn_idx = 0;
    let mut jump_targets: HashSet<u32> = opt_table.iter().copied().collect();
    while insn_idx < iseq_size {
        // Get the current pc and opcode
        let pc = unsafe { rb_iseq_pc_at_idx(iseq, insn_idx) };

        // Strip any ZJIT profiling instrumentation so we read the ISEQ's
        // original opcodes, mirroring the main translation loop. We map rather
        // than mutate the ISEQ because the caller must not dictate the profiling
        // policy of a callee it inlines. Only control-flow opcodes matter here
        // and those are never instrumented, but decoding both sites identically
        // keeps them from diverging.
        //
        // try_into() call below is unfortunate. Maybe pick i32 instead of usize for opcodes.
        let opcode: u32 = unsafe { rb_zjit_insn_to_bare_insn(rb_iseq_opcode_at_pc(iseq, pc)) }
            .try_into()
            .unwrap();
        insn_idx += insn_len(opcode as usize);
        match opcode {
            YARVINSN_branchunless | YARVINSN_jump | YARVINSN_branchif | YARVINSN_branchnil
            | YARVINSN_branchunless_without_ints | YARVINSN_jump_without_ints | YARVINSN_branchif_without_ints | YARVINSN_branchnil_without_ints => {
                let offset = get_arg(pc, 0).as_i64();
                jump_targets.insert(insn_idx_at_offset(insn_idx, offset));
            }
            YARVINSN_opt_new => {
                let offset = get_arg(pc, 1).as_i64();
                jump_targets.insert(insn_idx_at_offset(insn_idx, offset));
            }
            YARVINSN_opt_case_dispatch => {
                // The `when` bodies are already jump targets of the `===` chain that
                // follows, but the else offset is only reachable by fallthrough there.
                if let Some(entries) = cdhash_fixnum_entries(get_arg(pc, 0)) {
                    for (_, offset) in entries {
                        jump_targets.insert(insn_idx_at_offset(insn_idx, offset));
                    }
                    jump_targets.insert(insn_idx_at_offset(insn_idx, get_arg(pc, 1).as_i64()));
                }
            }
            YARVINSN_leave | YARVINSN_opt_invokebuiltin_delegate_leave => {
                if insn_idx < iseq_size {
                    jump_targets.insert(insn_idx);
                }
            }
            _ => {}
        }
    }
    let mut result = jump_targets.into_iter().collect::<Vec<_>>();
    result.sort();
    BytecodeInfo { jump_targets: result }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum CallType {
    Splat,
    Kwarg,
    Tailcall,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParseError {
    StackUnderflow(FrameState),
    MalformedIseq(u32), // insn_idx into iseq_encoded
    Validation(ValidationError),
    NotAllowed,
    DirectiveInduced,
}

/// Return the number of locals in the current ISEQ (includes parameters)
fn num_locals(iseq: *const rb_iseq_t) -> usize {
    (unsafe { get_iseq_body_local_table_size(iseq) }).to_usize()
}

/// The callee local slot each JIT-to-JIT call argument fills, in argument order.
/// Index 0 is `self`, which has no local slot and is reported as `None`; the rest
/// mirror the local walk in [`compile_jit_entry_state`].
fn jit_entry_arg_locals(iseq: IseqPtr, passed_opt_num: usize) -> Vec<Option<usize>> {
    let params = unsafe { iseq.params() };
    let param_size = params.size.to_usize();
    let opt_num: usize = params.opt_num.try_into().expect("iseq param opt_num >= 0");
    let lead_num: usize = params.lead_num.try_into().expect("iseq param lead_num >= 0");
    let kw_bits_idx: Option<usize> = if unsafe { rb_get_iseq_flags_has_kw(iseq) } {
        let keyword = unsafe { rb_get_iseq_body_param_keyword(iseq) };
        if keyword.is_null() { None } else { Some(unsafe { (*keyword).bits_start } as usize) }
    } else {
        None
    };

    let mut arg_locals = vec![None];
    for local_idx in 0..num_locals(iseq) {
        if (lead_num + passed_opt_num..lead_num + opt_num).contains(&local_idx) { continue; }
        if Some(local_idx) == kw_bits_idx { continue; }
        if local_idx >= param_size { continue; }
        arg_locals.push(Some(local_idx));
    }
    arg_locals
}

/// Number of declared keyword parameters on the callee, or zero if the
/// callee does not accept any keywords.
fn callee_kw_num(iseq: *const rb_iseq_t) -> usize {
    if unsafe { rb_get_iseq_flags_has_kw(iseq) } {
        let keyword = unsafe { rb_get_iseq_body_param_keyword(iseq) };
        if keyword.is_null() {
            0
        } else {
            unsafe { (*keyword).num as usize }
        }
    } else {
        0
    }
}

/// Local table index of the hidden `kw_bits` storage slot used by
/// `checkkeyword`, or `None` when the callee has no keyword parameters.
fn callee_kw_bits_local_idx(iseq: *const rb_iseq_t) -> Option<usize> {
    if !unsafe { rb_get_iseq_flags_has_kw(iseq) } {
        return None;
    }

    let keyword = unsafe { rb_get_iseq_body_param_keyword(iseq) };
    if keyword.is_null() {
        return None;
    }

    Some(unsafe { (*keyword).bits_start } as usize)
}

/// True if `splatarray`'s operand has only ever been an Array at this bytecode index, so the
/// conversion can be replaced with a type guard.
fn splat_operand_is_array(payload: &crate::payload::IseqPayload, insn_idx: YarvInsnIdx) -> bool {
    let Some(summary) = payload.profile.get_operand_types(insn_idx)
        .and_then(|types| types.first())
        .map(TypeDistributionSummary::new) else { return false };
    summary.is_monomorphic()
        && summary.buckets().first()
            .is_some_and(|&profiled| Type::from_profiled_type(profiled).is_subtype(types::ArrayExact))
}

/// If we can't handle the type of send (yet), bail out.
fn unhandled_call_type(flags: u32) -> Result<(), CallType> {
    if (flags & VM_CALL_TAILCALL) != 0 { return Err(CallType::Tailcall); }
    Ok(())
}

/// If a given call to a c func uses overly complex arguments, then we won't specialize.
fn unspecializable_c_call_type(flags: u32) -> bool {
    ((flags & VM_CALL_KWARG) != 0) ||
    unspecializable_call_type(flags)
}

/// If a given call uses overly complex arguments, then we won't specialize.
fn unspecializable_call_type(flags: u32) -> bool {
    ((flags & VM_CALL_ARGS_SPLAT) != 0) ||
    ((flags & VM_CALL_KW_SPLAT) != 0) ||
    ((flags & VM_CALL_ARGS_BLOCKARG) != 0) ||
    ((flags & VM_CALL_FORWARDING) != 0)
}

/// We have IseqPayload, which keeps track of HIR Types in the interpreter, but this is not useful
/// or correct to query from inside the optimizer. Instead, ProfileOracle maps interpreter-recorded
/// type information onto the caller Function's HIR values at each Snapshot that represents a
/// profiled bytecode instruction.
#[derive(Debug)]
struct ProfileOracle {
    /// types maps Snapshot InsnIds to profiled type information for the HIR operands visible at
    /// that Snapshot. Inlined callees are translated directly into the caller, so their profile
    /// entries are already in caller InsnId space and can be appended without remapping.
    types: HashMap<InsnId, Vec<(InsnId, TypeDistributionSummary)>>,
}

impl ProfileOracle {
    fn new() -> Self {
        Self { types: Default::default() }
    }

    /// Look up profile entries for a Snapshot. Returns None when no profile data was recorded for
    /// the bytecode instruction represented by that Snapshot.
    fn get(&self, state: InsnId) -> Option<&[(InsnId, TypeDistributionSummary)]> {
        self.types.get(&state).map(|v| v.as_slice())
    }

    /// Record `summary` as the profile of `insn` at the `dst` Snapshot. Used by polymorphic
    /// dispatch to give each refined arm a monomorphic view of the receiver.
    fn add_entry(&mut self, dst: InsnId, insn: InsnId, summary: TypeDistributionSummary) {
        self.types.entry(dst).or_default().push((insn, summary));
    }

    /// Map the interpreter-recorded types of the stack onto the HIR operands on our compile-time virtual stack.
    fn profile_stack(&mut self, snapshot: InsnId, state: &FrameState) {
        let iseq_insn_idx = state.insn_idx;
        let payload = get_or_create_iseq_payload(state.iseq);
        let Some(operand_types) = payload.profile.get_operand_types(iseq_insn_idx) else { return };
        let entry = self.types.entry(snapshot).or_default();
        // operand_types is always going to be <= stack size (otherwise it would have an underflow
        // at run-time) so use that to drive iteration.
        for (idx, insn_type_distribution) in operand_types.iter().rev().enumerate() {
            let insn = state.stack_topn(idx).expect("Unexpected stack underflow in profiling");
            entry.push((insn, TypeDistributionSummary::new(insn_type_distribution)))
        }
    }

    /// Map the interpreter-recorded types of self onto the HIR self
    fn profile_self(&mut self, snapshot: InsnId, state: &FrameState, self_param: InsnId) {
        let iseq_insn_idx = state.insn_idx;
        let payload = get_or_create_iseq_payload(state.iseq);
        let Some(operand_types) = payload.profile.get_operand_types(iseq_insn_idx) else { return };
        let entry = self.types.entry(snapshot).or_default();
        if operand_types.is_empty() {
           return;
        }
        let self_type_distribution = &operand_types[0];
        entry.push((self_param, TypeDistributionSummary::new(self_type_distribution)))
    }

    /// Append profile entries produced while translating an inlined callee. The callee was added
    /// directly to the caller Function, so Snapshot and operand InsnIds already refer to caller
    /// storage.
    fn append(&mut self, callee: &ProfileOracle) {
        for (snapshot, entries) in &callee.types {
            self.types.entry(*snapshot).or_default().extend(entries.iter().cloned());
        }
    }

    /// Copy the profile entries recorded for the `src` Snapshot to the `dst` Snapshot, minus the
    /// entries for `exclude` (chased through guards). Used by polymorphic dispatch, where each
    /// refined arm gets a fresh Snapshot: the receiver must resolve from its refined type rather
    /// than the polymorphic profile, but the other operands' profiles should remain visible so
    /// argument-profile-dependent specializations (e.g. Array#[]) still apply.
    ///
    /// When `recv_summary` is `Some`, the receiver's entry is replaced with that summary instead
    /// of being dropped: a branch that selected one profiled type substitutes a monomorphic
    /// summary (the refined type only carries the class, while the profiled type also carries the
    /// shape that attr_reader ivar loads need), and the fallthrough substitutes a megamorphic one.
    fn copy_entries_except(&mut self, src: InsnId, dst: InsnId, exclude: InsnId, fun: &Function, recv_summary: Option<TypeDistributionSummary>) {
        let exclude = fun.chase_insn(exclude);
        let mut filtered: Vec<_> = self.types.get(&src).map_or_else(Vec::new, |entries| {
            entries.iter()
                .filter(|(insn, _)| fun.chase_insn(*insn) != exclude)
                .cloned()
                .collect()
        });
        if let Some(summary) = recv_summary {
            filtered.push((exclude, summary));
        }
        if !filtered.is_empty() {
            self.types.insert(dst, filtered);
        }
    }

    /// Copy every profile entry recorded for the `src` Snapshot to the `dst` Snapshot. Used by
    /// `send` method-name dispatch, where each arm gets a fresh Snapshot but sees exactly the
    /// same receiver and arguments as the original call site.
    fn copy_entries(&mut self, src: InsnId, dst: InsnId) {
        if let Some(entries) = self.types.get(&src).cloned() {
            self.types.entry(dst).or_default().extend(entries);
        }
    }
}

/// Return the method names to build a `send`/`__send__` method-name dispatch on, most frequent
/// first. Empty when this is not a specializable `send` call site or the profile is unusable.
fn send_method_names(profile: &crate::profile::IseqProfile, cd: *const rb_call_data, insn_idx: YarvInsnIdx, argc: usize) -> Vec<VALUE> {
    if argc == 0 {
        return vec![];
    }
    let ci = unsafe { (*cd).ci };
    let mid = unsafe { vm_ci_mid(ci) };
    if mid != ID!(send) && mid != ID!(__send__) {
        return vec![];
    }
    // Keep this in sync with profile_send_method_name: anything else is rejected downstream.
    let flags = unsafe { vm_ci_flag(ci) };
    if flags & (VM_CALL_ARGS_SPLAT | VM_CALL_KWARG | VM_CALL_KW_SPLAT | VM_CALL_ARGS_BLOCKARG | VM_CALL_FORWARDING) != 0 {
        return vec![];
    }
    let Some(summary) = profile.get_send_method_names(insn_idx) else { return vec![] };
    // A megamorphic name distribution means the profiler ran out of buckets, so the arms we
    // could build would not cover the call site. Leave it as a dynamic send.
    if summary.is_megamorphic() || summary.is_skewed_megamorphic() {
        return vec![];
    }
    summary.buckets().iter()
        .take_while(|profiled_type| !profiled_type.is_empty())
        .map(|profiled_type| profiled_type.class())
        .collect()
}

fn invalidates_locals(opcode: u32, operands: *const VALUE) -> bool {
    match opcode {
        // Control-flow is non-leaf in the interpreter because it can execute arbitrary code on
        // interrupt. But in the JIT, we side-exit if there is a pending interrupt.
        YARVINSN_jump
        | YARVINSN_branchunless
        | YARVINSN_branchif
        | YARVINSN_branchnil
        | YARVINSN_jump_without_ints
        | YARVINSN_branchunless_without_ints
        | YARVINSN_branchif_without_ints
        | YARVINSN_branchnil_without_ints
        | YARVINSN_leave => false,
        // TODO(max): Read the invokebuiltin target from operands and determine if it's leaf
        _ => unsafe { !rb_zjit_insn_leaf(opcode as i32, operands) }
    }
}

/// The index of the self parameter in the HIR function
pub const SELF_PARAM_IDX: usize = 0;

/// The most elements a `foo(*args)` call site will read out of the splat array inline. Each
/// element costs an array load plus a stack write, so long splats stay on the dynamic path.
const MAX_SPLAT_EXPANSION: usize = 16;

/// Controls how an ISEQ's bytecode is added to HIR.
#[derive(Clone, Copy)]
enum AddIseqMode {
    Standalone,
    /// Like `Standalone`, but the interpreter entry resumes at a catch-table
    /// continuation in the middle of the ISEQ instead of an opt-table entry.
    /// Used to compile `body->jit_exception`. No JIT-to-JIT entry blocks are
    /// generated: other JIT code always calls the `Standalone` version.
    ExceptionEntry(ExceptionEntry),
    Inlined {
        return_block: BlockId,
        /// The caller's post-send `Snapshot`. Allows side-exits to restore the outer frame.
        caller: InsnId,
        /// Inlining depth of every frame emitted for the callee.
        depth: InlineDepth,
        /// The JIT entry index selected by the call site's argument count.
        jit_entry_idx: usize,
        /// The literal block the caller passed to this frame, if any.
        blockiseq: Option<IseqPtr>,
        /// Set when this inlined frame is a block ISEQ whose lexical owner is the
        /// compiled function's own frame, so a `return` inside it returns from the
        /// compiled function. The value is the number of inlined frames that are on
        /// the CFP stack when it runs, this block's own frame included.
        block_return_pops: Option<u32>,
    },
}

/// Result of populating a Function with HIR for an ISEQ.
struct AddIseqResult {
    /// The callee body entry block where the inlined callee body begins
    /// executing, populated only in inlined mode.
    ///
    /// `add_iseq_to_hir` slices the opt table to begin at the entry the call
    /// site's argument count selects. This is that entry's block.
    /// The value is `None` in standalone mode, as standalone has no single body entry
    /// block and wires its JIT entry blocks through `fun.jit_entry_blocks` instead.
    body_entry_block: Option<BlockId>,
    /// Profile oracle populated during compilation. The caller decides whether
    /// to assign it to `fun.profiles` (top-level) or append it to an existing
    /// oracle (inliner).
    profiles: ProfileOracle,
}

/// A sibling shape needs at least this fraction (1/N) of a dispatch arm's samples to earn an arm
/// of its own. See the filter in [`emit_polymorphic_send`].
const ARM_SHAPE_MIN_SHARE: u32 = 4;

/// Whether any receiver class this site profiled resolves the call to a method whose frame setup
/// `type_specialize` can hand a `&blk` block handler to. Only ISEQ and C methods get such a
/// frame; the rest keep the dynamic send whatever the block argument is.
fn profiled_recv_takes_block_handler(
    fun: &Function,
    profiles: &ProfileOracle,
    recv: InsnId,
    state: InsnId,
    cd: *const rb_call_data,
) -> bool {
    let mid = unsafe { vm_ci_mid((*cd).ci) };
    fun.profile_summary(profiles, recv, state).buckets().iter().any(|profiled_type| {
        if profiled_type.is_empty() { return false; }
        let mut cme = unsafe { rb_callable_method_entry(profiled_type.class(), mid) };
        if cme.is_null() { return false; }
        let mut def_type = unsafe { get_cme_def_type(cme) };
        while def_type == VM_METHOD_TYPE_ALIAS {
            cme = unsafe { rb_aliased_callable_method_entry(cme) };
            def_type = unsafe { get_cme_def_type(cme) };
        }
        matches!(def_type, VM_METHOD_TYPE_ISEQ | VM_METHOD_TYPE_CFUNC)
    })
}

/// Emit a receiver-type-specialized dispatch for a send: one `HasType` branch per profiled
/// receiver type, each with its own `Send` that `type_specialize` can turn into a direct call,
/// plus a dynamic-send fallthrough. All branches join on a single block parameter.
///
/// Returns the block to continue compiling in and the joined result, or `None` when the receiver
/// is not polymorphic, in which case the caller should emit a single `Send`.
fn emit_polymorphic_send(
    fun: &mut Function,
    profiles: &mut ProfileOracle,
    mut block: BlockId,
    insn_idx: u32,
    exit_id: InsnId,
    exit_state: &FrameState,
    cd: *const rb_call_data,
    recv: InsnId,
    args: &[InsnId],
    block_handler: Option<BlockHandler>,
    opcode: u32,
    branch_monomorphic: bool,
) -> Option<(BlockId, InsnId)> {
    // A monomorphic profile normally becomes a GuardType, which side-exits when the profile turns
    // out to be wrong. `branch_monomorphic` asks for a branch instead: callers use it where the
    // alternative is a dynamic send anyway, so a missed branch is no worse than not specializing,
    // whereas a failed guard is much worse.
    let plan = match branch_monomorphic.then(|| fun.profile_summary(profiles, recv, exit_id)) {
        Some(summary) if summary.is_monomorphic() => SendChainPlan::Classes(summary),
        _ => fun.send_chain_plan(profiles, recv, exit_id, cd)?,
    };
    let summary = match plan {
        // A site whose profiled classes all inherit one method entry dispatches on that method
        // instead: one ancestor check covers every subclass, including ones the profile never saw.
        SendChainPlan::Ancestor(dispatch) => return Some(gen_send_ancestor_chain(
            fun, profiles, &dispatch, block, insn_idx, exit_state, exit_id,
            recv, cd, block_handler, args.to_vec(), opcode,
        )),
        SendChainPlan::Classes(summary) => summary,
    };
    let join_block = fun.new_block(insn_idx);
    let join_param = fun.push_insn(join_block, Insn::Param);
    // Dedup by expected type so immediate/heap variants
    // under the same Ruby class can still get separate branches.
    let mut seen_types = Vec::with_capacity(summary.buckets().len());
    for &profiled_type in summary.buckets() {
        if profiled_type.is_empty() { break; }
        let expected = Type::from_profiled_type(profiled_type);
        if seen_types.iter().any(|ty: &Type| ty.bit_equal(expected)) {
            continue;
        }
        seen_types.push(expected);
        // The branch only tests the class, so the shape this bucket carries is not a prediction
        // for the values that reach it -- a sibling bucket for the same class has a different
        // one, and even a lone bucket only saw what the unrefined call site happened to profile.
        // Hand it over anyway, tagged as a polymorphic arm: consumers branch on a polymorphic
        // arm's shape rather than guarding it (see the VM_METHOD_TYPE_IVAR/ATTRSET cases in
        // type_specialize), so a receiver with another shape takes the C fallback instead of
        // side-exiting out of a version that never promised the shape. Dropping it instead would
        // leave the branch with only the refined class, and a class without a shape turns every
        // ivar read on it into an rb_ivar_get call.
        // Hand over *every* bucket for this class, not just the one that named the arm. The arm
        // is entered by any receiver of the class, so a sibling bucket's shape is just as likely
        // to show up as this one's, and an ivar dispatch that only knows one of them sends the
        // rest to rb_ivar_get.
        // Ivar dispatch inside the arm can specialize each of these shapes, and every one it
        // does not know sends its receivers to rb_ivar_get. Rare shapes are not worth an arm
        // though: the arms live in the caller and share its inline budget, so a dispatch that
        // grows for a shape almost nobody has can push a hot callee out of the budget and cost
        // more in dynamic sends than it saves in ivar reads.
        let class_samples: u32 = summary.buckets().iter().enumerate()
            .filter(|(_, other)| !other.is_empty() && Type::from_profiled_type(**other).bit_equal(expected))
            .map(|(idx, _)| u32::from(summary.bucket_count(idx)))
            .sum();
        let recv_profile: Vec<ProfiledType> = summary.buckets().iter().enumerate()
            .filter(|(idx, other)| {
                !other.is_empty()
                    && Type::from_profiled_type(**other).bit_equal(expected)
                    // The bucket that named the arm always earns its place; a sibling has to
                    // carry a real share of the class's traffic.
                    && (**other == profiled_type
                        || u32::from(summary.bucket_count(*idx)) * ARM_SHAPE_MIN_SHARE >= class_samples)
            })
            .map(|(_, other)| other.as_polymorphic_arm())
            .collect();
        let has_type = fun.push_insn(block, Insn::HasType { val: recv, expected });
        let iftrue_block = fun.new_block(insn_idx);
        let fall_through = fun.new_block(insn_idx);
        fun.push_insn(block, Insn::CondBranch {
            val: has_type,
            if_true: BranchEdge { target: iftrue_block, args: vec![] },
            if_false: BranchEdge { target: fall_through, args: vec![] }
        });
        block = fall_through;
        // Take a fresh Snapshot rather than reusing exit_id so type specialization resolves the
        // receiver from this branch's profile instead of the polymorphic one keyed at exit_id.
        let snapshot = fun.push_insn(iftrue_block, Insn::Snapshot { state: Box::new(exit_state.clone()) });
        // Keep the other operands' profile entries visible at the fresh Snapshot so the
        // specialized send can still see argument profiles (e.g. Array#[] needs a Fixnum-profiled
        // index to be inlined). The receiver's polymorphic entry is replaced by the single
        // profiled type this branch selects, so specializations that need the shape (attr_reader
        // ivar loads) can still use it.
        profiles.copy_entries_except(exit_id, snapshot, recv, fun, Some(TypeDistributionSummary::monomorphic_variants(&recv_profile)));
        let refined_recv = fun.push_insn(iftrue_block, Insn::RefineType { val: recv, new_type: expected });
        fun.record_profiled_type(refined_recv, recv_profile[0]);
        let send = fun.push_insn(iftrue_block, Insn::Send { recv: refined_recv, cd, block: block_handler, args: args.to_vec(), state: snapshot, reason: Uncategorized(opcode.into()) });
        fun.push_insn(iftrue_block, Insn::Jump(BranchEdge { target: join_block, args: vec![send] }));
    }
    // In the fallthrough case, do a generic interpreter send and then join. Give it a Snapshot
    // whose receiver entry is megamorphic: the branches above already cover every profiled type,
    // so anything reaching here is a type the profile never saw. Without this, type_specialize
    // resolves the receiver from the original profile and re-speculates on a type this path has
    // just ruled out, and that guard then fails on every single call.
    let fallback_snapshot = fun.push_insn(block, Insn::Snapshot { state: Box::new(exit_state.clone()) });
    profiles.copy_entries_except(exit_id, fallback_snapshot, recv, fun, Some(TypeDistributionSummary::megamorphic()));
    let send = fun.push_insn(block, Insn::Send { recv, cd, block: block_handler, args: args.to_vec(), state: fallback_snapshot, reason: SendPolymorphicFallback });
    fun.push_insn(block, Insn::Jump(BranchEdge { target: join_block, args: vec![send] }));
    Some((join_block, join_param))
}

/// Compile ISEQ into High-level IR
pub fn iseq_to_hir(iseq: *const rb_iseq_t) -> Result<Function, ParseError> {
    iseq_to_hir_with_mode(iseq, AddIseqMode::Standalone)
}

/// Compile ISEQ into High-level IR with an entry at a catch-table continuation.
/// Used for `body->jit_exception`; see [`ExceptionEntry`].
pub fn iseq_to_hir_exception_entry(iseq: *const rb_iseq_t, entry: ExceptionEntry) -> Result<Function, ParseError> {
    iseq_to_hir_with_mode(iseq, AddIseqMode::ExceptionEntry(entry))
}

fn iseq_to_hir_with_mode(iseq: *const rb_iseq_t, mode: AddIseqMode) -> Result<Function, ParseError> {
    if !ZJITState::can_compile_iseq(iseq) {
        return Err(ParseError::NotAllowed);
    }
    let payload = get_or_create_iseq_payload(iseq);
    let mut fun = Function::new(iseq);
    fun.was_invalidated_for_singleton_class_creation = payload.was_invalidated_for_singleton_class_creation;
    fun.self_is_heap_object = payload.self_is_heap_object;
    if let AddIseqMode::ExceptionEntry(entry) = mode {
        fun.exception_entry = Some(entry);
    }

    let result = add_iseq_to_hir(&mut fun, iseq, mode)?;
    fun.profiles = Some(result.profiles);

    if let Err(err) = crate::stats::trace_compile_phase("validate", || fun.validate()) {
        debug!("ZJIT: {err:?}: Initial HIR:\n{}", FunctionPrinter::without_snapshot(&fun));
        return Err(ParseError::Validation(err));
    }
    Ok(fun)
}

/// Populate `fun` with HIR translated from `iseq`. Used both for top-level
/// compilation (`iseq_to_hir`) and for inlining an ISEQ directly into a caller
/// (the method inliner).
///
/// When `mode` is `AddIseqMode::Standalone`, generate the interpreter entry
/// block and, for ISEQs that can be entered by JIT-to-JIT calls, a JIT entry
/// block for each opt-table entry, push the JIT entry
/// blocks onto `fun.jit_entry_blocks`, and run the post-translation passes
/// (`seal_entries`, `set_param_types`, `infer_types`) before returning. When
/// `mode` is `AddIseqMode::Inlined`, only the body blocks are produced and the
/// post-translation passes are skipped; the caller is expected to run them once
/// translation of all constituent ISEQs is complete.
///
/// In inlined mode, `YARVINSN_leave` emits a `Jump(return_block, [retval])`
/// instead of `Insn::Return { val }`. The inliner uses this to wire the
/// callee's return paths directly to a continuation block in the caller,
/// avoiding a second rewrite pass.
fn add_iseq_to_hir(
    fun: &mut Function,
    iseq: *const rb_iseq_t,
    mode: AddIseqMode,
) -> Result<AddIseqResult, ParseError> {
    let payload = get_or_create_iseq_payload(iseq);
    let mut profiles = ProfileOracle::new();

    // In a final version there are no recompiles left, so a receiver guard that turns out to be
    // wrong side-exits on every call forever. Branch on the type instead: a missed branch costs a
    // dynamic send, which is what the site would have done without the speculation anyway.
    let branch_monomorphic_sends = fun.policy.no_side_exits;

    // Build the initial FrameState for a block being translated. In inlined
    // mode it carries the caller's post-send Snapshot and this frame's depth;
    // because every Snapshot emitted for the callee is cloned from one of these
    // initial states, those values propagate to the whole inlined body without a
    // separate rewrite pass.
    fn new_frame_state(mode: AddIseqMode, iseq: IseqPtr) -> FrameState {
        match mode {
            AddIseqMode::Inlined { caller, depth, .. } => FrameState::inlined(iseq, caller, depth),
            AddIseqMode::Standalone | AddIseqMode::ExceptionEntry(_) => FrameState::new(iseq),
        }
    }

    // Compute a map of PC->Block by finding jump targets.
    //
    // Standalone compilation translates every opt-table entry because each is a
    // reachable JIT-to-JIT entrypoint. The inliner, by contrast, enters at a
    // single entry fixed by the call site's argument count. So, the entries before
    // it would run default-init code for optionals the caller already supplied.
    // Those entries are known to be unreachable so slicing them off here avoids
    // translating prologue blocks that would only be discarded later, rather
    // than emitting them and relying on a downstream pass to prune the dead CFG.
    //
    // An exception-handler entry has exactly one entry PC, the catch-table
    // continuation the interpreter wants to resume at, which is unrelated to the
    // opt table.
    let jit_entry_start = match mode {
        AddIseqMode::Standalone | AddIseqMode::ExceptionEntry(_) => 0,
        AddIseqMode::Inlined { jit_entry_idx, .. } => jit_entry_idx,
    };
    let jit_entry_insns = match mode {
        AddIseqMode::ExceptionEntry(entry) => vec![entry.insn_idx],
        _ => unsafe { iseq.params() }.opt_table_slice()
            .get(jit_entry_start..)
            .expect("JIT entry index must be within the callee opt table")
            .iter().copied().map(VALUE::as_u32).collect::<Vec<_>>(),
    };
    let BytecodeInfo { jump_targets } = compute_bytecode_info(iseq, &jit_entry_insns);

    let compile_jit_entries = matches!(mode, AddIseqMode::Standalone) && iseq_supports_jit_entry(iseq);

    // Make all empty basic blocks. The ordering of the BBs matters for getting fallthrough jumps
    // in good places, but it's not necessary for correctness. TODO: Higher quality scheduling during lowering.
    let mut insn_idx_to_block = HashMap::default();
    // Materialize a block at each opt-table entry PC, placing each right after
    // its JIT-to-JIT entry block so fallthrough jumps land in good places.
    // Standalone mode emits a real JIT entry block per entry. Inlined mode emits
    // none here; only `body_entry_block` (the first sliced entry) is the inlined
    // body's entry, and the rest are default-init body blocks reached by
    // fallthrough.
    for insn_idx in jit_entry_insns.iter().copied() {
        if compile_jit_entries {
            let jit_entry_block = fun.new_block(insn_idx);
            fun.jit_entry_blocks.push(jit_entry_block);
        }
        insn_idx_to_block.entry(insn_idx).or_insert_with(|| fun.new_block(insn_idx));
    }
    // Make blocks for the rest of the jump targets
    for insn_idx in jump_targets {
        insn_idx_to_block.entry(insn_idx).or_insert_with(|| fun.new_block(insn_idx));
    }
    // Done, drop `mut`.
    let insn_idx_to_block = insn_idx_to_block;

    // The callee body entry block where the inlined callee body begins
    // executing. `jit_entry_insns` starts at the entry the call site selected
    // (since `opt_table_slice` was sliced above), so its first element is that entry.
    // Any later entries are the default-init blocks for the remaining unfilled
    // optionals, reached from this one by fallthrough rather than targeted directly.
    // Standalone compilation has no single body entry block, so it produces none.
    let body_entry_block = match mode {
        AddIseqMode::Standalone | AddIseqMode::ExceptionEntry(_) => None,
        AddIseqMode::Inlined { .. } => Some(insn_idx_to_block[&jit_entry_insns[0]]),
    };

    // The state the exception entry block loads from the CFP, threaded into the
    // queue below so the continuation block gets stack parameters for the VM
    // stack slots that were live at the continuation.
    let mut exception_entry_state = None;
    if let AddIseqMode::ExceptionEntry(entry) = mode {
        let target_block = insn_idx_to_block[&entry.insn_idx];
        exception_entry_state = Some(compile_exception_entry_block(fun, entry, target_block));
    }

    if matches!(mode, AddIseqMode::Standalone) {
        // Compile an entry_block for the interpreter
        compile_entry_block(fun, jit_entry_insns.as_slice(), &insn_idx_to_block);

        if compile_jit_entries {
            // Compile all JIT-to-JIT entry blocks
            for (jit_entry_idx, insn_idx) in jit_entry_insns.iter().enumerate() {
                let target_block = insn_idx_to_block.get(insn_idx)
                    .copied()
                    .expect("we make a block for each jump target and \
                             each entry in the ISEQ opt_table is a jump target");
                compile_jit_entry_block(fun, jit_entry_idx, target_block);
            }
        }
    }

    // Check if the EP is escaped for the ISEQ from the beginning. We give up
    // optimizing locals in that case because they're shared with other frames.
    let ep_starts_escaped = iseq_ep_starts_escaped(iseq);
    // Check if the EP has been escaped at some point in the ISEQ. If it has, then we assume that
    // its EP is shared with other frames.
    let seen_ep_escape = iseq_seen_ep_escape(iseq);
    let ep_escaped = ep_starts_escaped || seen_ep_escape;

    // Values `getblockparamproxy` pushed for this ISEQ's own local EP, i.e. the ones a
    // `foo(&blk)` site can pass on as this frame's block handler. See the use in
    // `YARVINSN_send`.
    let mut block_param_proxy_values: HashSet<InsnId> = HashSet::default();

    // Iteratively fill out basic blocks using a queue.
    // TODO(max): Basic block arguments at edges
    let mut queue = VecDeque::new();
    for &insn_idx in jit_entry_insns.iter() {
        // The exception entry seeds the state the entry block loaded, so that the
        // continuation block is given as many stack parameters as there are live
        // VM stack slots at the continuation.
        let state = exception_entry_state.take().unwrap_or_else(|| new_frame_state(mode, iseq));
        queue.push_back((state, insn_idx_to_block[&insn_idx], /*insn_idx=*/insn_idx, /*local_inval=*/false));
    }

    // Keep compiling blocks until the queue becomes empty
    let mut visited = HashSet::default();
    let iseq_size = unsafe { get_iseq_encoded_size(iseq) };
    // The pre-instruction frame state, hoisted out of both loops so that
    // `clone_from` below refills its `stack` and `locals` buffers instead of
    // allocating a fresh pair for every YARV instruction in the ISEQ.
    let mut exit_state = new_frame_state(mode, iseq);
    while let Some((incoming_state, mut block, mut insn_idx, mut local_inval)) = queue.pop_front() {
        // Compile each block only once
        if visited.contains(&block) { continue; }
        visited.insert(block);

        // Load basic block params first
        let mut self_param = fun.push_insn(block, Insn::Param);
        let mut state = {
            let mut result = new_frame_state(mode, iseq);
            let local_size = if jit_entry_insns.contains(&insn_idx) { num_locals(iseq) } else { incoming_state.locals.len() };
            for _ in 0..local_size {
                result.locals.push(fun.push_insn(block, Insn::Param));
            }
            for _ in incoming_state.stack {
                result.stack.push(fun.push_insn(block, Insn::Param));
            }
            result
        };

        // Start the block off with a Snapshot so that if we need to insert a new Guard later on
        // and we don't have a Snapshot handy, we can just iterate backward (at the earliest, to
        // the beginning of the block).
        fun.push_insn(block, Insn::Snapshot { state: Box::new(state.clone()) });
        while insn_idx < iseq_size {
            state.insn_idx = insn_idx as usize;
            // Get the current pc and opcode
            let pc = unsafe { rb_iseq_pc_at_idx(iseq, insn_idx) };
            state.pc = pc;
            exit_state.clone_from(&state);

            // Strip any ZJIT profiling instrumentation so we read the ISEQ's original opcodes,
            // leaving trace variants intact for the handling below.
            // try_into() call below is unfortunate. Maybe pick i32 instead of usize for opcodes.
            let opcode: u32 = unsafe { rb_zjit_insn_to_bare_insn(rb_iseq_opcode_at_pc(iseq, pc)) }
                .try_into()
                .unwrap();

            // We add NoTracePoint patch points before every instruction that could be affected by TracePoint.
            // This ensures that if TracePoint is enabled, we can exit the generated code as fast as possible.
            unsafe extern "C" {
                fn rb_iseq_event_flags(iseq: IseqPtr, pos: usize) -> rb_event_flag_t;
            }
            let exit_id = fun.push_insn(block, Insn::Snapshot { state: Box::new(exit_state.clone()) });

            // If TracePoint has been enabled after we have collected profiles, we'll see
            // trace_getinstancevariable in the ISEQ. We have to treat it like getinstancevariable
            // for profiling purposes: there is no operand on the stack to look up; we have
            // profiled cfp->self.
            if opcode == YARVINSN_getinstancevariable || opcode == YARVINSN_trace_getinstancevariable {
                profiles.profile_self(exit_id, &exit_state, self_param);
            } else if opcode == YARVINSN_setinstancevariable || opcode == YARVINSN_trace_setinstancevariable {
                profiles.profile_self(exit_id, &exit_state, self_param);
            } else if opcode == YARVINSN_definedivar || opcode == YARVINSN_trace_definedivar {
                profiles.profile_self(exit_id, &exit_state, self_param);
            } else if opcode == YARVINSN_invokeblock || opcode == YARVINSN_trace_invokeblock {
                if get_option!(stats) {
                    let iseq_insn_idx = exit_state.insn_idx;
                    if let Some(operand_types) = payload.profile.get_operand_types(iseq_insn_idx) {
                        if let [self_type_distribution] = &operand_types[..] {
                            let summary = TypeDistributionSummary::new(&self_type_distribution);
                            if summary.is_monomorphic() {
                                let profiled_type = summary.bucket(0);
                                let obj = profiled_type.class();
                                if profiled_type.is_block_ifunc() {
                                    fun.count(block, Counter::invokeblock_handler_monomorphic_ifunc);
                                } else if unsafe { rb_IMEMO_TYPE_P(obj, imemo_iseq) == 1 } {
                                    fun.count(block, Counter::invokeblock_handler_monomorphic_iseq);
                                } else {
                                    fun.count(block, Counter::invokeblock_handler_monomorphic_other);
                                }
                            } else if summary.is_skewed_polymorphic() || summary.is_polymorphic() {
                                if summary.buckets().iter().any(|ty| ty.is_block_ifunc()) {
                                    fun.count(block, Counter::invokeblock_handler_polymorphic_ifunc);
                                } else {
                                    fun.count(block, Counter::invokeblock_handler_polymorphic);
                                }
                            } else if summary.is_skewed_megamorphic() || summary.is_megamorphic() {
                                if summary.buckets().iter().any(|ty| ty.is_block_ifunc()) {
                                    fun.count(block, Counter::invokeblock_handler_megamorphic_ifunc);
                                } else {
                                    fun.count(block, Counter::invokeblock_handler_megamorphic);
                                }
                            } else {
                                fun.count(block, Counter::invokeblock_handler_no_profiles);
                            }
                        } else {
                            fun.count(block, Counter::invokeblock_handler_no_profiles);
                        }
                    }
                }
            } else if opcode == YARVINSN_getblockparamproxy || opcode == YARVINSN_trace_getblockparamproxy {
                if get_option!(stats) {
                    let iseq_insn_idx = exit_state.insn_idx;
                    if let Some([block_handler_distribution]) = payload.profile.get_operand_types(iseq_insn_idx) {
                        let summary = TypeDistributionSummary::new(block_handler_distribution);

                        if summary.is_monomorphic() {
                            let obj = summary.bucket(0).class();
                            if unsafe { rb_IMEMO_TYPE_P(obj, imemo_iseq) == 1} {
                                fun.count(block, Counter::getblockparamproxy_handler_iseq);
                            } else if unsafe { rb_IMEMO_TYPE_P(obj, imemo_ifunc) == 1} {
                                fun.count(block, Counter::getblockparamproxy_handler_ifunc);
                            }
                            else if obj.nil_p() {
                                fun.count(block, Counter::getblockparamproxy_handler_nil);
                            }
                            else if obj.symbol_p() {
                                fun.count(block, Counter::getblockparamproxy_handler_symbol);
                            } else if unsafe { rb_obj_is_proc(obj).test() } {
                                fun.count(block, Counter::getblockparamproxy_handler_proc);
                            }
                        } else if summary.is_polymorphic() || summary.is_skewed_polymorphic() {
                          fun.count(block, Counter::getblockparamproxy_handler_polymorphic);
                        } else if summary.is_megamorphic() || summary.is_skewed_megamorphic() {
                          fun.count(block, Counter::getblockparamproxy_handler_megamorphic);
                        }
                    } else {
                        fun.count(block, Counter::getblockparamproxy_handler_no_profiles);
                    }
                }
            }
            else {
                profiles.profile_stack(exit_id, &exit_state);
            }

            // Flag a future getlocal/setlocal to add a patch point if this instruction is not leaf.
            if invalidates_locals(opcode, unsafe { pc.offset(1) }) {
                local_inval = true;
            }

            if unsafe { rb_iseq_event_flags(iseq, insn_idx as usize) } != 0 {
                fun.push_insn(block, Insn::PatchPoint { invariant: Invariant::NoTracePoint, state: exit_id });
            }

            // Increment zjit_insn_count for each YARV instruction if --zjit-stats is enabled.
            if get_option!(stats) {
                fun.push_insn(block, Insn::IncrCounter(Counter::zjit_insn_count));
            }
            // Move to the next instruction to compile
            insn_idx += insn_len(opcode as usize);

            match opcode {
                YARVINSN_nop => {},
                YARVINSN_putnil => { state.stack_push(fun.push_insn(block, Insn::Const { val: Const::Value(Qnil) })); },
                YARVINSN_putobject => { state.stack_push(fun.push_insn(block, Insn::Const { val: Const::Value(get_arg(pc, 0)) })); },
                YARVINSN_putspecialobject => {
                    let value_type = SpecialObjectType::from(get_arg(pc, 0).as_u32());
                    let insn = if value_type == SpecialObjectType::VMCore {
                        Insn::Const { val: Const::Value(unsafe { rb_mRubyVMFrozenCore }) }
                    } else {
                        Insn::PutSpecialObject { value_type, state: exit_id }
                    };
                    state.stack_push(fun.push_insn(block, insn));
                }
                YARVINSN_dupstring => {
                    let val = fun.push_insn(block, Insn::Const { val: Const::Value(get_arg(pc, 0)) });
                    let insn_id = fun.push_insn(block, Insn::StringCopy { val, chilled: false, state: exit_id });
                    state.stack_push(insn_id);
                }
                YARVINSN_dupchilledstring => {
                    let val = fun.push_insn(block, Insn::Const { val: Const::Value(get_arg(pc, 0)) });
                    let insn_id = fun.push_insn(block, Insn::StringCopy { val, chilled: true, state: exit_id });
                    state.stack_push(insn_id);
                }
                YARVINSN_putself => { state.stack_push(self_param); }
                YARVINSN_intern => {
                    let val = state.stack_pop()?;
                    let insn_id = fun.push_insn(block, Insn::StringIntern { val, state: exit_id });
                    state.stack_push(insn_id);
                }
                YARVINSN_concatstrings => {
                    let count = get_arg(pc, 0).as_u32();
                    debug_assert!(count > 0, "concatstrings should have arguments");
                    let strings = state.stack_pop_n(count as usize)?;
                    let insn_id = fun.push_insn(block, Insn::StringConcat { strings, state: exit_id });
                    state.stack_push(insn_id);
                }
                YARVINSN_toregexp => {
                    // First arg contains the options (multiline, extended, ignorecase) used to create the regexp
                    let opt = get_arg(pc, 0).as_usize();
                    let count = get_arg(pc, 1).as_usize();
                    let values = state.stack_pop_n(count)?;
                    let insn_id = fun.push_insn(block, Insn::ToRegexp { opt, values, state: exit_id });
                    state.stack_push(insn_id);
                }
                YARVINSN_once => {
                    // `once` runs the body ISEQ the first time it is reached and caches the
                    // result in the inline storage entry, e.g. for `/#{...}/o` literals.
                    let body_iseq = get_arg(pc, 0).as_iseq();
                    let ise = get_arg(pc, 1).as_ptr();
                    let insn_id = fun.push_insn(block, Insn::Once { body_iseq, ise, state: exit_id });
                    state.stack_push(insn_id);
                }
                YARVINSN_newarray => {
                    let count = get_arg(pc, 0).as_usize();
                    let elements = state.stack_pop_n(count)?;
                    state.stack_push(fun.push_insn(block, Insn::NewArray { elements, state: exit_id }));
                }
                YARVINSN_opt_newarray_send => {
                    let count = get_arg(pc, 0).as_usize();
                    let method = get_arg(pc, 1).as_u32();
                    let (bop, insn) = match method {
                        VM_OPT_NEWARRAY_SEND_MAX => {
                            let elements = state.stack_pop_n(count)?;
                            (BOP_MAX, Insn::ArrayMax { elements, state: exit_id })
                        }
                        VM_OPT_NEWARRAY_SEND_MIN => {
                            let elements = state.stack_pop_n(count)?;
                            (BOP_MIN, Insn::ArrayMin { elements, state: exit_id })
                        }
                        VM_OPT_NEWARRAY_SEND_HASH => {
                            let elements = state.stack_pop_n(count)?;
                            (BOP_HASH, Insn::ArrayHash { elements, state: exit_id })
                        }
                        VM_OPT_NEWARRAY_SEND_INCLUDE_P => {
                            let target = state.stack_pop()?;
                            let elements = state.stack_pop_n(count - 1)?;
                            (BOP_INCLUDE_P, Insn::ArrayInclude { elements, target, state: exit_id })
                        }
                        VM_OPT_NEWARRAY_SEND_PACK => {
                            let fmt = state.stack_pop()?;
                            let elements = state.stack_pop_n(count - 1)?;
                            (BOP_PACK, Insn::ArrayPackBuffer { elements, fmt, buffer: None, state: exit_id })
                        }
                        VM_OPT_NEWARRAY_SEND_PACK_BUFFER => {
                            let buffer = state.stack_pop()?;
                            let fmt = state.stack_pop()?;
                            let elements = state.stack_pop_n(count - 2)?;
                            (BOP_PACK, Insn::ArrayPackBuffer { elements, fmt, buffer: Some(buffer), state: exit_id })
                        }
                        _ => {
                            // Unknown opcode; side-exit into the interpreter
                            fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledNewarraySend(method)), recompile: None });
                            break;  // End the block
                        }
                    };
                    if !fun.guard_bop_not_redefined(block, ARRAY_REDEFINED_OP_FLAG, bop, exit_id) {
                        // If the basic operation is already redefined, we cannot optimize it.
                        break;  // End the block
                    }
                    state.stack_push(fun.push_insn(block, insn));
                }
                YARVINSN_duparray => {
                    let val = fun.push_insn(block, Insn::Const { val: Const::Value(get_arg(pc, 0)) });
                    let insn_id = fun.push_insn(block, Insn::ArrayDup { val, state: exit_id });
                    state.stack_push(insn_id);
                }
                YARVINSN_opt_duparray_send => {
                    let ary = get_arg(pc, 0);
                    let method_id = get_arg(pc, 1).as_u64();
                    let argc = get_arg(pc, 2).as_usize();
                    if argc != 1 {
                        break;
                    }
                    let target = state.stack_pop()?;
                    let bop = match method_id {
                        x if x == ID!(include_p).0 => BOP_INCLUDE_P,
                        _ => {
                            fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledDuparraySend(method_id)), recompile: None });
                            break;
                        },
                    };
                    if !fun.guard_bop_not_redefined(block, ARRAY_REDEFINED_OP_FLAG, bop, exit_id) {
                        break;  // End the block
                    }
                    let insn_id = fun.push_insn(block, Insn::DupArrayInclude { ary, target, state: exit_id });
                    state.stack_push(insn_id);
                }
                YARVINSN_newhash => {
                    let count = get_arg(pc, 0).as_usize();
                    assert!(count % 2 == 0, "newhash count should be even");
                    let mut elements = vec![];
                    for _ in 0..(count/2) {
                        let value = state.stack_pop()?;
                        let key = state.stack_pop()?;
                        elements.push(value);
                        elements.push(key);
                    }
                    elements.reverse();
                    state.stack_push(fun.push_insn(block, Insn::NewHash { elements, state: exit_id }));
                }
                YARVINSN_duphash => {
                    let val = fun.push_insn(block, Insn::Const { val: Const::Value(get_arg(pc, 0)) });
                    let insn_id = fun.push_insn(block, Insn::HashDup { val, state: exit_id });
                    state.stack_push(insn_id);
                }
                YARVINSN_splatarray => {
                    let flag = get_arg(pc, 0);
                    let result_must_be_mutable = flag.test();
                    let val = state.stack_pop()?;
                    let obj = if result_must_be_mutable {
                        fun.push_insn(block, Insn::ToNewArray { val, state: exit_id })
                    } else if splat_operand_is_array(payload, exit_state.insn_idx) {
                        // `splatarray false` hands back its operand untouched when it is already
                        // an Array (see vm_splat_array), so a type guard replaces the call.
                        fun.push_insn(block, Insn::GuardType { val, guard_type: types::ArrayExact, state: exit_id, recompile: Some(Recompile) })
                    } else {
                        fun.push_insn(block, Insn::ToArray { val, state: exit_id })
                    };
                    state.stack_push(obj);
                }
                YARVINSN_splatkw => {
                    let block_val = state.stack_pop()?;
                    let hash = state.stack_pop()?;
                    // Get profiled type of hash (operand index 0)
                    let summary = payload.profile.get_operand_types(exit_state.insn_idx)
                        .and_then(|types| types.first())
                        .map(|dist| TypeDistributionSummary::new(dist));
                    // Guard for one shape only when the profile says the site sticks to it.
                    // Otherwise fall back to the generic conversion rather than side-exiting:
                    // a side exit here ends the block, so everything after a `**opts` call in
                    // the method would go uncompiled. `**opts` alternating between nil and a
                    // Hash is common in Rails code, so that costs a lot of coverage.
                    let monomorphic_ty = summary.as_ref()
                        .filter(|summary| summary.is_monomorphic())
                        .map(|summary| Type::from_profiled_type(summary.bucket(0)));
                    let obj = match monomorphic_ty {
                        Some(ty) if ty.is_subtype(types::NilClass) =>
                            fun.push_insn(block, Insn::GuardType { val: hash, guard_type: types::NilClass, state: exit_id, recompile: None }),
                        Some(ty) if ty.is_subtype(types::HashExact) =>
                            fun.push_insn(block, Insn::GuardType { val: hash, guard_type: types::HashExact, state: exit_id, recompile: None }),
                        _ => fun.push_insn(block, Insn::ToHash { val: hash, state: exit_id }),
                    };
                    state.stack_push(obj);
                    state.stack_push(block_val);
                }
                YARVINSN_concattoarray => {
                    let right = state.stack_pop()?;
                    let left = state.stack_pop()?;
                    let right_array = fun.push_insn(block, Insn::ToArray { val: right, state: exit_id });
                    fun.push_insn(block, Insn::ArrayExtend { left, right: right_array, state: exit_id });
                    state.stack_push(left);
                }
                YARVINSN_pushtoarray => {
                    let count = get_arg(pc, 0).as_usize();
                    let vals = state.stack_pop_n(count)?;
                    let array = state.stack_pop()?;
                    fun.guard_not_frozen(block, array, exit_id);
                    for val in vals.into_iter() {
                        fun.push_insn(block, Insn::ArrayPush { array, val, state: exit_id });
                    }
                    state.stack_push(array);
                }
                YARVINSN_putobject_INT2FIX_0_ => {
                    state.stack_push(fun.push_insn(block, Insn::Const { val: Const::Value(VALUE::fixnum_from_usize(0)) }));
                }
                YARVINSN_putobject_INT2FIX_1_ => {
                    state.stack_push(fun.push_insn(block, Insn::Const { val: Const::Value(VALUE::fixnum_from_usize(1)) }));
                }
                YARVINSN_defined => {
                    // (rb_num_t op_type, VALUE obj, VALUE pushval)
                    let op_type: defined_type = get_arg(pc, 0).as_usize().try_into().unwrap();
                    let obj = get_arg(pc, 1);
                    let pushval = get_arg(pc, 2);
                    let v = state.stack_pop()?;
                    let local_iseq = unsafe { rb_get_iseq_body_local_iseq(iseq) };
                    let insn = if op_type == DEFINED_YIELD && unsafe { rb_get_iseq_body_type(local_iseq) } != ISEQ_TYPE_METHOD {
                        // `yield` goes to the block handler stowed in the "local" iseq which is
                        // the current iseq or a parent. Only the "method" iseq type can be passed a
                        // block handler. (e.g. `yield` in the top level script is a syntax error.)
                        //
                        // Similar to gen_is_block_given
                        Insn::Const { val: Const::Value(Qnil) }
                    } else {
                        if op_type == DEFINED_YIELD && matches!(mode, AddIseqMode::Inlined { .. }) {
                            // If we are inlining a method that has a blockiseq handler, we can fold Defined(DEFINED_YIELD).
                            // TODO(max): If we handle non-blockiseq block arguments such as
                            // &:symbol or just &block forwarding, we need to revisit this and
                            // check flags.
                            let has_block = matches!(mode, AddIseqMode::Inlined { blockiseq: Some(_), .. });
                            if has_block {
                                Insn::Const { val: Const::Value(pushval) }
                            } else {
                                Insn::Const { val: Const::Value(Qnil) }
                            }
                        } else {
                            // For DEFINED_YIELD, codegen materializes the local EP inline (similar to
                            // gen_is_block_given) to check for a block handler. Precompute the lexical
                            // distance from this iseq up to local_iseq so codegen does not have to
                            // walk the parent chain. Any DEFINED_YIELD reaching this branch has a
                            // method local_iseq by construction -- the above branch has already
                            // diverted the non-method case to Qnil.
                            let lep_level = if op_type == DEFINED_YIELD {
                                get_lvar_level(iseq)
                            } else {
                                0
                            };
                            Insn::Defined { op_type, obj, pushval, v, lep_level, state: exit_id }
                        }
                    };
                    state.stack_push(fun.push_insn(block, insn));
                }
                YARVINSN_definedivar => {
                    // (ID id, IVC ic, VALUE pushval)
                    let id = ID(get_arg(pc, 0).as_u64());
                    let pushval = get_arg(pc, 2);
                    fn can_optimize(profiled_type: ProfiledType) -> Option<ShapeId> {
                        // Runtime immediates cannot pass the HeapBasicObject guard, so don't
                        // generate unreachable shape branches for profiled immediate buckets.
                        if profiled_type.flags().is_immediate() { return None; }
                        // Class/module/T_DATA ivars use different storage rules.
                        // Let the fallthrough DefinedIvar handle these.
                        if !profiled_type.flags().is_t_object() { return None; }
                        let profiled_shape = profiled_type.shape();
                        assert!(profiled_shape.is_valid());
                        // Too-complex shapes use hash tables for ivars;
                        // rb_shape_get_iv_index doesn't work for them.
                        // Let the fallthrough DefinedIvar handle these.
                        if profiled_shape.is_complex() { return None; }
                        Some(profiled_shape)
                    }
                    if let Some(summary) = fun.polymorphic_summary(&profiles, self_param, exit_id) {
                        self_param = fun.push_insn(block, Insn::GuardType { val: self_param, guard_type: types::HeapBasicObject, state: exit_id, recompile: None });
                        let join_block = fun.new_block(insn_idx);
                        let join_param = fun.push_insn(join_block, Insn::Param);
                        // Dedup by expected shape so objects with different classes but the
                        // same shape can share code.
                        let mut seen_shape = Vec::with_capacity(summary.buckets().len());
                        for &profiled_type in summary.buckets() {
                            // End of the buckets
                            if profiled_type.is_empty() { break; }
                            let Some(profiled_shape) = can_optimize(profiled_type) else { continue };
                            if seen_shape.contains(&profiled_shape) { continue; }
                            seen_shape.push(profiled_shape);
                            let actual_shape = fun.load_shape(block, self_param);
                            // The expected shape can change over run, so we put it
                            // as a pointer to keep it stable in snapshot tests.
                            let expected_shape = fun.push_insn(block, Insn::Const { val: Const::CShape(profiled_shape) });
                            let has_shape = fun.push_insn(block, Insn::IsBitEqual { left: actual_shape, right: expected_shape });
                            let iftrue_block = fun.new_block(insn_idx);
                            let target = BranchEdge { target: iftrue_block, args: vec![] };
                            let fall_through = fun.new_block(insn_idx);

                            fun.push_insn(block, Insn::CondBranch { val: has_shape,
                                if_true: target,
                                if_false: BranchEdge { target: fall_through, args: vec![] }
                            });

                            block = fall_through;
                            let mut ivar_index: attr_index_t = 0;
                            let result = if unsafe { rb_shape_get_iv_index(profiled_shape.0, id, &mut ivar_index) } {
                                fun.push_insn(iftrue_block, Insn::Const { val: Const::Value(pushval) })
                            } else {
                                fun.push_insn(iftrue_block, Insn::Const { val: Const::Value(Qnil) })
                            };
                            fun.push_insn(iftrue_block, Insn::Jump(BranchEdge { target: join_block, args: vec![result] }));
                        }
                        // In the fallthrough case, do a generic interpreter definedivar and then join.
                        let result = fun.push_insn(block, Insn::DefinedIvar { self_val: self_param, id, pushval, state: exit_id });
                        fun.push_insn(block, Insn::Jump(BranchEdge { target: join_block, args: vec![result] }));
                        state.stack_push(join_param);
                        block = join_block;
                    } else {
                        if let Some(profiled_shape) = (if fun.policy.no_side_exits { None } else { Some(true) })
                            .and_then(|_| fun.monomorphic_summary(&profiles, self_param, exit_id))
                            .and_then(can_optimize) {
                            self_param = fun.guard_heap(block, self_param, exit_id);
                            let shape = fun.load_shape(block, self_param);
                            fun.guard_shape(block, shape, profiled_shape, exit_id, Some(Recompile));
                            let mut ivar_index: attr_index_t = 0;
                            let result = if unsafe { rb_shape_get_iv_index(profiled_shape.0, id, &mut ivar_index) } {
                                fun.push_insn(block, Insn::Const { val: Const::Value(pushval) })
                            } else {
                                // If there is no IVAR index, then the ivar was undefined when we
                                // entered the compiler.  That means we can just return nil for this
                                // shape + iv name
                                fun.push_insn(block, Insn::Const { val: Const::Value(Qnil) })
                            };
                            state.stack_push(result);
                        } else {
                            state.stack_push(fun.push_insn(block, Insn::DefinedIvar { self_val: self_param, id, pushval, state: exit_id }));
                        }
                    }
                }
                YARVINSN_checkkeyword => {
                    // When a keyword is unspecified past index 32, a hash will be used instead.
                    // This can only happen in iseqs taking more than 32 keywords.
                    // In this case, we side exit to the interpreter.
                    if unsafe {(*rb_get_iseq_body_param_keyword(iseq)).num >= VM_KW_SPECIFIED_BITS_MAX.try_into().unwrap()} {
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::TooManyKeywordParameters), recompile: None });
                        break;
                    }
                    let ep_offset = get_arg(pc, 0).as_u32();
                    let index = get_arg(pc, 1).as_u64();
                    let index: u8 = index.try_into().map_err(|_| ParseError::MalformedIseq(insn_idx))?;
                    // Use FrameState to get kw_bits when possible, just like getlocal_WC_0.
                    let val = if !local_inval {
                        state.getlocal(ep_offset)
                    } else if ep_escaped {
                        let ep = fun.get_ep(block, 0);
                        fun.get_local_from_ep(block, iseq, ep, ep_offset, 0, types::BasicObject)
                    } else {
                        let exit_id = fun.push_insn(block, Insn::Snapshot { state: Box::new(exit_state.without_locals()) });
                        fun.push_insn(block, Insn::PatchPoint { invariant: Invariant::NoEPEscape(iseq), state: exit_id });
                        local_inval = false;
                        state.getlocal(ep_offset)
                    };
                    state.stack_push(fun.push_insn(block, Insn::FixnumBitCheck { val, index }));
                }
                YARVINSN_checkmatch => {
                    let flag = get_arg(pc, 0).as_u32();
                    let pattern = state.stack_pop()?;
                    let target = state.stack_pop()?;
                    let result = fun.push_insn(block, Insn::CheckMatch { target, pattern, flag, state: exit_id });
                    state.stack_push(result);
                }
                YARVINSN_getconstant => {
                    let id = ID(get_arg(pc, 0).as_u64());
                    let allow_nil = state.stack_pop()?;
                    let klass = state.stack_pop()?;
                    let result = fun.push_insn(block, Insn::GetConstant { klass, id, allow_nil, state: exit_id });
                    state.stack_push(result);
                }
                YARVINSN_opt_getconstant_path => {
                    let ic: *const iseq_inline_constant_cache = get_arg(pc, 0).as_ptr();
                    let idlist: *const ID = unsafe { (*ic).segments };
                    let ice = unsafe { (*ic).entry };
                    let can_fold = !ice.is_null()
                        && unsafe { (*ice).ic_cref }.is_null()
                        && (unsafe { rb_jit_constcache_shareable(ice) } || fun.assume_single_ractor_mode(block, exit_id));
                    let result = if can_fold {
                        // Invalidate output code on any constant writes associated with constants
                        // referenced after the PatchPoint.
                        fun.push_insn(block, Insn::PatchPoint { invariant: Invariant::StableConstantNames { idlist }, state: exit_id });
                        fun.push_insn(block, Insn::Const { val: Const::Value(unsafe { (*ice).value }) })
                    } else {
                        fun.push_insn(block, Insn::GetConstantPath { ic, state: exit_id })
                    };
                    state.stack_push(result);

                    // Check for `::RubyVM::ZJIT` for directives
                    unsafe {
                        let mut current_segment = (*ic).segments;
                        let mut segments = [ID(0); 4 /* expected segment length */];
                        for segment in segments.iter_mut() {
                            *segment = current_segment.read();
                            if *segment == ID(0) {
                                break;
                            }
                            current_segment = current_segment.add(1);
                        }
                        if [ID!(NULL), ID!(RubyVM), ID!(ZJIT), ID(0)] == segments {
                            debug_assert_ne!(ID!(NULL), ID(0));
                            let ruby_vm_mod = rb_const_lookup(rb_cObject, ID!(RubyVM));
                            if !ruby_vm_mod.is_null() && (*ruby_vm_mod).value == rb_cRubyVM {
                                let zjit_module = VALUE(state::ZJIT_MODULE.load(Ordering::Relaxed));
                                let lookedup_module = rb_const_lookup(rb_cRubyVM, ID!(ZJIT));
                                if !lookedup_module.is_null() && (*lookedup_module).value == zjit_module {
                                    fun.insn_types[result.to_usize()] = Type::from_value(zjit_module);
                                }
                            }
                        }
                    }
                }
                YARVINSN_branchunless | YARVINSN_branchunless_without_ints => {
                    let offset = get_arg(pc, 0).as_i64();
                    if opcode == YARVINSN_branchunless && offset < 0 {
                        fun.push_insn(block, Insn::CheckInterrupts { state: exit_id });
                    }
                    let val = state.stack_pop()?;
                    let test_id = fun.push_insn(block, Insn::Test { val });
                    let target_idx = insn_idx_at_offset(insn_idx, offset);
                    let target = insn_idx_to_block[&target_idx];
                    let nil_false_type = types::Falsy;
                    let nil_false = fun.push_insn(block, Insn::RefineType { val, new_type: nil_false_type });
                    let mut iffalse_state = state.clone();
                    iffalse_state.replace(val, nil_false);
                    let fall_through = fun.new_block(insn_idx);

                    fun.push_insn(block, Insn::CondBranch {
                        val: test_id,
                        if_true: BranchEdge { target: fall_through, args: vec![] },
                        if_false: BranchEdge { target, args: iffalse_state.as_args(self_param) }
                    });

                    block = fall_through;

                    let not_nil_false_type = types::Truthy;
                    let not_nil_false = fun.push_insn(block, Insn::RefineType { val, new_type: not_nil_false_type });
                    state.replace(val, not_nil_false);
                    queue.push_back((state.clone(), target, target_idx, local_inval));
                }
                YARVINSN_branchif | YARVINSN_branchif_without_ints => {
                    let offset = get_arg(pc, 0).as_i64();
                    if opcode == YARVINSN_branchif && offset < 0 {
                        fun.push_insn(block, Insn::CheckInterrupts { state: exit_id });
                    }
                    let val = state.stack_pop()?;
                    let test_id = fun.push_insn(block, Insn::Test { val });
                    let target_idx = insn_idx_at_offset(insn_idx, offset);
                    let target = insn_idx_to_block[&target_idx];
                    let not_nil_false_type = types::Truthy;
                    let not_nil_false = fun.push_insn(block, Insn::RefineType { val, new_type: not_nil_false_type });
                    let mut iftrue_state = state.clone();
                    iftrue_state.replace(val, not_nil_false);

                    let fall_through = fun.new_block(insn_idx);

                    fun.push_insn(block, Insn::CondBranch {
                        val: test_id,
                        if_true: BranchEdge { target, args: iftrue_state.as_args(self_param) },
                        if_false: BranchEdge { target: fall_through, args: vec![] }
                    });

                    block = fall_through;

                    let nil_false_type = types::Falsy;
                    let nil_false = fun.push_insn(block, Insn::RefineType { val, new_type: nil_false_type });
                    state.replace(val, nil_false);
                    queue.push_back((state.clone(), target, target_idx, local_inval));
                }
                YARVINSN_branchnil | YARVINSN_branchnil_without_ints => {
                    let offset = get_arg(pc, 0).as_i64();
                    if opcode == YARVINSN_branchnil && offset < 0 {
                        fun.push_insn(block, Insn::CheckInterrupts { state: exit_id });
                    }
                    let val = state.stack_pop()?;
                    let test_id = fun.push_insn(block, Insn::HasType { val, expected: types::NilClass });
                    let target_idx = insn_idx_at_offset(insn_idx, offset);
                    let target = insn_idx_to_block[&target_idx];
                    let nil = fun.push_insn(block, Insn::Const { val: Const::Value(Qnil) });
                    let mut iftrue_state = state.clone();
                    iftrue_state.replace(val, nil);

                    let fall_through = fun.new_block(insn_idx);

                    fun.push_insn(block, Insn::CondBranch {
                        val: test_id,
                        if_true: BranchEdge { target, args: iftrue_state.as_args(self_param) },
                        if_false: BranchEdge { target: fall_through, args: vec![] }
                    });

                    block = fall_through;
                    let new_type = types::NotNil;
                    let not_nil = fun.push_insn(block, Insn::RefineType { val, new_type });
                    state.replace(val, not_nil);
                    queue.push_back((state.clone(), target, target_idx, local_inval));
                }
                YARVINSN_opt_case_dispatch => {
                    let key = state.stack_pop()?;
                    // The interpreter jumps straight out of `opt_case_dispatch` on a hit, so
                    // the `===` chain that follows is dead code there and never gets profiled.
                    // Compiling the chain therefore means an unprofiled `Integer#===` cfunc
                    // call per `when` clause tested. Compile the hash lookup instead, as a
                    // binary search over the (Fixnum) keys.
                    // The chain calls `Integer#===` on each `when` literal, so the lookup only
                    // agrees with it while that stays the stock implementation.
                    let unredefined = unsafe { rb_BASIC_OP_UNREDEFINED_P(BOP_EQQ, INTEGER_REDEFINED_OP_FLAG) };
                    let Some(entries) = unredefined.then(|| cdhash_fixnum_entries(get_arg(pc, 0))).flatten() else {
                        // Fall through to the `===` chain.
                        continue;
                    };
                    fun.push_insn(block, Insn::PatchPoint {
                        invariant: Invariant::BOPRedefined { klass: INTEGER_REDEFINED_OP_FLAG, bop: BOP_EQQ },
                        state: exit_id,
                    });
                    // Only Fixnum keys can match an all-Fixnum dispatch hash. Anything else
                    // falls through to the `===` chain, which handles every type correctly.
                    let is_fixnum = fun.push_insn(block, Insn::HasType { val: key, expected: types::Fixnum });
                    let chain_block = fun.new_block(insn_idx);
                    let dispatch_block = fun.new_block(insn_idx);
                    fun.push_insn(block, Insn::CondBranch {
                        val: is_fixnum,
                        if_true: BranchEdge { target: dispatch_block, args: vec![] },
                        if_false: BranchEdge { target: chain_block, args: vec![] },
                    });
                    let key_fixnum = fun.push_insn(dispatch_block, Insn::RefineType { val: key, new_type: types::Fixnum });
                    let mut state = state.clone();
                    state.replace(key, key_fixnum);
                    let else_idx = insn_idx_at_offset(insn_idx, get_arg(pc, 1).as_i64());
                    let else_block = insn_idx_to_block[&else_idx];
                    // Emit a comparison tree over the sorted keys. Each leaf range is scanned
                    // linearly; anything bigger splits on a pivot key.
                    let mut work = vec![(0usize, entries.len(), dispatch_block)];
                    while let Some((lo, hi, mut cur)) = work.pop() {
                        if hi - lo > 3 {
                            let mid = lo + (hi - lo) / 2;
                            let pivot = fun.push_insn(cur, Insn::Const { val: Const::Value(VALUE::fixnum_from_isize(entries[mid].0 as isize)) });
                            let less = fun.push_insn(cur, Insn::FixnumLt { left: key_fixnum, right: pivot });
                            let less_c = fun.push_insn(cur, Insn::Test { val: less });
                            let lo_block = fun.new_block(insn_idx);
                            let hi_block = fun.new_block(insn_idx);
                            fun.push_insn(cur, Insn::CondBranch {
                                val: less_c,
                                if_true: BranchEdge { target: lo_block, args: vec![] },
                                if_false: BranchEdge { target: hi_block, args: vec![] },
                            });
                            work.push((lo, mid, lo_block));
                            work.push((mid, hi, hi_block));
                            continue;
                        }
                        for &(key_value, offset) in &entries[lo..hi] {
                            let target_idx = insn_idx_at_offset(insn_idx, offset);
                            let target = insn_idx_to_block[&target_idx];
                            let expected = fun.push_insn(cur, Insn::Const { val: Const::Value(VALUE::fixnum_from_isize(key_value as isize)) });
                            let matches = fun.push_insn(cur, Insn::IsBitEqual { left: key_fixnum, right: expected });
                            let next = fun.new_block(insn_idx);
                            fun.push_insn(cur, Insn::CondBranch {
                                val: matches,
                                if_true: BranchEdge { target, args: state.as_args(self_param) },
                                if_false: BranchEdge { target: next, args: vec![] },
                            });
                            queue.push_back((state.clone(), target, target_idx, local_inval));
                            cur = next;
                        }
                        fun.push_insn(cur, Insn::Jump(BranchEdge { target: else_block, args: state.as_args(self_param) }));
                        queue.push_back((state.clone(), else_block, else_idx, local_inval));
                    }
                    // Keep compiling the `===` chain for non-Fixnum keys.
                    block = chain_block;
                }
                YARVINSN_opt_new => {
                    let cd: *const rb_call_data = get_arg(pc, 0).as_ptr();
                    let dst = get_arg(pc, 1).as_i64();

                    // Check if #new resolves to rb_class_new_instance_pass_kw.
                    // TODO: Guard on a profiled class and add a patch point for #new redefinition
                    let argc = crate::profile::num_arguments_on_stack(cd);
                    let ci = unsafe { (*cd).ci };
                    let flags = unsafe { rb_vm_ci_flag(ci) };
                    assert_eq!(flags & VM_CALL_ARGS_BLOCKARG, 0);
                    let val = state.stack_topn(argc)?;
                    let test_id = fun.push_insn(block, Insn::IsMethodCfunc { val, cd, cfunc: rb_class_new_instance_pass_kw as *const u8, state: exit_id });

                    // Jump to the fallback block if it's not the expected function.
                    // Skip CheckInterrupts since the #new call will do it very soon anyway.
                    let target_idx = insn_idx_at_offset(insn_idx, dst);
                    let target = insn_idx_to_block[&target_idx];
                    let fall_through = fun.new_block(insn_idx);
                    fun.push_insn(block, Insn::CondBranch {
                        val: test_id,
                        if_true: BranchEdge { target: fall_through, args: vec![] },
                        if_false: BranchEdge { target, args: state.as_args(self_param) }
                    });
                    block = fall_through;
                    queue.push_back((state.clone(), target, target_idx, local_inval));

                    // Move on to the fast path
                    let insn_id = fun.push_insn(block, Insn::ObjectAlloc { val, state: exit_id });
                    state.stack_setn(argc, insn_id);
                    state.stack_setn(argc + 1, insn_id);
                }
                YARVINSN_jump | YARVINSN_jump_without_ints => {
                    let offset = get_arg(pc, 0).as_i64();
                    if opcode == YARVINSN_jump && offset < 0 {
                        fun.push_insn(block, Insn::CheckInterrupts { state: exit_id });
                    }
                    let target_idx = insn_idx_at_offset(insn_idx, offset);
                    let target = insn_idx_to_block[&target_idx];
                    let _branch_id = fun.push_insn(block, Insn::Jump(
                        BranchEdge { target, args: state.as_args(self_param) }
                    ));
                    queue.push_back((state.clone(), target, target_idx, local_inval));
                    break;  // Don't enqueue the next block as a successor
                }
                YARVINSN_getlocal_WC_0 => {
                    let ep_offset = get_arg(pc, 0).as_u32();
                    if !local_inval {
                        // The FrameState is the source of truth for locals until invalidated.
                        // In case of JIT-to-JIT send locals might never end up in EP memory.
                        let val = state.getlocal(ep_offset);
                        state.stack_push(val);
                    } else if ep_escaped {
                        // Read the local using EP
                        let ep = fun.get_ep(block, 0);
                        let val = fun.get_local_from_ep(block, iseq, ep, ep_offset, 0, types::BasicObject);
                        state.setlocal(ep_offset, val); // remember the result to spill on side-exits
                        state.stack_push(val);
                    } else {
                        assert!(local_inval); // if check above
                        // There has been some non-leaf call since JIT entry or the last patch point,
                        // so add a patch point to make sure locals have not been escaped.
                        let exit_id = fun.push_insn(block, Insn::Snapshot { state: Box::new(exit_state.without_locals()) }); // skip spilling locals
                        fun.push_insn(block, Insn::PatchPoint { invariant: Invariant::NoEPEscape(iseq), state: exit_id });
                        local_inval = false;

                        // Read the local from FrameState
                        let val = state.getlocal(ep_offset);
                        state.stack_push(val);
                    }
                }
                YARVINSN_setlocal_WC_0 => {
                    let ep_offset = get_arg(pc, 0).as_u32();
                    let val = state.stack_pop()?;
                    if ep_escaped {
                        // Write the local using EP
                        fun.push_insn(block, Insn::SetLocal { val, ep_offset, level: 0, state: exit_id });
                    } else if local_inval {
                        // If there has been any non-leaf call since JIT entry or the last patch point,
                        // add a patch point to make sure locals have not been escaped.
                        let exit_id = fun.push_insn(block, Insn::Snapshot { state: Box::new(exit_state.without_locals()) }); // skip spilling locals
                        fun.push_insn(block, Insn::PatchPoint { invariant: Invariant::NoEPEscape(iseq), state: exit_id });
                        local_inval = false;
                    }
                    // Write the local into FrameState
                    state.setlocal(ep_offset, val);
                }
                YARVINSN_getlocal_WC_1 => {
                    let ep_offset = get_arg(pc, 0).as_u32();
                    let ep = fun.get_ep(block, 1);
                    state.stack_push(fun.get_local_from_ep(block, iseq, ep, ep_offset, 1, types::BasicObject));
                }
                YARVINSN_setlocal_WC_1 => {
                    let ep_offset = get_arg(pc, 0).as_u32();
                    fun.push_insn(block, Insn::SetLocal { val: state.stack_pop()?, ep_offset, level: 1, state: exit_id });
                }
                YARVINSN_getlocal => {
                    let ep_offset = get_arg(pc, 0).as_u32();
                    let level = get_arg(pc, 1).as_u32();
                    if level == 0 && !local_inval {
                        // Same optimization as getlocal_WC_0: use FrameState
                        let val = state.getlocal(ep_offset);
                        state.stack_push(val);
                    } else {
                        let ep = fun.get_ep(block, level);
                        let val = fun.get_local_from_ep(block, iseq, ep, ep_offset, level, types::BasicObject);
                        if level == 0 {
                            state.setlocal(ep_offset, val);
                        }
                        state.stack_push(val);
                    }
                }
                YARVINSN_setlocal => {
                    let ep_offset = get_arg(pc, 0).as_u32();
                    let level = get_arg(pc, 1).as_u32();
                    fun.push_insn(block, Insn::SetLocal { val: state.stack_pop()?, ep_offset, level, state: exit_id });
                }
                YARVINSN_setblockparam => {
                    let ep_offset = get_arg(pc, 0).as_u32();
                    let level = get_arg(pc, 1).as_u32();
                    let val = state.stack_pop()?;
                    fun.push_insn(block, Insn::SetLocal { val, ep_offset, level, state: exit_id });
                    if level == 0 {
                        state.setlocal(ep_offset, val);
                    }
                    let ep = fun.get_ep(block, level);
                    let flags = fun.load_field(block, ep, FieldName::VM_ENV_DATA_INDEX_FLAGS, SIZEOF_VALUE_I32 * (VM_ENV_DATA_INDEX_FLAGS as i32), types::CInt64);
                    let modified_flag = fun.push_insn(block, Insn::Const {
                        val: Const::CInt64(VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM.into()),
                    });
                    let modified = fun.push_insn(block, Insn::IntOr { left: flags, right: modified_flag });
                    fun.push_insn(block, Insn::StoreField {
                        recv: ep,
                        id: FieldName::VM_ENV_DATA_INDEX_FLAGS,
                        offset: SIZEOF_VALUE_I32 * (VM_ENV_DATA_INDEX_FLAGS as i32),
                        val: modified,
                        num_bits: types::CInt64.num_bits(),
                    });
                }
                YARVINSN_getblockparamproxy => {
                    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
                    enum ProfiledBlockHandlerFamily {
                        Nil,
                        IseqOrIfunc,
                        Proc,
                    }
                    impl ProfiledBlockHandlerFamily {
                        fn from_profiled_type(profiled_type: ProfiledType) -> Option<Self> {
                            let obj = profiled_type.class();
                            if obj.nil_p() {
                                Some(Self::Nil)
                            } else if unsafe {
                                rb_IMEMO_TYPE_P(obj, imemo_iseq) == 1
                                    || rb_IMEMO_TYPE_P(obj, imemo_ifunc) == 1
                            } {
                                Some(Self::IseqOrIfunc)
                            } else if unsafe { rb_obj_is_proc(obj).test() } {
                                Some(Self::Proc)
                            } else {
                                None
                            }
                        }
                    }

                    let ep_offset = get_arg(pc, 0).as_u32();
                    let level = get_arg(pc, 1).as_u32();
                    let branch_insn_idx = exit_state.insn_idx as u32;

                    // `getblockparamproxy` has two semantic paths:
                    // - modified: return the already-materialized block local from EP
                    // - unmodified: inspect the block handler and produce proxy/nil
                    let modified_block = fun.new_block(branch_insn_idx);
                    let unmodified_block = fun.new_block(branch_insn_idx);
                    let join_block = fun.new_block(insn_idx);
                    let join_result = fun.push_insn(join_block, Insn::Param);
                    let join_local = if level == 0 { Some(fun.push_insn(join_block, Insn::Param)) } else { None };

                    let ep = fun.get_ep(block, level);
                    let flags = fun.load_ep_flags(block, ep);
                    let is_modified = fun.push_insn(block, Insn::IsBlockParamModified { flags });

                    fun.push_insn(block, Insn::CondBranch {
                        val: is_modified,
                        if_true: BranchEdge { target: modified_block, args: vec![] },
                        if_false: BranchEdge { target: unmodified_block, args: vec![] }
                    });

                    // Push modified block: load the block local via EP.
                    let modified_val = fun.get_local_from_ep(modified_block, iseq, ep, ep_offset, level, types::BasicObject);
                    let mut modified_args = vec![modified_val];
                    if level == 0 { modified_args.push(modified_val); }
                    fun.push_insn(modified_block, Insn::Jump(BranchEdge { target: join_block, args: modified_args }));

                    let original_local = if level == 0 { Some(state.getlocal(ep_offset)) } else { None };
                    // `block_handler & 1 == 1` accepts both ISEQ (0b01) and ifunc
                    // (0b11) handlers. Keep a compile-time check that this shortcut
                    // does not accidentally accept symbol block handlers.
                    const _: () = assert!(RUBY_SYMBOL_FLAG & 1 == 0, "guard below rejects symbol block handlers");


                    let profiled_block_summary = payload.profile.get_operand_types(exit_state.insn_idx)
                        .and_then(|types| types.first())
                        .map(TypeDistributionSummary::new);

                    let mut profiled_handlers = Vec::new();
                    if let Some(summary) = profiled_block_summary.as_ref() {
                        if summary.is_monomorphic() || summary.is_polymorphic() || summary.is_skewed_polymorphic() {
                            for &profiled_type in summary.buckets() {
                                if profiled_type.is_empty() {
                                    break;
                                }
                                if let Some(profiled_handler) = ProfiledBlockHandlerFamily::from_profiled_type(profiled_type) {
                                    if !profiled_handlers.contains(&profiled_handler) {
                                        profiled_handlers.push(profiled_handler);
                                    }
                                }
                            }
                        }
                    }

                    if profiled_handlers.is_empty() {
                        // Sites we could not specialize -- no profile yet, or a megamorphic one.
                        // Dispatch on the two families that need no C call to recognize instead
                        // of guarding for ISEQ/ifunc alone: a `&blk` parameter left unpassed is
                        // an ordinary thing for a caller to do, and a guard makes every such
                        // call side-exit and abandon the rest of the method.
                        profiled_handlers.push(ProfiledBlockHandlerFamily::IseqOrIfunc);
                        profiled_handlers.push(ProfiledBlockHandlerFamily::Nil);
                    }

                    match profiled_handlers.as_slice() {
                        [] => unreachable!("profiled_handlers was just given a default"),
                        // A single supported profiled family. Emit a monomorphic fast path
                        [profiled_handler] => match profiled_handler {
                            ProfiledBlockHandlerFamily::Nil => {
                                let block_handler = fun.load_ep_env_field(unmodified_block, ep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, VM_ENV_DATA_INDEX_SPECVAL, types::CInt64);
                                fun.push_insn(unmodified_block, Insn::GuardBitEquals { val: block_handler, expected: Const::CInt64(VM_BLOCK_HANDLER_NONE.into()), reason: Box::new(SideExitReason::BlockParamProxyNotNil), state: exit_id, recompile: Some(Recompile) });
                                let nil_val = fun.push_insn(unmodified_block, Insn::Const { val: Const::Value(Qnil) });
                                let mut args = vec![nil_val];
                                if let Some(local) = original_local {
                                    args.push(local);
                                }
                                fun.push_insn(unmodified_block, Insn::Jump(BranchEdge { target: join_block, args }));
                            }
                            ProfiledBlockHandlerFamily::IseqOrIfunc => {
                                let block_handler = fun.load_ep_env_field(unmodified_block, ep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, VM_ENV_DATA_INDEX_SPECVAL, types::CInt64);
                                // This handles two cases which are nearly identical.
                                // Block handler is a tagged pointer. Look at the tag.
                                //   VM_BH_ISEQ_BLOCK_P(): block_handler & 0x03 == 0x01
                                //   VM_BH_IFUNC_P():      block_handler & 0x03 == 0x03
                                // So to check for either of those cases we can use: val & 0x1 == 0x1

                                // Bail out if the block handler is neither ISEQ nor ifunc
                                fun.push_insn(unmodified_block, Insn::GuardAnyBitSet { val: block_handler, mask: Const::CUInt64(0x1), mask_name: None, reason: Box::new(SideExitReason::BlockParamProxyNotIseqOrIfunc), state: exit_id, recompile: Some(Recompile) });
                                // TODO(Shopify/ruby#753): GC root, so we should be able to avoid unnecessary GC tracing
                                let proxy_val = fun.push_insn(unmodified_block, Insn::Const { val: Const::Value(unsafe { rb_block_param_proxy }) });
                                let mut args = vec![proxy_val];
                                if let Some(local) = original_local {
                                    args.push(local);
                                }
                                fun.push_insn(unmodified_block, Insn::Jump(BranchEdge { target: join_block, args }));
                            }
                            ProfiledBlockHandlerFamily::Proc => {
                                let proc_val = fun.load_ep_env_field(unmodified_block, ep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, VM_ENV_DATA_INDEX_SPECVAL, types::BasicObject);
                                let is_proc = fun.push_insn(unmodified_block, Insn::CCall {
                                    cfunc: rb_obj_is_proc as *const u8,
                                    recv: proc_val,
                                    args: vec![],
                                    name: ID!(rb_obj_is_proc),
                                    owner: Qnil,
                                    return_type: types::BasicObject,
                                    elidable: true,
                                });
                                fun.push_insn(unmodified_block, Insn::GuardBitEquals { val: is_proc, expected: Const::Value(Qtrue), reason: Box::new(SideExitReason::BlockParamProxyNotProc), state: exit_id, recompile: Some(Recompile) });
                                let mut args = vec![proc_val];
                                if let Some(local) = original_local {
                                    args.push(local);
                                }
                                fun.push_insn(unmodified_block, Insn::Jump(BranchEdge { target: join_block, args }));
                            }
                        },
                        // Multiple supported profiled families. Emit a polymorphic dispatch
                        _ => {
                            let block_handler = fun.load_ep_env_field(unmodified_block, ep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, VM_ENV_DATA_INDEX_SPECVAL, types::CInt64);
                            let profiled_blocks = profiled_handlers.iter()
                            .map(|&kind| (kind, fun.new_block(branch_insn_idx)))
                            .collect::<Vec<_>>();

                            let mut current_block = unmodified_block;

                            for &(kind, profiled_block) in &profiled_blocks {
                                match kind {
                                    ProfiledBlockHandlerFamily::Nil => {
                                        let none_handler = fun.push_insn(current_block, Insn::Const {
                                            val: Const::CInt64(VM_BLOCK_HANDLER_NONE.into()),
                                        });
                                        let is_none = fun.push_insn(current_block, Insn::IsBitEqual {
                                            left: block_handler,
                                            right: none_handler,
                                        });

                                        let next_block = fun.new_block(branch_insn_idx);

                                        fun.push_insn(current_block, Insn::CondBranch {
                                            val: is_none,
                                            if_true: BranchEdge { target: profiled_block, args: vec![] },
                                            if_false: BranchEdge { target: next_block, args: vec![] },
                                        });

                                        current_block = next_block;

                                        let val = fun.push_insn(profiled_block, Insn::Const { val: Const::Value(Qnil) });
                                        let mut args = vec![val];
                                        if let Some(local) = original_local { args.push(local); }
                                        fun.push_insn(profiled_block, Insn::Jump(BranchEdge { target: join_block, args }));

                                    }
                                    ProfiledBlockHandlerFamily::IseqOrIfunc => {
                                        // This handles two cases which are nearly identical.
                                        // Block handler is a tagged pointer. Look at the tag.
                                        //   VM_BH_ISEQ_BLOCK_P(): block_handler & 0x03 == 0x01
                                        //   VM_BH_IFUNC_P():      block_handler & 0x03 == 0x03
                                        // So to check for either of those cases we can use: val & 0x1 == 0x1
                                        let tag_mask = fun.push_insn(current_block, Insn::Const { val: Const::CInt64(0x1) });
                                        let tag_bits = fun.push_insn(current_block, Insn::IntAnd {
                                            left: block_handler,
                                            right: tag_mask,
                                        });
                                        let is_iseq_or_ifunc = fun.push_insn(current_block, Insn::IsBitEqual {
                                            left: tag_bits,
                                            right: tag_mask,
                                        });
                                        let next_block = fun.new_block(branch_insn_idx);
                                        fun.push_insn(current_block, Insn::CondBranch {
                                            val: is_iseq_or_ifunc,
                                            if_true: BranchEdge { target: profiled_block, args: vec![] },
                                            if_false: BranchEdge { target: next_block, args: vec![] },
                                        });
                                        current_block = next_block;

                                        // TODO(Shopify/ruby#753): GC root, so we should be able to avoid unnecessary GC tracing
                                        let val = fun.push_insn(profiled_block, Insn::Const { val: Const::Value(unsafe { rb_block_param_proxy }) });
                                        let mut args = vec![val];
                                        if let Some(local) = original_local { args.push(local); }
                                        fun.push_insn(profiled_block, Insn::Jump(BranchEdge { target: join_block, args }));
                                    },
                                    ProfiledBlockHandlerFamily::Proc => {
                                        let proc_check_block = fun.new_block(branch_insn_idx);
                                        let next_block = fun.new_block(branch_insn_idx);
                                        fun.push_insn(current_block, Insn::Jump(BranchEdge { target: proc_check_block, args: vec![] }));

                                        let proc_val = fun.load_ep_env_field(proc_check_block, ep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, VM_ENV_DATA_INDEX_SPECVAL, types::BasicObject);
                                        let proc_result = fun.push_insn(proc_check_block, Insn::CCall {
                                            cfunc: rb_obj_is_proc as *const u8,
                                            recv: proc_val,
                                            args: vec![],
                                            name: ID!(rb_obj_is_proc),
                                            owner: Qnil,
                                            return_type: types::BasicObject,
                                            elidable: true,
                                        });
                                        let true_val = fun.push_insn(proc_check_block, Insn::Const { val: Const::Value(Qtrue) });
                                        let is_proc = fun.push_insn(proc_check_block, Insn::IsBitEqual { left: proc_result, right: true_val });
                                        fun.push_insn(proc_check_block, Insn::CondBranch {
                                            val: is_proc,
                                            if_true: BranchEdge { target: profiled_block, args: vec![] },
                                            if_false: BranchEdge { target: next_block, args: vec![] },
                                        });
                                        current_block = next_block;

                                        let mut args = vec![proc_val];
                                        if let Some(local) = original_local { args.push(local); }
                                        fun.push_insn(profiled_block, Insn::Jump(BranchEdge { target: join_block, args }));
                                    }
                                }
                            }

                            fun.push_insn(current_block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::BlockParamProxyProfileNotCovered), recompile: None });
                        }
                    }

                    // Continue compilation from the merged continuation block at the next
                    // instruction.
                    if let Some(local_param) = join_local {
                        state.setlocal(ep_offset, local_param);
                    }
                    // Remember that this value encodes the block handler of the frame at `level`.
                    // A `foo(&blk)` site can then pass that handler straight through instead of
                    // going through the interpreter, but only when the frame it came from is the
                    // one `VM_CF_BLOCK_HANDLER` would read, i.e. the local EP.
                    if level == get_lvar_level(iseq) {
                        block_param_proxy_values.insert(join_result);
                    }
                    state.stack_push(join_result);
                    block = join_block;
                }
                YARVINSN_getblockparam => {
                    let ep_offset = get_arg(pc, 0).as_u32();
                    let level = get_arg(pc, 1).as_u32();
                    let branch_insn_idx = exit_state.insn_idx as u32;

                    // If the block param is already a Proc (modified), read it from EP.
                    // Otherwise, convert it to a Proc and store it to EP.
                    let modified_block = fun.new_block(branch_insn_idx);
                    let unmodified_block = fun.new_block(branch_insn_idx);
                    let join_block = fun.new_block(insn_idx);
                    let join_param = fun.push_insn(join_block, Insn::Param);

                    let ep = fun.get_ep(block, level);
                    let flags = fun.load_ep_flags(block, ep);
                    let is_modified = fun.push_insn(block, Insn::IsBlockParamModified { flags });

                    fun.push_insn(block, Insn::CondBranch {
                        val: is_modified,
                        if_true: BranchEdge { target: modified_block, args: vec![] },
                        if_false: BranchEdge { target: unmodified_block, args: vec![] }
                    });

                    // Push modified block: read Proc from EP.
                    let modified_val = fun.get_local_from_ep(modified_block, iseq, ep, ep_offset, level, types::BasicObject);
                    fun.push_insn(modified_block, Insn::Jump(BranchEdge { target: join_block, args: vec![modified_val] }));

                    // Push unmodified block: convert block handler to Proc.
                    let unmodified_val = fun.push_insn(unmodified_block, Insn::GetBlockParam {
                        ep_offset,
                        level,
                        state: exit_id,
                    });
                    fun.push_insn(unmodified_block, Insn::Jump(BranchEdge { target: join_block, args: vec![unmodified_val] }));

                    // Continue compilation from the join block at the next instruction.
                    if level == 0 {
                        state.setlocal(ep_offset, join_param);
                    }
                    state.stack_push(join_param);
                    block = join_block;
                }
                YARVINSN_pop => { state.stack_pop()?; }
                YARVINSN_dup => { state.stack_push(state.stack_top()?); }
                YARVINSN_dupn => {
                    // Duplicate the top N element of the stack. As we push, n-1 naturally
                    // points higher in the original stack.
                    let n = get_arg(pc, 0).as_usize();
                    for _ in 0..n {
                        state.stack_push(state.stack_topn(n-1)?);
                    }
                }
                YARVINSN_swap => {
                    let right = state.stack_pop()?;
                    let left = state.stack_pop()?;
                    state.stack_push(right);
                    state.stack_push(left);
                }
                YARVINSN_setn => {
                    let n = get_arg(pc, 0).as_usize();
                    let top = state.stack_top()?;
                    state.stack_setn(n, top);
                }
                YARVINSN_topn => {
                    let n = get_arg(pc, 0).as_usize();
                    let top = state.stack_topn(n)?;
                    state.stack_push(top);
                }
                YARVINSN_adjuststack => {
                    let mut n = get_arg(pc, 0).as_usize();
                    while n > 0 {
                        state.stack_pop()?;
                        n -= 1;
                    }
                }
                YARVINSN_opt_neq => {
                    // NB: opt_neq has two cd; get_arg(0) is for eq and get_arg(1) is for neq
                    let cd: *const rb_call_data = get_arg(pc, 1).as_ptr();
                    let call_info = unsafe { (*cd).ci };
                    let flags = unsafe { rb_vm_ci_flag(call_info) };
                    if let Err(call_type) = unhandled_call_type(flags) {
                        // Can't handle the call type; side-exit into the interpreter
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledCallType(call_type)), recompile: None });
                        break;  // End the block
                    }
                    let argc = crate::profile::num_arguments_on_stack(cd);
                    assert_eq!(flags & VM_CALL_ARGS_BLOCKARG, 0);

                    // Side-exit send fallbacks while tracing to avoid FLAG_FINISH breaking throw TAG_RETURN semantics
                    if unsafe { rb_zjit_iseq_tracing_currently_enabled() } {
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::SendWhileTracing), recompile: None });
                        break;
                    }
                    let args = state.stack_pop_n(argc as usize)?;
                    let recv = state.stack_pop()?;
                    let send = fun.push_insn(block, Insn::Send { recv, cd, block: None, args, state: exit_id, reason: Uncategorized(opcode.into()) });
                    state.stack_push(send);
                }
                YARVINSN_opt_hash_freeze => {
                    let klass = HASH_REDEFINED_OP_FLAG;
                    let bop = BOP_FREEZE;
                    if !fun.guard_bop_not_redefined(block, klass, bop, exit_id) {
                        break;  // End the block
                    }
                    let recv = fun.push_insn(block, Insn::Const { val: Const::Value(get_arg(pc, 0)) });
                    state.stack_push(recv);
                }
                YARVINSN_opt_ary_freeze => {
                    let klass = ARRAY_REDEFINED_OP_FLAG;
                    let bop = BOP_FREEZE;
                    if !fun.guard_bop_not_redefined(block, klass, bop, exit_id) {
                        break;  // End the block
                    }
                    let recv = fun.push_insn(block, Insn::Const { val: Const::Value(get_arg(pc, 0)) });
                    state.stack_push(recv);
                }
                YARVINSN_opt_str_freeze => {
                    let klass = STRING_REDEFINED_OP_FLAG;
                    let bop = BOP_FREEZE;
                    if !fun.guard_bop_not_redefined(block, klass, bop, exit_id) {
                        break;  // End the block
                    }
                    let recv = fun.push_insn(block, Insn::Const { val: Const::Value(get_arg(pc, 0)) });
                    state.stack_push(recv);
                }
                YARVINSN_opt_str_uminus => {
                    let klass = STRING_REDEFINED_OP_FLAG;
                    let bop = BOP_UMINUS;
                    if !fun.guard_bop_not_redefined(block, klass, bop, exit_id) {
                        break;  // End the block
                    }
                    let recv = fun.push_insn(block, Insn::Const { val: Const::Value(get_arg(pc, 0)) });
                    state.stack_push(recv);
                }
                YARVINSN_leave => {
                    fun.push_insn(block, Insn::CheckInterrupts { state: exit_id });
                    let val = state.stack_pop()?;
                    match mode {
                        AddIseqMode::Standalone | AddIseqMode::ExceptionEntry(_) => fun.push_insn(block, Insn::Return { val, pop_inlined_frames: 0 }),
                        AddIseqMode::Inlined { return_block, .. } => { fun.push_insn(block, Insn::Jump(BranchEdge { target: return_block, args: vec![val] })) }
                    };
                    break;  // Don't enqueue the next block as a successor
                }
                YARVINSN_throw => {
                    let throw_state = get_arg(pc, 0).as_u32();
                    let val = state.stack_pop()?;
                    // A `return` inside a block we inlined out of the compiled function's own
                    // frame needs no throw at all: the frame it unwinds to is the frame we are
                    // about to return from, and inline_block_iseq() checked that nothing between
                    // the two has an `ensure` to run. Return the value directly, discarding the
                    // inlined frames still on the CFP stack.
                    if let AddIseqMode::Inlined { block_return_pops: Some(pops), .. } = mode {
                        if throw_state == RUBY_TAG_RETURN as u32 {
                            fun.push_insn(block, Insn::Return { val, pop_inlined_frames: pops });
                            break;  // Don't enqueue the next block as a successor
                        }
                    }
                    fun.push_insn(block, Insn::Throw { throw_state, val, state: exit_id });
                    break;  // Don't enqueue the next block as a successor
                }

                // These are opt_send_without_block and all the opt_* instructions
                // specialized to a certain method that could also be serviced
                // using the general send implementation. The optimizer start from
                // a general send for all of these later in the pipeline.
                YARVINSN_opt_nil_p |
                YARVINSN_opt_plus |
                YARVINSN_opt_minus |
                YARVINSN_opt_mult |
                YARVINSN_opt_div |
                YARVINSN_opt_mod |
                YARVINSN_opt_eq |
                YARVINSN_opt_lt |
                YARVINSN_opt_le |
                YARVINSN_opt_gt |
                YARVINSN_opt_ge |
                YARVINSN_opt_ltlt |
                YARVINSN_opt_aset |
                YARVINSN_opt_length |
                YARVINSN_opt_size |
                YARVINSN_opt_aref |
                YARVINSN_opt_empty_p |
                YARVINSN_opt_succ |
                YARVINSN_opt_and |
                YARVINSN_opt_or |
                YARVINSN_opt_not |
                YARVINSN_opt_regexpmatch2 |
                YARVINSN_opt_send_without_block => {
                    let cd: *const rb_call_data = get_arg(pc, 0).as_ptr();
                    let call_info = unsafe { (*cd).ci };
                    let flags = unsafe { rb_vm_ci_flag(call_info) };
                    if let Err(call_type) = unhandled_call_type(flags) {
                        // Can't handle tailcall; side-exit into the interpreter
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledCallType(call_type)), recompile: None });
                        break;  // End the block
                    }
                    let argc = crate::profile::num_arguments_on_stack(cd);
                    let mid = unsafe { rb_vm_ci_mid(call_info) };

                    // Check for calls to directives
                    if argc == 0
                        && (mid == ID!(induce_side_exit_bang) || mid == ID!(induce_compile_failure_bang) || mid == ID!(induce_breakpoint_bang))
                        && fun.type_of(state.stack_top()?)
                              .ruby_object()
                              .is_some_and(|obj| obj == VALUE(state::ZJIT_MODULE.load(Ordering::Relaxed)))
                    {

                        if mid == ID!(induce_side_exit_bang)
                            && state::zjit_module_method_match_serial(ID!(induce_side_exit_bang), &state::INDUCE_SIDE_EXIT_SERIAL)
                        {
                            fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::DirectiveInduced), recompile: None });
                            break;  // End the block
                        }
                        if mid == ID!(induce_compile_failure_bang)
                            && state::zjit_module_method_match_serial(ID!(induce_compile_failure_bang), &state::INDUCE_COMPILE_FAILURE_SERIAL)
                        {
                            return Err(ParseError::DirectiveInduced);
                        }
                        if mid == ID!(induce_breakpoint_bang)
                            && state::zjit_module_method_match_serial(ID!(induce_breakpoint_bang), &state::INDUCE_BREAKPOINT_SERIAL)
                        {
                            fun.push_insn(block, Insn::BreakPoint);
                            state.stack_pop()?; // pop the receiver (::RubyVM::ZJIT)
                            state.stack_push(fun.push_insn(block, Insn::Const { val: Const::Value(Qnil) }));
                        }
                    }

                    // Side-exit send fallbacks while tracing to avoid FLAG_FINISH breaking throw TAG_RETURN semantics
                    if unsafe { rb_zjit_iseq_tracing_currently_enabled() } {
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::SendWhileTracing), recompile: None });
                        break;
                    }

                    let args = state.stack_pop_n(argc as usize)?;
                    let recv = state.stack_pop()?;

                    // `recv.send(:name, ...)` chooses its callee from the first argument, so the
                    // call site's method ID (`send`) tells the optimizer nothing. Branch on the
                    // method names the profiler saw and let each arm resolve as a call to that
                    // method; anything unseen falls through to the ordinary dynamic send.
                    let send_names = send_method_names(&payload.profile, cd, exit_state.insn_idx, args.len());
                    if !send_names.is_empty() {
                        let join_block = fun.new_block(insn_idx);
                        let join_param = fun.push_insn(join_block, Insn::Param);
                        for name in send_names {
                            let expected = fun.push_insn(block, Insn::Const { val: Const::Value(name) });
                            let is_name = fun.push_insn(block, Insn::IsBitEqual { left: args[0], right: expected });
                            let iftrue_block = fun.new_block(insn_idx);
                            let fall_through = fun.new_block(insn_idx);
                            fun.push_insn(block, Insn::CondBranch {
                                val: is_name,
                                if_true: BranchEdge { target: iftrue_block, args: vec![] },
                                if_false: BranchEdge { target: fall_through, args: vec![] }
                            });
                            block = fall_through;
                            // Each arm needs its own Snapshot so that the recorded method name
                            // only applies to that arm. The operand profiles still apply, so
                            // carry them over.
                            let snapshot = fun.push_insn(iftrue_block, Insn::Snapshot { state: Box::new(exit_state.clone()) });
                            profiles.copy_entries(exit_id, snapshot);
                            // Keep the full argument list (including the method name) so that a
                            // send that turns out not to be specializable still lowers to a
                            // correct dynamic `send`. type_specialize drops the name argument
                            // when, and only when, it resolves the call.
                            let send = fun.push_insn(iftrue_block, Insn::Send { recv, cd, block: None, args: args.clone(), state: snapshot, reason: Uncategorized(opcode.into()) });
                            fun.send_mid_overrides.insert(send, unsafe { rb_sym2id(name) });
                            fun.push_insn(iftrue_block, Insn::Jump(BranchEdge { target: join_block, args: vec![send] }));
                        }
                        let send = fun.push_insn(block, Insn::Send { recv, cd, block: None, args, state: exit_id, reason: SendUnprofiledMethodName });
                        fun.push_insn(block, Insn::Jump(BranchEdge { target: join_block, args: vec![send] }));
                        state.stack_push(join_param);
                        block = join_block;
                    } else if let Some((new_block, result)) = emit_polymorphic_send(
                        fun, &mut profiles, block, insn_idx, exit_id, &exit_state,
                        cd, recv, &args, None, opcode, branch_monomorphic_sends,
                    ) {
                        block = new_block;
                        state.stack_push(result);
                    } else {
                        // Maybe monomorphic; handled in type_specialize
                        let send = fun.push_insn(block, Insn::Send { recv, cd, block: None, args, state: exit_id, reason: Uncategorized(opcode.into()) });
                        state.stack_push(send);
                    }
                }
                YARVINSN_send => {
                    let cd: *const rb_call_data = get_arg(pc, 0).as_ptr();
                    let blockiseq: IseqPtr = get_arg(pc, 1).as_iseq();
                    let call_info = unsafe { (*cd).ci };
                    let flags = unsafe { rb_vm_ci_flag(call_info) };
                    if let Err(call_type) = unhandled_call_type(flags) {
                        // Can't handle tailcall; side-exit into the interpreter
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledCallType(call_type)), recompile: None });
                        break;  // End the block
                    }
                    // Side-exit send fallbacks while tracing to avoid FLAG_FINISH breaking throw TAG_RETURN semantics
                    if unsafe { rb_zjit_iseq_tracing_currently_enabled() } {
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::SendWhileTracing), recompile: None });
                        break;
                    }
                    let block_arg = (flags & VM_CALL_ARGS_BLOCKARG) != 0;

                    let args = state.stack_pop_n(crate::profile::num_arguments_on_stack(cd))?;
                    let recv = state.stack_pop()?;
                    let block_handler = if !blockiseq.is_null() {
                        Some(BlockHandler::BlockIseq(blockiseq))
                    } else if block_arg {
                        Some(BlockHandler::BlockArg)
                    } else {
                        None
                    };
                    // A block argument that is nil at run time is stripped by type_specialize,
                    // which is what lets `bar(&block)` become a direct send. Anything else keeps
                    // the send dynamic, so only branch on the receiver's type when the block
                    // argument has a chance of being nil.
                    let block_arg_summary = if block_arg {
                        args.last().map(|&insn| fun.profile_summary(&profiles, insn, exit_id))
                    } else {
                        None
                    };
                    let block_arg_can_be_nil = match &block_arg_summary {
                        None => true,
                        Some(summary) => summary.buckets().iter().any(|ty| !ty.is_empty() && ty.is_nil()),
                    };
                    // `def foo(&block) = bar(&block)` forwarding sites see both nil and non-nil
                    // block arguments. type_specialize can only turn `bar(&block)` into a direct
                    // send when it can prove the block argument is nil, so a site that sometimes
                    // receives a block makes every call dynamic. Branch on nil instead so the
                    // common no-block case still gets a direct send.
                    let split_nil_block_arg = blockiseq.is_null() && block_arg_can_be_nil
                        && block_arg_summary.as_ref().is_some_and(|summary| !summary.is_monomorphic());
                    // Calls with a literal block get the same receiver chain as block-less sends.
                    // Blocks are how Ruby iterates, so leaving these on the dynamic send gives up
                    // on the receiver of every polymorphic `node.each_child_node { ... }`-style
                    // call. The chain continues from its join block, so the
                    // reload_locals_modified_by_block below still covers every arm.
                    let dispatch_on_recv = block_arg_can_be_nil;

                    // `foo(&blk)` where `blk` came straight from `getblockparamproxy`: when the
                    // value really is the proxy, `vm_caller_setup_arg_block` hands the callee this
                    // frame's own block handler, which `type_specialize` can write into the callee
                    // frame without going through the interpreter. Branch on it rather than
                    // guarding, because the other side of the branch is a block param that
                    // `setblockparam` materialized, which is a legitimate value to reach here.
                    //
                    // The profile filter is exact: `rb_block_param_proxy` carries a singleton
                    // class (it defines `call` on itself), so no other object profiles with that
                    // class.
                    let proxy_class = unsafe { rb_block_param_proxy }.class_of();
                    let proxy_split = blockiseq.is_null() && block_arg
                        && !unspecializable_call_type(flags & !VM_CALL_ARGS_BLOCKARG)
                        && args.last().is_some_and(|arg| block_param_proxy_values.contains(arg))
                        && block_arg_summary.as_ref().is_some_and(|summary| summary.buckets().iter().any(|profiled_type|
                            !profiled_type.is_empty() && profiled_type.class() == proxy_class))
                        // Only an ISEQ or C callee's frame setup takes the handler; for anything
                        // else the call stays dynamic and the branch would be dead weight.
                        && profiled_recv_takes_block_handler(fun, &profiles, recv, exit_id, cd);
                    let proxy_join = if proxy_split {
                        let block_arg_insn = *args.last().unwrap();
                        let join_block = fun.new_block(insn_idx);
                        let join_param = fun.push_insn(join_block, Insn::Param);
                        let proxy_block = fun.new_block(insn_idx);
                        let rest_block = fun.new_block(insn_idx);
                        let proxy = unsafe { rb_block_param_proxy };
                        let proxy_const = fun.push_insn(block, Insn::Const { val: Const::Value(proxy) });
                        let is_proxy = fun.push_insn(block, Insn::IsBitEqual { left: block_arg_insn, right: proxy_const });
                        fun.push_insn(block, Insn::CondBranch {
                            val: is_proxy,
                            if_true: BranchEdge { target: proxy_block, args: vec![] },
                            if_false: BranchEdge { target: rest_block, args: vec![] },
                        });

                        // Refine the block argument to the proxy object so that type_specialize
                        // recognizes it and replaces it with a load of this frame's block handler.
                        let mut proxy_args = args.clone();
                        *proxy_args.last_mut().unwrap() = fun.push_insn(proxy_block, Insn::RefineType { val: block_arg_insn, new_type: Type::from_value(proxy) });
                        let (proxy_block, proxy_send) = match emit_polymorphic_send(
                            fun, &mut profiles, proxy_block, insn_idx, exit_id, &exit_state,
                            cd, recv, &proxy_args, block_handler, opcode, true,
                        ) {
                            Some(result) => result,
                            None => {
                                let send = fun.push_insn(proxy_block, Insn::Send { recv, cd, block: block_handler, args: proxy_args, state: exit_id, reason: Uncategorized(opcode.into()) });
                                (proxy_block, send)
                            }
                        };
                        fun.push_insn(proxy_block, Insn::Jump(BranchEdge { target: join_block, args: vec![proxy_send] }));
                        block = rest_block;
                        Some((join_block, join_param))
                    } else {
                        None
                    };

                    if split_nil_block_arg {
                        let block_arg_insn = *args.last().unwrap();
                        let join_block = fun.new_block(insn_idx);
                        let join_param = fun.push_insn(join_block, Insn::Param);
                        let nil_block = fun.new_block(insn_idx);
                        let other_block = fun.new_block(insn_idx);
                        let is_nil = fun.push_insn(block, Insn::HasType { val: block_arg_insn, expected: types::NilClass });
                        fun.push_insn(block, Insn::CondBranch {
                            val: is_nil,
                            if_true: BranchEdge { target: nil_block, args: vec![] },
                            if_false: BranchEdge { target: other_block, args: vec![] },
                        });

                        // Nil branch: refine the block argument so type_specialize sees it as
                        // statically nil, strips it, and emits a direct send.
                        let mut nil_args = args.clone();
                        *nil_args.last_mut().unwrap() = fun.push_insn(nil_block, Insn::RefineType { val: block_arg_insn, new_type: types::NilClass });
                        let (nil_block, nil_send) = match emit_polymorphic_send(
                            fun, &mut profiles, nil_block, insn_idx, exit_id, &exit_state,
                            cd, recv, &nil_args, block_handler, opcode, true,
                        ) {
                            Some(result) => result,
                            None => {
                                let send = fun.push_insn(nil_block, Insn::Send { recv, cd, block: block_handler, args: nil_args, state: exit_id, reason: Uncategorized(opcode.into()) });
                                (nil_block, send)
                            }
                        };
                        fun.push_insn(nil_block, Insn::Jump(BranchEdge { target: join_block, args: vec![nil_send] }));

                        let other_send = fun.push_insn(other_block, Insn::Send { recv, cd, block: block_handler, args, state: exit_id, reason: SendBlockArgNotNil });
                        fun.push_insn(other_block, Insn::Jump(BranchEdge { target: join_block, args: vec![other_send] }));

                        block = join_block;
                        state.stack_push(join_param);
                    } else if let Some((new_block, result)) = dispatch_on_recv.then(|| emit_polymorphic_send(
                        fun, &mut profiles, block, insn_idx, exit_id, &exit_state,
                        cd, recv, &args, block_handler, opcode, branch_monomorphic_sends,
                    )).flatten() {
                        block = new_block;
                        state.stack_push(result);
                    } else {
                        let send = fun.push_insn(block, Insn::Send { recv, cd, block: block_handler, args, state: exit_id, reason: Uncategorized(opcode.into()) });
                        state.stack_push(send);
                    }

                    // Rejoin the block-param-proxy arm emitted above.
                    if let Some((join_block, join_param)) = proxy_join {
                        let result = state.stack_pop()?;
                        fun.push_insn(block, Insn::Jump(BranchEdge { target: join_block, args: vec![result] }));
                        block = join_block;
                        state.stack_push(join_param);
                    }

                    if let Some(BlockHandler::BlockIseq(blockiseq)) = block_handler {
                        // Reload locals that may have been modified by the blockiseq.
                        if !ep_escaped && !state.locals.is_empty() {
                            fun.gen_post_send_no_ep_escape_patch_point(block, &state, insn_idx);
                        }
                        fun.reload_locals_modified_by_block(block, iseq, blockiseq, &mut state, ep_escaped);
                    }
                }
                YARVINSN_sendforward => {
                    let cd: *const rb_call_data = get_arg(pc, 0).as_ptr();
                    let blockiseq: IseqPtr = get_arg(pc, 1).as_iseq();
                    let call_info = unsafe { (*cd).ci };
                    let flags = unsafe { rb_vm_ci_flag(call_info) };
                    let forwarding = (flags & VM_CALL_FORWARDING) != 0;
                    if let Err(call_type) = unhandled_call_type(flags) {
                        // Can't handle the call type; side-exit into the interpreter
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledCallType(call_type)), recompile: None });
                        break;  // End the block
                    }
                    // Side-exit send fallbacks while tracing to avoid FLAG_FINISH breaking throw TAG_RETURN semantics
                    if unsafe { rb_zjit_iseq_tracing_currently_enabled() } {
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::SendWhileTracing), recompile: None });
                        break;
                    }
                    let argc = unsafe { vm_ci_argc((*cd).ci) };

                    let args = state.stack_pop_n(argc as usize + usize::from(forwarding))?;
                    let recv = state.stack_pop()?;
                    let send_forward = fun.push_insn(block, Insn::SendForward { recv, cd, blockiseq, args, state: exit_id, reason: SendForwardNotSpecialized });
                    state.stack_push(send_forward);

                    if !blockiseq.is_null() {
                        // Reload locals that may have been modified by the blockiseq.
                        if !ep_escaped && !state.locals.is_empty() {
                            fun.gen_post_send_no_ep_escape_patch_point(block, &state, insn_idx);
                        }
                        fun.reload_locals_modified_by_block(block, iseq, blockiseq, &mut state, ep_escaped);
                    }
                }
                YARVINSN_invokesuper => {
                    let cd: *const rb_call_data = get_arg(pc, 0).as_ptr();
                    let call_info = unsafe { (*cd).ci };
                    let flags = unsafe { rb_vm_ci_flag(call_info) };
                    if let Err(call_type) = unhandled_call_type(flags) {
                        // Can't handle tailcall; side-exit into the interpreter
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledCallType(call_type)), recompile: None });
                        break;  // End the block
                    }
                    // Side-exit send fallbacks while tracing to avoid FLAG_FINISH breaking throw TAG_RETURN semantics
                    if unsafe { rb_zjit_iseq_tracing_currently_enabled() } {
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::SendWhileTracing), recompile: None });
                        break;
                    }
                    let args = state.stack_pop_n(crate::profile::num_arguments_on_stack(cd))?;
                    let recv = state.stack_pop()?;
                    let blockiseq: IseqPtr = get_arg(pc, 1).as_ptr();
                    let result = fun.push_insn(block, Insn::InvokeSuper { recv, cd, blockiseq, args, state: exit_id, reason: Uncategorized(opcode.into()) });
                    state.stack_push(result);

                    if !blockiseq.is_null() {
                        // Reload locals that may have been modified by the blockiseq.
                        if !ep_escaped && !state.locals.is_empty() {
                            fun.gen_post_send_no_ep_escape_patch_point(block, &state, insn_idx);
                        }
                        fun.reload_locals_modified_by_block(block, iseq, blockiseq, &mut state, ep_escaped);
                    }
                }
                YARVINSN_invokesuperforward => {
                    let cd: *const rb_call_data = get_arg(pc, 0).as_ptr();
                    let blockiseq: IseqPtr = get_arg(pc, 1).as_iseq();
                    let call_info = unsafe { (*cd).ci };
                    let flags = unsafe { rb_vm_ci_flag(call_info) };
                    let forwarding = (flags & VM_CALL_FORWARDING) != 0;
                    if let Err(call_type) = unhandled_call_type(flags) {
                        // Can't handle tailcall; side-exit into the interpreter
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledCallType(call_type)), recompile: None });
                        break;  // End the block
                    }
                    // Side-exit send fallbacks while tracing to avoid FLAG_FINISH breaking throw TAG_RETURN semantics
                    if unsafe { rb_zjit_iseq_tracing_currently_enabled() } {
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::SendWhileTracing), recompile: None });
                        break;
                    }
                    let argc = unsafe { vm_ci_argc((*cd).ci) };
                    let args = state.stack_pop_n(argc as usize + usize::from(forwarding))?;
                    let recv = state.stack_pop()?;
                    let result = fun.push_insn(block, Insn::InvokeSuperForward { recv, cd, blockiseq, args, state: exit_id, reason: InvokeSuperForwardNotSpecialized });
                    state.stack_push(result);

                    if !blockiseq.is_null() {
                        // Reload locals that may have been modified by the blockiseq.
                        if !ep_escaped && !state.locals.is_empty() {
                            fun.gen_post_send_no_ep_escape_patch_point(block, &state, insn_idx);
                        }
                        fun.reload_locals_modified_by_block(block, iseq, blockiseq, &mut state, ep_escaped);
                    }
                }
                YARVINSN_invokeblock => {
                    let cd: *const rb_call_data = get_arg(pc, 0).as_ptr();
                    let call_info = unsafe { (*cd).ci };
                    let flags = unsafe { rb_vm_ci_flag(call_info) };
                    if let Err(call_type) = unhandled_call_type(flags) {
                        // Can't handle tailcall; side-exit into the interpreter
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledCallType(call_type)), recompile: None });
                        break;  // End the block
                    }
                    // Side-exit send fallbacks while tracing to avoid FLAG_FINISH breaking throw TAG_RETURN semantics
                    if unsafe { rb_zjit_iseq_tracing_currently_enabled() } {
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::SendWhileTracing), recompile: None });
                        break;
                    }
                    let args = state.stack_pop_n(crate::profile::num_arguments_on_stack(cd))?;

                    // The profiled block handler distribution. All the specializations below
                    // (IFUNC, inline-ISEQ, and polymorphic ISEQ dispatch) key off this summary.
                    let block_handler_summary = payload.profile.get_operand_types(exit_state.insn_idx).and_then(|types| {
                        if let [block_handler_distribution] = types {
                            Some(TypeDistributionSummary::new(block_handler_distribution))
                        } else {
                            None
                        }
                    });
                    // The monomorphic block handler class the profile recorded, if any.
                    let block_handler_class = block_handler_summary.as_ref().and_then(|summary| {
                        if !summary.is_monomorphic() { return None; }
                        Some(summary.bucket(0).class())
                    });

                    // A one-argument `yield` to a block that takes several parameters
                    // auto-splats: setup_parameters_complex()'s arg_setup_block case
                    // destructures the value with rb_check_array_type() into the block's
                    // parameters. The direct ISEQ dispatches below pass arguments in registers
                    // with none of that setup, so they only accept an exact arity match, which
                    // leaves the very common `pairs.each { |a, b| ... }` shape on the generic
                    // path. Do the destructuring here instead, so the direct dispatch sees the
                    // arity it wants.
                    //
                    // The check must be a branch, not a guard. A yielded value that is not an
                    // Array of exactly this length is perfectly legal -- it is nil-filled or
                    // truncated -- so a guard would side-exit on every call at a site that sees
                    // one, and recompiling would only speculate the same way again. The miss
                    // joins the generic `invokeblock`, which handles every shape.
                    let caller_argc = args.len();
                    let mut args = args;
                    let mut autosplat_join: Option<(BlockId, InsnId)> = None;
                    let mut call_state = exit_id;
                    if let Some(splat_num) = autosplat_direct_dispatch_arity(mode, flags, exit_state.iseq, block_handler_class, args.len()) {
                        let arg0 = args[0];
                        let join_block = fun.new_block(insn_idx);
                        let join_param = fun.push_insn(join_block, Insn::Param);
                        let length_block = fun.new_block(insn_idx);
                        let expand_block = fun.new_block(insn_idx);
                        let fallback_block = fun.new_block(insn_idx);

                        // Array subclasses and to_ary-able objects take the fallback. Both are
                        // rare at a hot yield site and modelling them here would cost a call.
                        let is_array = fun.push_insn(block, Insn::HasType { val: arg0, expected: types::ArrayExact });
                        fun.push_insn(block, Insn::CondBranch {
                            val: is_array,
                            if_true: BranchEdge { target: length_block, args: vec![] },
                            if_false: BranchEdge { target: fallback_block, args: vec![] },
                        });

                        let array = fun.push_insn(length_block, Insn::RefineType { val: arg0, new_type: types::ArrayExact });
                        let length = fun.push_insn(length_block, Insn::ArrayLength { array });
                        let expected_length = fun.push_insn(length_block, Insn::Const { val: Const::CInt64(splat_num as i64) });
                        let length_matches = fun.push_insn(length_block, Insn::IsBitEqual { left: length, right: expected_length });
                        fun.push_insn(length_block, Insn::CondBranch {
                            val: length_matches,
                            if_true: BranchEdge { target: expand_block, args: vec![] },
                            if_false: BranchEdge { target: fallback_block, args: vec![] },
                        });

                        let fallback_result = fun.push_insn(fallback_block, Insn::InvokeBlock {
                            cd, args: args.clone(), state: exit_id, reason: InvokeBlockAutosplatMiss,
                        });
                        fun.push_insn(fallback_block, Insn::Jump(BranchEdge { target: join_block, args: vec![fallback_result] }));

                        args = (0..splat_num).map(|idx| {
                            let index = fun.push_insn(expand_block, Insn::Const { val: Const::CInt64(idx as i64) });
                            fun.push_insn(expand_block, Insn::ArrayAref { array, index })
                        }).collect();
                        // The frame the callee is pushed on top of. The dispatches below derive
                        // the caller's saved SP from `state.stack().len() - args.len()`, so the
                        // stack has to end in the expanded arguments even though the interpreter
                        // only ever had the one Array there. Guards keep side-exiting to
                        // `exit_id`, which still describes the interpreter's own stack. This is
                        // the same split prepare_direct_send_args() makes for reordered kwargs.
                        let expanded_state = fun.frame_state(exit_id).with_replaced_args(&args, caller_argc);
                        call_state = fun.push_insn(expand_block, Insn::Snapshot { state: Box::new(expanded_state) });
                        // Continue the dispatch selection below in the expanded arm. It is
                        // guaranteed to pick one of the two single-ISEQ direct dispatches:
                        // autosplat_direct_dispatch_arity() only returns an arity that makes
                        // one of them eligible, and both key off this same `args.len()`.
                        block = expand_block;
                        autosplat_join = Some((join_block, join_param));
                    }

                    // Collect the profiled ISEQ blocks that can be invoked directly with a JIT-to-JIT call.
                    // `fallback_reason` narrows to the specific condition that stopped this site
                    // from specializing, so `--zjit-stats` can tell the cases apart.
                    let mut fallback_reason = if !can_direct_invoke_block(flags) {
                        InvokeBlockComplexArgs
                    } else if block_handler_summary.is_none() {
                        InvokeBlockNoProfile
                    } else {
                        InvokeBlockNotSpecialized
                    };
                    let inline_iseq = if can_direct_invoke_block(flags) {
                        block_handler_class.and_then(|obj| {
                            if unsafe { rb_IMEMO_TYPE_P(obj, imemo_iseq) == 1 } {
                                let iseq = obj.as_iseq();
                                match direct_invoke_block_adapt(iseq, args.len()) {
                                    Ok(adapt) => return Some((iseq, adapt)),
                                    Err(reason) => fallback_reason = reason,
                                }
                            } else {
                                fallback_reason = InvokeBlockHandlerNotIseqProfile;
                            }
                            None
                        })
                    } else { None };

                    // For polymorphic yield sites, collect the profiled ISEQ blocks that can
                    // dispatch directly. Iterators like Integer#times are typically called with a
                    // different block per call site, so requiring a monomorphic profile would
                    // leave every such shared yield site on the generic fallback. Buckets are
                    // ordered by frequency, so the hottest block is compared first below.
                    // Each candidate carries either the arity to auto-splat the lone yielded
                    // argument into, or the static reshape that matches its parameters. Unlike
                    // the single-ISEQ dispatches above, a candidate that needs the expansion can
                    // be mixed in freely here: the expansion's miss joins this site's own generic
                    // fallback, which still holds the unexpanded argument.
                    let mut polymorphic_iseqs: Vec<(IseqPtr, BlockDispatchArgs)> = vec![];
                    if let Some(summary) = block_handler_summary.as_ref() {
                        if can_direct_invoke_block(flags)
                            && (summary.is_megamorphic() || summary.is_skewed_megamorphic()) {
                            fallback_reason = InvokeBlockMegamorphicProfile;
                        }
                        if can_direct_invoke_block(flags)
                            && (summary.is_monomorphic() || summary.is_polymorphic() || summary.is_skewed_polymorphic()) {
                            for &profiled_type in summary.buckets() {
                                if profiled_type.is_empty() {
                                    break;
                                }
                                let obj = profiled_type.class();
                                if unsafe { rb_IMEMO_TYPE_P(obj, imemo_iseq) == 1 } {
                                    let iseq = obj.as_iseq();
                                    if polymorphic_iseqs.iter().any(|&(seen, _)| seen == iseq) {
                                        continue;
                                    }
                                    if let Ok(adapt) = direct_invoke_block_adapt(iseq, args.len()) {
                                        polymorphic_iseqs.push((iseq, BlockDispatchArgs::Adapt(adapt)));
                                    } else if args.len() == 1 {
                                        if let Some(splat_num) = block_autosplat_arity(iseq) {
                                            if direct_invoke_block_adapt(iseq, splat_num).is_ok() {
                                                polymorphic_iseqs.push((iseq, BlockDispatchArgs::AutoSplat(splat_num)));
                                            }
                                        }
                                    }
                                }
                            }
                            // Buckets that are not directly dispatchable (Proc, IFUNC and symbol
                            // handlers, blocks whose parameters don't match the yield) were skipped
                            // above and have to take the generic fallback at run time. Only keep
                            // the chain if the blocks in it account for most of the profile;
                            // otherwise every execution pays for the comparisons and still ends up
                            // in rb_vm_invokeblock.
                            let covered = summary.coverage(|_, profiled_type| {
                                !profiled_type.is_empty()
                                    && unsafe { rb_IMEMO_TYPE_P(profiled_type.class(), imemo_iseq) == 1 }
                                    && polymorphic_iseqs.iter().any(|&(seen, _)| seen == profiled_type.class().as_iseq())
                            });
                            if covered < CHAIN_COVERAGE_THRESHOLD {
                                polymorphic_iseqs.clear();
                                if !summary.is_monomorphic() {
                                    fallback_reason = InvokeBlockChainCoverage;
                                }
                            }
                        }
                    }

                    let inlined_known_block = if let AddIseqMode::Inlined { blockiseq: Some(bi), .. } = mode {
                        if can_direct_invoke_block(flags)
                            // Only methods are inlined today, so exit_state.iseq is always a method iseq and this is
                            // always 0. That matters because the emit below is guard-free and bakes in both level 0
                            // and *this* frame's block (bi) — sound only when the yield resolves to this exact frame.
                            // TODO: if block iseqs become inlinable, a yield here could resolve to an ancestor frame
                            // (level > 0). To stay guard-free we'd bake in get_lvar_level(...) as the level and fetch
                            // that ancestor's blockiseq from the inline caller chain instead of bi.
                            && get_lvar_level(exit_state.iseq) == 0 {
                            match direct_invoke_block_adapt(bi, args.len()) {
                                Ok(adapt) => Some((bi, adapt)),
                                Err(reason) => {
                                    fallback_reason = reason;
                                    None
                                }
                            }
                        } else { None }
                    } else { None };

                    // A `yield` to a block that does a non-local `return` is the worst case for
                    // the JIT-to-JIT dispatch below: the callee always ends in a `throw` that
                    // unwinds every native frame back to the interpreter. When the block is a
                    // literal of the frame we are inlined into (so `return` unwinds to exactly
                    // the frame the compiled function returns from), inline the block's body
                    // here instead; add_iseq_to_hir turns its `throw` into a plain `Return`.
                    let inlined_block_result = match (inlined_known_block, mode) {
                        (Some((bi, adapt)), AddIseqMode::Inlined { depth: 1, .. })
                            if block_return_inlinable(bi, iseq, fun.iseq()) =>
                        {
                            // The inlined body aliases its parameters to `args` positionally, and
                            // the frame it pushes is laid out from the state, so both have to be
                            // reshaped together the way an out-of-line dispatch arm would.
                            let (inline_args, inline_state) = fun.adapt_block_args(block, adapt, args.clone(), call_state);
                            inline_block_at_yield(fun, &mut profiles, &mut block, bi, &inline_args, caller_argc, inline_state, exit_id, &exit_state, insn_idx)
                        }
                        _ => None,
                    };

                    let result = if let Some(result) = inlined_block_result {
                        result
                    } else if let Some((block_iseq, adapt)) = inlined_known_block {
                        fun.push_invoke_block_iseq_direct(block, block_iseq, 0, adapt, args, call_state, exit_id, false)
                    } else if let Some((block_iseq, adapt)) = inline_iseq {
                        let level = get_lvar_level(exit_state.iseq);
                        if fun.policy.no_side_exits && autosplat_join.is_none() {
                            // This is the ISEQ's final version, so a failing guard would exit to
                            // the interpreter forever. Dispatch on a branch instead, joining on
                            // the generic fallback. The auto-splatted arm can't take this path:
                            // its arguments were expanded for the direct call, so the generic
                            // `invokeblock` there would yield the wrong argument list.
                            let (continue_block, result) = fun.dispatch_invoke_block(
                                block, insn_idx, level, cd, &[(block_iseq, adapt)], false, args, exit_id, fallback_reason);
                            block = continue_block;
                            result
                        } else {
                            fun.push_invoke_block_iseq_direct(block, block_iseq, level, adapt, args, call_state, exit_id, true)
                        }
                    } else if !polymorphic_iseqs.is_empty() {
                        // Dispatch on the runtime block ISEQ over the profiled candidates, joining
                        // on the generic fallback for anything else. Unlike the monomorphic path
                        // above, a miss must not side-exit: the site is known to see multiple
                        // blocks, so a guard would keep failing and recompiling.
                        let level = get_lvar_level(exit_state.iseq);
                        let ep = fun.get_ep(block, level);
                        let block_handler = fun.load_ep_env_field(block, ep, FieldName::VM_ENV_DATA_INDEX_SPECVAL, VM_ENV_DATA_INDEX_SPECVAL, types::CInt64);

                        let join_block = fun.new_block(insn_idx);
                        let join_param = fun.push_insn(join_block, Insn::Param);
                        let dispatch_block = fun.new_block(insn_idx);
                        let fallback_block = fun.new_block(insn_idx);

                        // The handler must be an ISEQ block: VM_BH_ISEQ_BLOCK_P is `& 0x3 == 0x1`.
                        let tag_mask = fun.push_insn(block, Insn::Const { val: Const::CInt64(0x3) });
                        let tag = fun.push_insn(block, Insn::IntAnd { left: block_handler, right: tag_mask });
                        let iseq_tag = fun.push_insn(block, Insn::Const { val: Const::CInt64(0x1) });
                        let tag_matches = fun.push_insn(block, Insn::IsBitEqual { left: tag, right: iseq_tag });
                        fun.push_insn(block, Insn::CondBranch {
                            val: tag_matches,
                            if_true: BranchEdge { target: dispatch_block, args: vec![] },
                            if_false: BranchEdge { target: fallback_block, args: vec![] },
                        });

                        // captured = block_handler & ~0x3 (struct rb_captured_block *)
                        let untag_mask = fun.push_insn(dispatch_block, Insn::Const { val: Const::CInt64(!0x3) });
                        let captured = fun.push_insn(dispatch_block, Insn::IntAnd { left: block_handler, right: untag_mask });
                        let captured_iseq = fun.load_captured_code_iseq(dispatch_block, captured);

                        let mut compare_block = dispatch_block;
                        for &(block_iseq, dispatch_args) in &polymorphic_iseqs {
                            let expected = fun.push_insn(compare_block, Insn::Const { val: Const::CPtr(block_iseq as *const u8) });
                            let iseq_matches = fun.push_insn(compare_block, Insn::IsBitEqual { left: captured_iseq, right: expected });
                            let direct_block = fun.new_block(insn_idx);
                            let miss_block = fun.new_block(insn_idx);
                            fun.push_insn(compare_block, Insn::CondBranch {
                                val: iseq_matches,
                                if_true: BranchEdge { target: direct_block, args: vec![] },
                                if_false: BranchEdge { target: miss_block, args: vec![] },
                            });
                            // This candidate takes several parameters from the one yielded
                            // value, so destructure it the way arg_setup_block would. Anything
                            // that is not an Array of exactly that length joins the generic
                            // fallback below, which handles every shape.
                            let (direct_block, call_args, call_state) = match dispatch_args {
                                BlockDispatchArgs::Adapt(adapt) => {
                                    let (call_args, call_state) = fun.adapt_block_args(direct_block, adapt, args.clone(), exit_id);
                                    (direct_block, call_args, call_state)
                                }
                                BlockDispatchArgs::AutoSplat(splat_num) => {
                                    let arg0 = args[0];
                                    let length_block = fun.new_block(insn_idx);
                                    let expand_block = fun.new_block(insn_idx);
                                    let is_array = fun.push_insn(direct_block, Insn::HasType { val: arg0, expected: types::ArrayExact });
                                    fun.push_insn(direct_block, Insn::CondBranch {
                                        val: is_array,
                                        if_true: BranchEdge { target: length_block, args: vec![] },
                                        if_false: BranchEdge { target: fallback_block, args: vec![] },
                                    });
                                    let array = fun.push_insn(length_block, Insn::RefineType { val: arg0, new_type: types::ArrayExact });
                                    let length = fun.push_insn(length_block, Insn::ArrayLength { array });
                                    let expected_length = fun.push_insn(length_block, Insn::Const { val: Const::CInt64(splat_num as i64) });
                                    let length_matches = fun.push_insn(length_block, Insn::IsBitEqual { left: length, right: expected_length });
                                    fun.push_insn(length_block, Insn::CondBranch {
                                        val: length_matches,
                                        if_true: BranchEdge { target: expand_block, args: vec![] },
                                        if_false: BranchEdge { target: fallback_block, args: vec![] },
                                    });
                                    let expanded: Vec<InsnId> = (0..splat_num).map(|idx| {
                                        let index = fun.push_insn(expand_block, Insn::Const { val: Const::CInt64(idx as i64) });
                                        fun.push_insn(expand_block, Insn::ArrayAref { array, index })
                                    }).collect();
                                    // See the single-ISEQ expansion above: the dispatch reads the
                                    // caller's saved SP off this state, so its stack has to end
                                    // in the expanded arguments.
                                    let expanded_state = fun.frame_state(exit_id).with_replaced_args(&expanded, caller_argc);
                                    let expanded_state = fun.push_insn(expand_block, Insn::Snapshot { state: Box::new(expanded_state) });
                                    (expand_block, expanded, expanded_state)
                                }
                            };
                            let direct_result = fun.push_insn(direct_block, Insn::InvokeBlockIseqDirect { iseq: block_iseq, captured, args: call_args, state: call_state });
                            fun.push_insn(direct_block, Insn::Jump(BranchEdge { target: join_block, args: vec![direct_result] }));
                            compare_block = miss_block;
                        }
                        fun.push_insn(compare_block, Insn::Jump(BranchEdge { target: fallback_block, args: vec![] }));

                        // The chain only covers the ISEQ blocks the profile saw. Everything else
                        // still gets an IFUNC test before entering the interpreter's send path:
                        // it is a tag compare on a path that was about to call
                        // rb_vm_invokeblock, and handler distributions drift once the profiling
                        // window closes.
                        let (fallback_block, fallback_result) = fun.dispatch_invoke_block(
                            fallback_block, insn_idx, level, cd, &[], true, args, exit_id,
                            InvokeBlockPolymorphicMiss);
                        fun.push_insn(fallback_block, Insn::Jump(BranchEdge { target: join_block, args: vec![fallback_result] }));

                        // Continue compilation from the join block
                        block = join_block;
                        join_param
                    } else if can_direct_invoke_block(flags) {
                        // Always test for an IFUNC handler here, even when the profile never saw
                        // one. The test is a couple of instructions on a path that would
                        // otherwise make a generic `rb_vm_invokeblock` call, and handler
                        // distributions shift after the profiling window closes: chunky-png's
                        // hottest yield site profiles ISEQ blocks only, yet 39% of its runtime
                        // handlers are IFUNCs. In inlined code the function ISEQ is the caller
                        // while `exit_state.iseq` is the callee containing this `invokeblock`.
                        let level = get_lvar_level(exit_state.iseq);
                        let (continue_block, result) = fun.dispatch_invoke_block(
                            block, insn_idx, level, cd, &[], true, args, exit_id, fallback_reason);
                        // Continue compilation from the block the dispatch ended in
                        block = continue_block;
                        result
                    } else {
                        fun.push_insn(block, Insn::InvokeBlock { cd, args, state: exit_id, reason: fallback_reason })
                    };
                    // Rejoin the arm that did not auto-splat. A non-local `return` out of an
                    // inlined block never gets here: Insn::Return terminates its own block.
                    let result = if let Some((join_block, join_param)) = autosplat_join {
                        fun.push_insn(block, Insn::Jump(BranchEdge { target: join_block, args: vec![result] }));
                        block = join_block;
                        join_param
                    } else {
                        result
                    };
                    state.stack_push(result);
                }
                YARVINSN_getglobal => {
                    let id = ID(get_arg(pc, 0).as_u64());
                    let result = fun.push_insn(block, Insn::GetGlobal { id, state: exit_id });
                    state.stack_push(result);
                }
                YARVINSN_setglobal => {
                    let id = ID(get_arg(pc, 0).as_u64());
                    let val = state.stack_pop()?;
                    fun.push_insn(block, Insn::SetGlobal { id, val, state: exit_id });
                }
                YARVINSN_getinstancevariable => {
                    let id = ID(get_arg(pc, 0).as_u64());
                    let ic = get_arg(pc, 1).as_ptr();
                    // ic is in arg 1
                    // Assume single-Ractor mode to omit gen_prepare_non_leaf_call on gen_getivar
                    // TODO: We only really need this if self_val is a class/module
                    if !fun.assume_single_ractor_mode(block, exit_id) {
                        // gen_getivar assumes single Ractor; side-exit into the interpreter
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledYARVInsn(opcode)), recompile: None });
                        break;  // End the block
                    }
                    let summary = fun.profile_summary(&profiles, self_param, exit_id);
                    let self_param = fun.guard_heap(block, self_param, exit_id);
                    // Filter out profiled types we don't care to optimize
                    let profiled_types = summary.buckets().iter().filter(|profiled_type| {
                        // Don't read past the end of the profiled types
                        !profiled_type.is_empty()
                        // Instance variable lookups on immediate values are always nil; don't bother
                        && !profiled_type.flags().is_immediate()
                        // Too-complex shapes use hash tables for ivars;
                        // rb_shape_get_iv_index doesn't work for them.
                        // Let the fallthrough GetIvar handle these.
                        && !profiled_type.shape().is_complex()
                    }).collect::<Vec<_>>();
                    // Whether every shape the profile recorded gets an arm. Filtered-out buckets
                    // and buckets we never saw (megamorphic overflow) both mean some receivers
                    // reach this site without a matching arm.
                    let covers_profile = profiled_types.len() == summary.buckets().iter().filter(|t| !t.is_empty()).count()
                        && !summary.is_megamorphic() && !summary.is_skewed_megamorphic();
                    // We might have two objects of class A and B with the same shape; de-duplicate
                    // profiled types by shape. This is just an optimization to reduce code size.
                    let mut profiled_types_unique_shapes = Vec::with_capacity(profiled_types.len());
                    for &profiled_type in profiled_types {
                        if profiled_types_unique_shapes.iter().any(|t: &ProfiledType| t.shape() == profiled_type.shape()) {
                            continue;
                        }
                        profiled_types_unique_shapes.push(profiled_type);
                    }
                    let Some((new_block, result)) = fun.dispatch_getivar(
                        &profiled_types_unique_shapes,
                        covers_profile,
                        block,
                        insn_idx,
                        self_param,
                        id,
                        ic,
                        exit_id,
                        // A receiver that misses every arm of a shape chain is not news the
                        // profiler can act on: the site is already known to be shape-polymorphic,
                        // so recompiling would rebuild the same chain out of the same buckets and
                        // exit again on the next miss. Do the access generically instead, and let
                        // the reprofile weigh the shapes for a later version.
                        ShapeMiss::CallFallback,
                    ) else {
                        // Side-exiting unconditionally; end the block
                        break;
                    };
                    block = new_block;
                    state.stack_push(result);
                }
                YARVINSN_setinstancevariable => {
                    let id = ID(get_arg(pc, 0).as_u64());
                    let ic: *const iseq_inline_iv_cache_entry = get_arg(pc, 1).as_ptr();
                    // Assume single-Ractor mode to omit gen_prepare_non_leaf_call on gen_setivar
                    // TODO: We only really need this if self_val is a class/module
                    if !fun.assume_single_ractor_mode(block, exit_id) {
                        // gen_setivar assumes single Ractor; side-exit into the interpreter
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledYARVInsn(opcode)), recompile: None });
                        break;  // End the block
                    }
                    let val = state.stack_pop()?;
                    let unrefined_self_param = self_param;
                    let summary = fun.profile_summary(&profiles, self_param, exit_id);
                    // Filter out profiled types we don't know how to optimize and de-duplicate shapes.
                    let mut seen_shapes = Vec::with_capacity(summary.buckets().len());
                    let mut specs = Vec::with_capacity(summary.buckets().len());
                    let mut unoptimized_reason = None;
                    for &profiled_type in summary.buckets() {
                        if profiled_type.is_empty() {
                            continue;
                        }
                        let shape = if profiled_type.flags().is_immediate() {
                            INVALID_SHAPE_ID
                        } else {
                            profiled_type.shape()
                        };
                        if seen_shapes.contains(&shape) {
                            continue;
                        }
                        seen_shapes.push(shape);
                        match fun.prepare_optimized_setivar(id, profiled_type) {
                            Ok(spec) => specs.push(spec),
                            Err(counter) => {
                                unoptimized_reason.get_or_insert(counter);
                            },
                        }
                    }
                    if !specs.is_empty() || unoptimized_reason.is_none() {
                        self_param = fun.guard_heap(block, self_param, exit_id);
                    }
                    // Megamorphic profiles saw more shapes than there are buckets, so the specs
                    // cannot account for every receiver that reaches this site.
                    let covers_profile = !summary.is_megamorphic() && !summary.is_skewed_megamorphic();
                    let Some(new_block) = fun.dispatch_setivar(
                        &specs,
                        unoptimized_reason,
                        covers_profile,
                        block,
                        insn_idx,
                        self_param,
                        id,
                        ic,
                        val,
                        exit_id,
                        // See the getinstancevariable case: a shape chain that misses does the
                        // access generically rather than exiting to be recompiled into the same
                        // chain.
                        ShapeMiss::CallFallback,
                    ) else {
                        // Side-exiting unconditionally; end the block.
                        break;
                    };
                    block = new_block;
                    // SetIvar will raise if self is an immediate. If it raises, we will have
                    // exited JIT code. So upgrade the type within JIT code to a heap object.
                    self_param = fun.push_insn(block, Insn::RefineType { val: unrefined_self_param, new_type: types::HeapBasicObject });
                }
                YARVINSN_getclassvariable => {
                    let id = ID(get_arg(pc, 0).as_u64());
                    let ic = get_arg(pc, 1).as_ptr();
                    let result = fun.push_insn(block, Insn::GetClassVar { id, ic, state: exit_id });
                    state.stack_push(result);
                }
                YARVINSN_setclassvariable => {
                    let id = ID(get_arg(pc, 0).as_u64());
                    let ic = get_arg(pc, 1).as_ptr();
                    let val = state.stack_pop()?;
                    fun.push_insn(block, Insn::SetClassVar { id, val, ic, state: exit_id });
                }
                YARVINSN_opt_reverse => {
                    // Reverse the order of the top N stack items.
                    let n = get_arg(pc, 0).as_usize();
                    for i in 0..n/2 {
                        let bottom = state.stack_topn(n - 1 - i)?;
                        let top = state.stack_topn(i)?;
                        state.stack_setn(i, bottom);
                        state.stack_setn(n - 1 - i, top);
                    }
                }
                YARVINSN_newrange => {
                    let flag = RangeType::from(get_arg(pc, 0).as_u32());
                    let high = state.stack_pop()?;
                    let low = state.stack_pop()?;
                    let insn_id = fun.push_insn(block, Insn::NewRange { low, high, flag, state: exit_id });
                    state.stack_push(insn_id);
                }
                YARVINSN_invokebuiltin => {
                    let bf: *const rb_builtin_function = get_arg(pc, 0).as_ptr();
                    let mut args = vec![];
                    for _ in 0..unsafe { (*bf).argc } {
                        args.push(state.stack_pop()?);
                    }
                    args.push(self_param);
                    args.reverse();

                    // Check if this builtin is annotated
                    let return_type = ZJITState::get_method_annotations()
                        .get_builtin_return_type(bf);

                    let builtin_attrs = unsafe { rb_jit_iseq_builtin_attrs(iseq) };
                    let leaf = builtin_attrs & BUILTIN_ATTR_LEAF != 0;

                    let insn_id = fun.try_inline_invoke_builtin(block, Insn::InvokeBuiltin {
                        bf,
                        recv: self_param,
                        args,
                        state: exit_id,
                        leaf,
                        return_type,
                    });
                    state.stack_push(insn_id);
                }
                YARVINSN_opt_invokebuiltin_delegate |
                YARVINSN_opt_invokebuiltin_delegate_leave => {
                    let bf: *const rb_builtin_function = get_arg(pc, 0).as_ptr();
                    let argc = unsafe { (*bf).argc } as usize;
                    let index = get_arg(pc, 1).as_usize();

                    let mut args = vec![self_param];
                    for &local in state.locals().skip(index).take(argc) {
                        args.push(local);
                    }

                    // Check if this builtin is annotated
                    let return_type = ZJITState::get_method_annotations()
                        .get_builtin_return_type(bf);

                    let builtin_attrs = unsafe { rb_jit_iseq_builtin_attrs(iseq) };
                    let leaf = builtin_attrs & BUILTIN_ATTR_LEAF != 0;

                    let insn_id = fun.try_inline_invoke_builtin(block, Insn::InvokeBuiltin {
                        bf,
                        recv: self_param,
                        args,
                        state: exit_id,
                        leaf,
                        return_type,
                    });
                    state.stack_push(insn_id);
                }
                YARVINSN_objtostring => {
                    let cd: *const rb_call_data = get_arg(pc, 0).as_ptr();
                    let argc = crate::profile::num_arguments_on_stack(cd);
                    assert_eq!(0, argc, "objtostring should not have args");
                    let recv = state.stack_pop()?;
                    // TODO(max): Handle polymorphic profiles
                    let result = if let Some(profiled_type) = fun.monomorphic_summary(&profiles, recv, exit_id) {
                        if profiled_type.is_string() {
                            // TODO(max): Do we need PatchPoint? We are checking T_STRING-ness.
                            fun.push_insn(block, Insn::PatchPoint { invariant: Invariant::NoSingletonClass { klass: profiled_type.class() }, state: exit_id });
                            fun.push_insn(block, Insn::GuardType { val: recv, guard_type: types::String, state: exit_id, recompile: None })
                        } else {
                            let recv = fun.push_insn(block, Insn::GuardType { val: recv, guard_type: Type::from_profiled_type(profiled_type), state: exit_id, recompile: None });
                            fun.push_insn(block, Insn::Send { recv, cd, block: None, args: vec![], state: exit_id, reason: ObjToStringNotString })
                        }
                    } else {
                        let has_type = fun.push_insn(block, Insn::HasType { val: recv, expected: types::String });
                        let iftrue_block = fun.new_block(insn_idx);
                        let iffalse_block = fun.new_block(insn_idx);
                        let join_block = fun.new_block(insn_idx);
                        fun.push_insn(block, Insn::CondBranch {
                            val: has_type,
                            if_true: BranchEdge { target: iftrue_block, args: vec![] },
                            if_false: BranchEdge { target: iffalse_block, args: vec![] }
                        });
                        // true block
                        let refined = fun.push_insn(iftrue_block, Insn::RefineType { val: recv, new_type: types::String });
                        fun.push_insn(iftrue_block, Insn::Jump(BranchEdge { target: join_block, args: vec![refined] }));
                        // false block
                        let refined = fun.push_insn(iffalse_block, Insn::RefineType { val: recv, new_type: types::NotString });
                        let send = fun.push_insn(iffalse_block, Insn::Send { recv: refined, cd, block: None, args: vec![], state: exit_id, reason: ObjToStringNotString });
                        fun.push_insn(iffalse_block, Insn::Jump(BranchEdge { target: join_block, args: vec![send] }));
                        // join block
                        block = join_block;
                        fun.push_insn(join_block, Insn::Param)
                    };
                    state.stack_push(result);
                }
                YARVINSN_anytostring => {
                    let str = state.stack_pop()?;
                    let val = state.stack_pop()?;

                    // Mirror logic of rb_obj_as_string_result() (`anytostring` in insns.def)
                    let has_type = fun.push_insn(block, Insn::HasType { val: str, expected: types::String });
                    let iftrue_block = fun.new_block(insn_idx);
                    let iffalse_block = fun.new_block(insn_idx);
                    let join_block = fun.new_block(insn_idx);
                    fun.push_insn(block, Insn::CondBranch {
                        val: has_type,
                        if_true: BranchEdge { target: iftrue_block, args: vec![] },
                        if_false: BranchEdge { target: iffalse_block, args: vec![] }
                    });
                    // true block
                    let refined = fun.push_insn(iftrue_block, Insn::RefineType { val: str, new_type: types::String });
                    fun.push_insn(iftrue_block, Insn::Jump(BranchEdge { target: join_block, args: vec![refined] }));
                    // false block
                    let anytostring = fun.push_insn(iffalse_block, Insn::AnyToString { val, state: exit_id });
                    fun.push_insn(iffalse_block, Insn::Jump(BranchEdge { target: join_block, args: vec![anytostring] }));
                    // join block
                    block = join_block;
                    let result = fun.push_insn(join_block, Insn::Param);
                    state.stack_push(result);
                }
                YARVINSN_getspecial => {
                    let key = get_arg(pc, 0).as_u64();
                    let svar = get_arg(pc, 1).as_u64();

                    if svar == 0 {
                        // TODO: Handle non-backref
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnknownSpecialVariable(key)), recompile: None });
                        // End the block
                        break;
                    } else if svar & 0x01 != 0 {
                        // Handle symbol backrefs like $&, $`, $', $+
                        let shifted_svar: u8 = (svar >> 1).try_into().unwrap();
                        let symbol_type = SpecialBackrefSymbol::try_from(shifted_svar).expect("invalid backref symbol");
                        let result = fun.push_insn(block, Insn::GetSpecialSymbol { symbol_type, state: exit_id });
                        state.stack_push(result);
                    } else {
                        // Handle number backrefs like $1, $2, $3
                        let result = fun.push_insn(block, Insn::GetSpecialNumber { nth: svar, state: exit_id });
                        state.stack_push(result);
                    }
                }
                YARVINSN_expandarray => {
                    let num = get_arg(pc, 0).as_u64();
                    let flag = get_arg(pc, 1).as_u64();
                    if flag != 0 {
                        // We don't (yet) handle 0x01 (rest args), 0x02 (post args), or 0x04
                        // (reverse?)
                        //
                        // Unhandled opcode; side-exit into the interpreter
                        fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledYARVInsn(opcode)), recompile: None });
                        break;  // End the block
                    }
                    let val = state.stack_pop()?;

                    // Compile the one shape the profile (or the static type) says this site has,
                    // and let a guard failure re-profile and recompile. `expandarray` pushes
                    // element num-1 first and element 0 last in every shape.
                    match fun.expandarray_shape(&profiles, val, exit_id) {
                        ExpandArrayShape::Array => {
                            let array = fun.push_insn(block, Insn::GuardType { val, guard_type: types::ArrayExact, state: exit_id, recompile: Some(Recompile) });
                            let length = fun.push_insn(block, Insn::ArrayLength { array });
                            let expected = fun.push_insn(block, Insn::Const { val: Const::CInt64(num as i64) });
                            fun.push_insn(block, Insn::GuardGreaterEq { left: length, right: expected, reason: Box::new(SideExitReason::ExpandArray), state: exit_id, recompile: Some(Recompile) });
                            for i in (0..num).rev() {
                                // We do not emit a length guard here because in-bounds is already
                                // ensured by the expandarray length check above.
                                let index = fun.push_insn(block, Insn::Const { val: Const::CInt64(i.try_into().unwrap()) });
                                let element = fun.push_insn(block, Insn::ArrayAref { array, index });
                                state.stack_push(element);
                            }
                        }
                        ExpandArrayShape::Scalar => {
                            // The value is expected to have no #to_ary, so `vm_expandarray()`
                            // treats it as the one-element array [value]: element 0 is the value
                            // itself and everything past it is nil. We still have to run the
                            // conversion, because #to_ary is looked up at run time and can be
                            // defined after this code was compiled.
                            let converted = fun.push_insn(block, Insn::CheckArrayType { val, state: exit_id });
                            fun.push_insn(block, Insn::GuardType { val: converted, guard_type: types::NilClass, state: exit_id, recompile: Some(Recompile) });
                            if num > 0 {
                                let nil = fun.push_insn(block, Insn::Const { val: Const::Value(Qnil) });
                                for _ in 0..num - 1 {
                                    state.stack_push(nil);
                                }
                                state.stack_push(val);
                            }
                        }
                        ExpandArrayShape::General => {
                            // No shape to speculate on, or no way to recompile if we guessed
                            // wrong. Let the VM's own conversion handle every case, and read the
                            // elements with ArrayArefOrNil so an array too short for its targets
                            // nil-fills instead of exiting. The indices are nonnegative constants,
                            // so they need no AdjustBounds.
                            let array = fun.push_insn(block, Insn::ToAryForExpand { val, state: exit_id });
                            let length = fun.push_insn(block, Insn::ArrayLength { array });
                            for i in (0..num).rev() {
                                let index = fun.push_insn(block, Insn::Const { val: Const::CInt64(i.try_into().unwrap()) });
                                let element = fun.push_insn(block, Insn::ArrayArefOrNil { array, index, length });
                                state.stack_push(element);
                            }
                        }
                        ExpandArrayShape::NoProfile => {
                            fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::NoProfileExpandArray), recompile: Some(Recompile) });
                            break;  // End the block
                        }
                    }
                }
                _ => {
                    // Unhandled opcode; side-exit into the interpreter
                    fun.push_insn(block, Insn::SideExit { state: exit_id, reason: Box::new(SideExitReason::UnhandledYARVInsn(opcode)), recompile: None });
                    break;  // End the block
                }
            }

            if insn_idx_to_block.contains_key(&insn_idx) {
                let target = insn_idx_to_block[&insn_idx];
                fun.push_insn(block, Insn::Jump(BranchEdge { target, args: state.as_args(self_param) }));
                queue.push_back((state, target, insn_idx, local_inval));
                break;  // End the block
            }
        }
    }

    if matches!(mode, AddIseqMode::Standalone | AddIseqMode::ExceptionEntry(_)) {
        // Populate the entries superblock with an Entries instruction targeting all entry blocks
        fun.seal_entries();

        fun.set_param_types();
        fun.infer_types();

        match get_option!(dump_hir_init) {
            Some(DumpHIR::WithoutSnapshot) => println!("Initial HIR:\n{}", FunctionPrinter::without_snapshot(fun)),
            Some(DumpHIR::All) => println!("Initial HIR:\n{}", FunctionPrinter::with_snapshot(fun)),
            Some(DumpHIR::Debug) => println!("Initial HIR:\n{:#?}", fun),
            None => {},
        }
    }
    if matches!(mode, AddIseqMode::Inlined { .. }) {
        // Materialized inlined frames also need fresh interpreter profiles.
        reset_profiles_remaining(iseq);
    }

    Ok(AddIseqResult { body_entry_block, profiles })
}

/// Compile the interpreter entry block for an exception handler entry
/// (`body->jit_exception`). Unlike the ordinary entry block, this resumes in the
/// middle of the ISEQ, so it reads back every local from the frame's EP and
/// every live VM stack slot that the interpreter's unwinder wrote before it
/// jumped to the catch-table continuation.
///
/// Returns the `FrameState` it loaded so that the continuation block is created
/// with a matching number of stack parameters.
fn compile_exception_entry_block(fun: &mut Function, entry: ExceptionEntry, target_block: BlockId) -> FrameState {
    let entry_block = fun.entry_block;
    // Codegen reads `fun.exception_entry` to emit the `cfp->pc` guard and to
    // rebase the SP register onto the frame's stack base.
    fun.push_insn(entry_block, Insn::EntryPoint { jit_entry_idx: None });

    let iseq = fun.iseq;
    let self_param = fun.load_self(entry_block);
    let mut state = FrameState::new(iseq);

    // Locals may have been escaped to the heap by a block created earlier in
    // this frame, so read them through cfp->ep rather than through the SP
    // shortcut that the method entry can use.
    let ep = fun.get_ep(entry_block, 0);
    for local_idx in 0..num_locals(iseq) {
        let ep_offset = local_idx_to_ep_offset(iseq, local_idx);
        let ep_offset = u32::try_from(ep_offset)
            .unwrap_or_else(|_| panic!("Could not convert ep_offset {ep_offset} to u32"));
        let val = fun.get_local_from_ep(entry_block, iseq, ep, ep_offset, 0, types::BasicObject);
        state.locals.push(val);
    }

    // Read back the VM stack slots that are live at the continuation. They were
    // all written to the VM stack: ZJIT frames are materialized by
    // rb_zjit_materialize_frames() during unwinding, and the throw value (if the
    // catch type pushes one) is written by vm_exec_handle_exception().
    if entry.stack_size > 0 {
        let sp = fun.load_sp(entry_block);
        for stack_idx in 0..entry.stack_size {
            let offset = i32::try_from(stack_idx * SIZEOF_VALUE)
                .unwrap_or_else(|_| panic!("Could not convert stack offset {stack_idx} to i32"));
            let val = fun.load_field(entry_block, sp, FieldName::StackSlot, offset, types::BasicObject);
            state.stack.push(val);
        }
    }

    fun.push_insn(entry_block, Insn::Jump(BranchEdge { target: target_block, args: state.as_args(self_param) }));
    state
}

/// Compile an entry_block for the interpreter
fn compile_entry_block(fun: &mut Function, jit_entry_insns: &[u32], insn_idx_to_block: &HashMap<u32, BlockId>) {
    let mut entry_block = fun.entry_block;
    let (self_param, entry_state) = compile_entry_state(fun);
    let mut pc: Option<InsnId> = None;
    let &all_opts_passed_insn_idx = jit_entry_insns.last().unwrap();

    // Check-and-jump for each missing optional PC
    let mut iter = jit_entry_insns.iter().peekable();
    while let Some(&jit_entry_insn) = iter.next() {
        if jit_entry_insn == all_opts_passed_insn_idx {
            continue;
        }
        let target_block = insn_idx_to_block.get(&jit_entry_insn)
            .copied()
            .expect("we make a block for each jump target and \
                     each entry in the ISEQ opt_table is a jump target");
        // Load PC once at the start of the block, shared among all cases
        let pc = *pc.get_or_insert_with(|| fun.load_pc(entry_block));
        let expected_pc = fun.push_insn(entry_block, Insn::Const {
            val: Const::CPtr(unsafe { rb_iseq_pc_at_idx(fun.iseq, jit_entry_insn) } as *const u8),
        });
        let test_id = fun.push_insn(entry_block, Insn::IsBitEqual { left: pc, right: expected_pc });

        let next_insn_idx = **iter.peek().expect("last entry is skipped so there is always a next");
        let fall_through = fun.new_block(next_insn_idx);

        fun.push_insn(entry_block, Insn::CondBranch {
            val: test_id,
            if_true: BranchEdge { target: target_block, args: entry_state.as_args(self_param) },
            if_false: BranchEdge { target: fall_through, args: vec![] }
        });
        entry_block = fall_through;
    }

    // Terminate the block with a jump to the block with all optionals passed
    let target_block = insn_idx_to_block.get(&all_opts_passed_insn_idx)
        .copied()
        .expect("we make a block for each jump target and \
                 each entry in the ISEQ opt_table is a jump target");
    fun.push_insn(entry_block, Insn::Jump(BranchEdge { target: target_block, args: entry_state.as_args(self_param) }));
}

/// Compile initial locals for an entry_block for the interpreter
fn compile_entry_state(fun: &mut Function) -> (InsnId, FrameState) {
    let entry_block = fun.entry_block;
    fun.push_insn(entry_block, Insn::EntryPoint { jit_entry_idx: None });

    let iseq = fun.iseq;
    let params = unsafe { iseq.params() };
    let param_size = params.size.to_usize();
    let rest_param_idx = iseq_rest_param_idx(params);

    let self_param = fun.load_self(entry_block);
    let mut entry_state = FrameState::new(iseq);
    // If the ISEQ does not escape EP, we can assume EP + 1 == SP
    // TODO: This should maybe also consider if the EP has historically been escaped in this iseq.
    // (see: https://github.com/Shopify/ruby/issues/774)
    let use_sp = !iseq_ep_starts_escaped(iseq);
    let mut base: Option<InsnId> = None;
    for local_idx in 0..num_locals(iseq) {
        if local_idx < param_size {
            let ep_offset = local_idx_to_ep_offset(iseq, local_idx);
            let ep_offset_u32 = u32::try_from(ep_offset)
                .unwrap_or_else(|_| panic!("Could not convert ep_offset {ep_offset} to u32"));
            let return_type = if Some(local_idx as i32) == rest_param_idx {
                types::ArrayExact
            } else {
                types::BasicObject
            };
            let recv = *base.get_or_insert_with(|| {
                if use_sp { fun.load_sp(entry_block) } else { fun.get_ep(entry_block, 0) }
            });
            let val = if use_sp {
                fun.get_local_from_sp(entry_block, iseq, recv, ep_offset_u32, return_type)
            } else {
                fun.get_local_from_ep(entry_block, iseq, recv, ep_offset_u32, 0, return_type)
            };
            entry_state.locals.push(val);
        } else {
            entry_state.locals.push(fun.push_insn(entry_block, Insn::Const { val: Const::Value(Qnil) }));
        }
    }
    (self_param, entry_state)
}

/// Compile a jit_entry_block
fn compile_jit_entry_block(fun: &mut Function, jit_entry_idx: usize, target_block: BlockId) {
    let jit_entry_block = fun.jit_entry_blocks[jit_entry_idx];
    fun.push_insn(jit_entry_block, Insn::EntryPoint { jit_entry_idx: Some(jit_entry_idx) });

    // Prepare entry_state with basic block params
    let (self_param, entry_state) = compile_jit_entry_state(fun, jit_entry_block, jit_entry_idx);

    if get_option!(stats) {
        fun.count_iseq_calls(jit_entry_block);
    }
    // Jump to target_block
    fun.push_insn(jit_entry_block, Insn::Jump(BranchEdge { target: target_block, args: entry_state.as_args(self_param) }));
}

/// Compile params and initial locals for a jit_entry_block
fn compile_jit_entry_state(fun: &mut Function, jit_entry_block: BlockId, jit_entry_idx: usize) -> (InsnId, FrameState) {
    let iseq = fun.iseq;
    let params = unsafe { iseq.params() };
    let param_size = params.size.to_usize();
    let opt_num: usize = params.opt_num.try_into().expect("iseq param opt_num >= 0");
    let lead_num: usize = params.lead_num.try_into().expect("iseq param lead_num >= 0");
    let passed_opt_num = jit_entry_idx;
    // We don't need to check iseq_ep_starts_escaped() because we
    // don't compile JIT entries for ISEQ_TYPE_MAIN/ISEQ_TYPE_EVAL.
    let seen_ep_escape = iseq_seen_ep_escape(iseq);

    // If the iseq has keyword parameters, the keyword bits local will be appended to the local table.
    let kw_bits_idx: Option<usize> = if unsafe { rb_get_iseq_flags_has_kw(iseq) } {
        let keyword = unsafe { rb_get_iseq_body_param_keyword(iseq) };
        if !keyword.is_null() {
            Some(unsafe { (*keyword).bits_start } as usize)
        } else {
            None
        }
    } else {
        None
    };

    let mut arg_idx: u32 = 0;
    // For `def` methods on classes that can only produce heap (non-immediate)
    // instances, `self` is a HeapBasicObject. See `iseq_self_is_heap_object`.
    let self_type = if fun.self_is_heap_object { types::HeapBasicObject } else { types::BasicObject };
    let self_param = fun.push_insn(jit_entry_block, Insn::LoadArg { idx: arg_idx, id: FieldName::SelfParam, val_type: self_type });
    arg_idx += 1;
    let mut entry_state = FrameState::new(iseq);
    let mut ep: Option<InsnId> = None;
    for local_idx in 0..num_locals(iseq) {
        let local = if (lead_num + passed_opt_num..lead_num + opt_num).contains(&local_idx) {
            // Omitted optionals are locals, so they start as nils before their code run
            fun.push_insn(jit_entry_block, Insn::Const { val: Const::Value(Qnil) })
        } else if Some(local_idx) == kw_bits_idx {
            // Read the kw_bits value written by the caller to the callee frame.
            // This tells us which optional keywords were NOT provided and need their defaults evaluated.
            // Note: The caller writes kw_bits to memory via gen_send_iseq_direct but does NOT pass it
            // as a C argument, so we must read it from EP memory rather than Param.
            let ep_offset = local_idx_to_ep_offset(iseq, local_idx);
            let ep_offset_u32 = u32::try_from(ep_offset)
                .unwrap_or_else(|_| panic!("Could not convert ep_offset {ep_offset} to u32"));
            let ep = *ep.get_or_insert_with(|| fun.get_ep(jit_entry_block, 0));
            fun.get_local_from_ep(
                jit_entry_block,
                iseq,
                ep,
                ep_offset_u32,
                0,
                types::BasicObject,
            )
        } else if local_idx < param_size {
            let id = unsafe { rb_zjit_local_id(iseq, local_idx.try_into().unwrap()) };
            let local = fun.push_insn(jit_entry_block, Insn::LoadArg { idx: arg_idx, id: id.into(), val_type: types::BasicObject });
            arg_idx += 1;
            local
        } else {
            fun.push_insn(jit_entry_block, Insn::Const { val: Const::Value(Qnil) })
        };
        entry_state.locals.push(local);

        // Once an ISEQ has escaped EP, HIR getlocal may need to read from the
        // VM frame instead of FrameState. Direct JIT-to-JIT entry passes locals
        // as C arguments, so initialize the frame slots here before such reads.
        if seen_ep_escape {
            let ep_offset = local_idx_to_ep_offset(iseq, local_idx);
            let local_id = unsafe { rb_zjit_local_id(iseq, local_idx.try_into().unwrap()) };
            let ep = *ep.get_or_insert_with(|| fun.get_ep(jit_entry_block, 0));
            fun.push_insn(jit_entry_block, Insn::StoreField {
                recv: ep,
                id: local_id.into(),
                offset: -(SIZEOF_VALUE_I32 * ep_offset),
                val: local,
                num_bits: types::BasicObject.num_bits(),
            });
        }
    }
    (self_param, entry_state)
}

pub struct Dominators {
    /// Immediate dominator for each block, indexed by BlockId.
    /// idom(root) = root (self-loop is sentinel), idom[unreachable] == IDOM_NONE.
    idoms: Vec<BlockId>,
    cfi: ControlFlowInfo,
}

/// Sentinel value for "no idom computed yet".
const IDOM_NONE: BlockId = BlockId(u32::MAX);

impl Dominators {
    pub fn new(f: &Function) -> Self {
        let cfi = ControlFlowInfo::new(f);
        Self::with_cfi(f, cfi)
    }

    /// Compute immediate dominators using the "engineered algorithm" from
    /// Cooper, Harvey & Kennedy, "A Simple, Fast Dominance Algorithm" (2001),
    /// Figure 3: <https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/dom14.pdf>
    pub fn with_cfi(f: &Function, cfi: ControlFlowInfo) -> Self {
        let rpo = cfi.reverse_post_order();
        let num_blocks = f.blocks.len();

        // Map BlockId -> RPO index for O(1) lookup in intersect.
        let mut rpo_order = vec![usize::MAX; num_blocks];
        for (idx, &block) in rpo.iter().enumerate() {
            rpo_order[block.to_usize()] = idx;
        }

        // Initialize idom: root's idom is itself, everything else is undefined.
        let mut idoms = vec![IDOM_NONE; num_blocks];
        let root = f.entries_block;
        idoms[root.to_usize()] = root;

        let mut changed = true;
        while changed {
            changed = false;
            for &block in rpo {
                if block == root { continue; }

                // Find the first predecessor that already has an idom computed.
                let preds = cfi.predecessors(block);
                let mut new_idom = IDOM_NONE;
                for &p in preds {
                    if idoms[p.to_usize()] != IDOM_NONE {
                        new_idom = p;
                        break;
                    }
                }
                if new_idom == IDOM_NONE { continue; }

                // Intersect with remaining processed predecessors.
                for &p in preds {
                    if p == new_idom { continue; }
                    if idoms[p.to_usize()] != IDOM_NONE {
                        new_idom = Self::intersect(&idoms, &rpo_order, p, new_idom);
                    }
                }

                if idoms[block.to_usize()] != new_idom {
                    idoms[block.to_usize()] = new_idom;
                    changed = true;
                }
            }
        }

        Self { idoms, cfi }
    }

    /// Walk up the dominator tree from two fingers until they meet.
    /// Uses RPO indices: a node with a *lower* RPO index is *higher* in the tree.
    fn intersect(idoms: &[BlockId], rpo_order: &[usize], mut b1: BlockId, mut b2: BlockId) -> BlockId {
        while b1 != b2 {
            while rpo_order[b1.to_usize()] > rpo_order[b2.to_usize()] {
                b1 = idoms[b1.to_usize()];
            }
            while rpo_order[b2.to_usize()] > rpo_order[b1.to_usize()] {
                b2 = idoms[b2.to_usize()];
            }
        }
        b1
    }

    /// Return the immediate dominator of `block`.
    pub fn idom(&self, block: BlockId) -> BlockId {
        self.idoms[block.to_usize()]
    }

    /// Return true if `left` is dominated by `right`.
    pub fn is_dominated_by(&self, left: BlockId, right: BlockId) -> bool {
        if self.idom(left) == IDOM_NONE { return false; }
        let mut block = left;
        loop {
            if block == right { return true; }
            if self.idom(block) == block { return false; }
            block = self.idom(block);
        }
    }

    /// Compute the full dominator set for `block` by walking the idom chain to the root.
    /// Returns dominators sorted by BlockId (ascending). Only used in tests;
    /// production code should use `idom()` or `is_dominated_by()` instead.
    pub fn dominators(&self, block: BlockId) -> Vec<BlockId> {
        let mut doms = Vec::new();
        if self.idom(block) != IDOM_NONE {
            let mut b = block;
            loop {
                doms.push(b);
                if self.idom(b) == b { break; }
                b = self.idom(b);
            }
        }
        doms.sort();
        doms
    }
}

pub struct ControlFlowInfo {
    num_blocks: usize,
    reverse_post_order: Vec<BlockId>,
    successor_map: HashMap<BlockId, Vec<BlockId>>,
    predecessor_map: HashMap<BlockId, Vec<BlockId>>,
}

impl ControlFlowInfo {
    pub fn new(function: &Function) -> Self {
        let mut successor_map: HashMap<BlockId, Vec<BlockId>> = HashMap::default();
        let mut predecessor_map: HashMap<BlockId, Vec<BlockId>> = HashMap::default();

        let reverse_post_order = function.reverse_post_order();
        for &block_id in &reverse_post_order {
            let mut successors: Vec<BlockId> = function.successors(block_id).collect();
            successors.dedup();

            // Update predecessors for successor blocks.
            for &succ_id in &successors {
                predecessor_map
                    .entry(succ_id)
                    .or_default()
                    .push(block_id);
            }

            // Store successors for this block.
            successor_map.insert(block_id, successors);
        }

        Self {
            num_blocks: function.num_blocks(),
            reverse_post_order,
            successor_map,
            predecessor_map,
        }
    }

    pub fn is_succeeded_by(&self, left: BlockId, right: BlockId) -> bool {
        self.successor_map.get(&right).is_some_and(|set| set.contains(&left))
    }

    pub fn is_preceded_by(&self, left: BlockId, right: BlockId) -> bool {
        self.predecessor_map.get(&right).is_some_and(|set| set.contains(&left))
    }

    pub fn predecessors(&self, block: BlockId) -> &[BlockId] {
        self.predecessor_map.get(&block).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn successors(&self, block: BlockId) -> &[BlockId] {
        self.successor_map.get(&block).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn reverse_post_order(&self) -> &[BlockId] {
        &self.reverse_post_order
    }
}

pub struct LoopInfo<'a> {
    cfi: &'a ControlFlowInfo,
    dominators: &'a Dominators,
    loop_depths: HashMap<BlockId, u32>,
    loop_headers: BlockSet,
    back_edge_sources: BlockSet,
}

impl<'a> LoopInfo<'a> {
    pub fn new(dominators: &'a Dominators) -> Self {
        let cfi = &dominators.cfi;
        let mut loop_headers: BlockSet = BlockSet::with_capacity(cfi.num_blocks());
        let mut loop_depths: HashMap<BlockId, u32> = HashMap::default();
        let mut back_edge_sources: BlockSet = BlockSet::with_capacity(cfi.num_blocks());
        let rpo = cfi.reverse_post_order();

        for &block in rpo {
            loop_depths.insert(block, 0);
        }

        // Collect loop headers.
        for &block in rpo {
            // Initialize the loop depths.
            for &predecessor in cfi.predecessors(block) {
                if dominators.is_dominated_by(predecessor, block) {
                    // Found a loop header, so then identify the natural loop.
                    loop_headers.insert(block);
                    back_edge_sources.insert(predecessor);
                    let loop_blocks = Self::find_natural_loop(cfi, block, predecessor);
                    // Increment the loop depth.
                    for loop_block in &loop_blocks {
                        *loop_depths.get_mut(loop_block).expect("Loop block should be populated.") += 1;
                    }
                }
            }
        }

        Self {
            cfi,
            dominators,
            loop_depths,
            loop_headers,
            back_edge_sources,
        }
    }

    fn find_natural_loop(
        cfi: &ControlFlowInfo,
        header: BlockId,
        back_edge_source: BlockId,
    ) -> HashSet<BlockId> {
        // todo(aidenfoxivey): Reimplement using BlockSet
        let mut loop_blocks = HashSet::default();
        let mut stack = vec![back_edge_source];

        loop_blocks.insert(header);
        loop_blocks.insert(back_edge_source);

        while let Some(block) = stack.pop() {
            for &pred in cfi.predecessors(block) {
                // Pushes to stack only if `pred` wasn't already in `loop_blocks`.
                if loop_blocks.insert(pred) {
                    stack.push(pred)
                }
            }
        }

        loop_blocks
    }

    pub fn loop_depth(&self, block: BlockId) -> u32 {
        self.loop_depths.get(&block).copied().unwrap_or(0)
    }

    pub fn is_back_edge_source(&self, block: BlockId) -> bool {
        self.back_edge_sources.get(block)
    }

    pub fn is_loop_header(&self, block: BlockId) -> bool {
        self.loop_headers.get(block)
    }
}

#[cfg(test)]
mod union_find_tests {
    use super::UnionFind;

    #[test]
    fn test_find_returns_self() {
        let uf = UnionFind::new();
        assert_eq!(uf.find(3usize), 3);
    }

    #[test]
    fn test_find_returns_target() {
        let mut uf = UnionFind::new();
        uf.make_equal_to(3, 4);
        assert_eq!(uf.find(3usize), 4);
    }

    #[test]
    fn test_find_with_unknown_element_returns_self() {
        let uf = UnionFind::new();
        assert_eq!(uf.find(10usize), 10);
    }

    #[test]
    fn test_find_halts_with_identity_make_equal_to() {
        let mut uf = UnionFind::<usize>::new();
        uf.make_equal_to(0, 0);
        assert_eq!(uf.find(0), 0);
    }

    #[test]
    fn test_find_returns_transitive_target() {
        let mut uf = UnionFind::new();
        uf.make_equal_to(3, 4);
        uf.make_equal_to(4, 5);
        assert_eq!(uf.find(3usize), 5);
        assert_eq!(uf.find(4usize), 5);
    }

    #[test]
    fn test_find_compresses_path() {
        let mut uf = UnionFind::new();
        uf.make_equal_to(3, 4);
        uf.make_equal_to(4, 5);
        assert_eq!(uf.at(3usize), 4);
        assert_eq!(uf.find(3usize), 5);
        assert_eq!(uf.at(3usize), 5);
    }

    #[test]
    fn test_make_equal_to_does_not_create_cycles() {
        let mut uf = UnionFind::new();
        uf.make_equal_to(3, 4);
        uf.make_equal_to(4, 5);
        uf.make_equal_to(5, 3);
        assert_eq!(uf.find(3usize), 5);
        assert_eq!(uf.find(4usize), 5);
        assert_eq!(uf.find(5usize), 5);
    }
}

#[cfg(test)]
mod rpo_tests {
    use super::*;

    #[test]
    fn one_block() {
        let mut function = Function::new(std::ptr::null());
        let entries = function.entries_block;
        let entry = function.entry_block;
        let val = function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        function.push_insn(entry, Insn::Return { val, pop_inlined_frames: 0 });
        function.seal_entries();
        assert_eq!(function.reverse_post_order(), vec![entries, entry]);
    }

    #[test]
    fn jump() {
        let mut function = Function::new(std::ptr::null());
        let entries = function.entries_block;
        let entry = function.entry_block;
        let exit = function.new_block(0);
        function.push_insn(entry, Insn::Jump(BranchEdge { target: exit, args: vec![] }));
        let val = function.push_insn(exit, Insn::Const { val: Const::Value(Qnil) });
        function.push_insn(exit, Insn::Return { val, pop_inlined_frames: 0 });
        function.seal_entries();
        assert_eq!(function.reverse_post_order(), vec![entries, entry, exit]);
    }

    #[test]
    fn diamond_iftrue() {
        let mut function = Function::new(std::ptr::null());
        let entries = function.entries_block;
        let entry = function.entry_block;
        let side = function.new_block(0);
        let exit = function.new_block(0);
        function.push_insn(side, Insn::Jump(BranchEdge { target: exit, args: vec![] }));
        let val = function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        function.push_insn(entry, Insn::CondBranch {
            val,
            if_true: BranchEdge { target: side, args: vec![] },
            if_false: BranchEdge { target: exit, args: vec![] }
        });
        let val = function.push_insn(exit, Insn::Const { val: Const::Value(Qnil) });
        function.push_insn(exit, Insn::Return { val, pop_inlined_frames: 0 });
        function.seal_entries();
        assert_eq!(function.reverse_post_order(), vec![entries, entry, side, exit]);
    }

    #[test]
    fn diamond_iffalse() {
        let mut function = Function::new(std::ptr::null());
        let entries = function.entries_block;
        let entry = function.entry_block;
        let side = function.new_block(0);
        let exit = function.new_block(0);
        function.push_insn(side, Insn::Jump(BranchEdge { target: exit, args: vec![] }));
        let val = function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        function.push_insn(entry, Insn::CondBranch {
            val,
            if_true: BranchEdge { target: exit, args: vec![] },
            if_false: BranchEdge { target: side, args: vec![] },
        });
        let val = function.push_insn(exit, Insn::Const { val: Const::Value(Qnil) });
        function.push_insn(exit, Insn::Return { val, pop_inlined_frames: 0 });
        function.seal_entries();
        assert_eq!(function.reverse_post_order(), vec![entries, entry, side, exit]);
    }

    #[test]
    fn a_loop() {
        let mut function = Function::new(std::ptr::null());
        let entries = function.entries_block;
        let entry = function.entry_block;
        function.push_insn(entry, Insn::Jump(BranchEdge { target: entry, args: vec![] }));
        function.seal_entries();
        assert_eq!(function.reverse_post_order(), vec![entries, entry]);
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[track_caller]
    fn assert_matches_err(res: Result<(), ValidationError>, expected: ValidationError) {
        match res {
            Err(validation_err) => {
                assert_eq!(validation_err, expected);
            }
            Ok(_) => panic!("Expected validation error"),
        }
    }

    #[test]
    fn one_block_no_terminator() {
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        function.seal_entries();
        assert_matches_err(function.validate(), ValidationError::BlockHasNoTerminator(entry));
    }

    #[test]
    fn one_block_terminator_not_at_end() {
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let val = function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        let insn_id = function.push_insn(entry, Insn::Return { val, pop_inlined_frames: 0 });
        function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        function.push_insn(entry, Insn::Unreachable);
        function.seal_entries();
        assert_matches_err(function.validate(), ValidationError::TerminatorNotAtEnd(entry, insn_id, 1));
    }

    #[test]
    fn iftrue_mismatch_args() {
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let side = function.new_block(0);
        let val = function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        let fall_through = function.new_block(1);
        function.push_insn(fall_through, Insn::Unreachable);
        function.push_insn(side, Insn::Unreachable);
        function.push_insn(entry, Insn::CondBranch {
            val,
            if_true: BranchEdge { target: side, args: vec![val, val, val] },
            if_false: BranchEdge { target: fall_through, args: vec![] }
        });
        function.seal_entries();
        assert_matches_err(function.validate(), ValidationError::MismatchedBlockArity(entry, 0, 3));
    }

    #[test]
    fn iffalse_mismatch_args() {
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let side = function.new_block(0);
        let val = function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        let fall_through = function.new_block(1);
        function.push_insn(fall_through, Insn::Unreachable);
        function.push_insn(side, Insn::Unreachable);
        function.push_insn(entry, Insn::CondBranch {
            val,
            if_true: BranchEdge { target: fall_through, args: vec![] },
            if_false: BranchEdge { target: side, args: vec![val, val, val] },
        });
        function.seal_entries();
        assert_matches_err(function.validate(), ValidationError::MismatchedBlockArity(entry, 0, 3));
    }

    #[test]
    fn jump_mismatch_args() {
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let side = function.new_block(0);
        let val = function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        function.push_insn(entry, Insn::Jump ( BranchEdge { target: side, args: vec![val, val, val] } ));
        function.push_insn(side, Insn::Unreachable);
        function.seal_entries();
        assert_matches_err(function.validate(), ValidationError::MismatchedBlockArity(entry, 0, 3));
    }

    #[test]
    fn not_defined_within_bb() {
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        // Create an instruction without making it belong to anything.
        let dangling = function.new_insn(Insn::Const{val: Const::CBool(true)});
        let val = function.push_insn(function.entry_block, Insn::ArrayDup { val: dangling, state: InsnId(0) });
        function.push_insn(function.entry_block, Insn::Unreachable);
        function.seal_entries();
        assert_matches_err(function.validate_definite_assignment(), ValidationError::OperandNotDefined(entry, val, dangling));
    }

    #[test]
    fn using_non_output_insn() {
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let const_ = function.push_insn(function.entry_block, Insn::Const{val: Const::CBool(true)});
        // Ret is a non-output instruction.
        let ret = function.push_insn(function.entry_block, Insn::Return { val: const_, pop_inlined_frames: 0 });
        let val = function.push_insn(function.entry_block, Insn::ArrayDup { val: ret, state: InsnId(0) });
        function.push_insn(function.entry_block, Insn::Unreachable);
        function.seal_entries();
        assert_matches_err(function.validate_definite_assignment(), ValidationError::OperandNotDefined(entry, val, ret));
    }

    #[test]
    fn not_dominated_by_diamond() {
        // This tests that one branch is missing a definition which fails.
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let side = function.new_block(0);
        let exit = function.new_block(0);
        let v0 = function.push_insn(side, Insn::Const { val: Const::Value(VALUE::fixnum_from_usize(3)) });
        function.push_insn(side, Insn::Jump(BranchEdge { target: exit, args: vec![] }));
        let val1 = function.push_insn(entry, Insn::Const { val: Const::CBool(false) });
        function.push_insn(entry, Insn::CondBranch {
            val: val1,
            if_true: BranchEdge { target: exit, args: vec![] },
            if_false: BranchEdge { target: side, args: vec![] },
        });
        let val2 = function.push_insn(exit, Insn::ArrayDup { val: v0, state: v0 });
        let const_ = function.push_insn(exit, Insn::Const{val: Const::CBool(true)});
        function.push_insn(exit, Insn::Return { val: const_, pop_inlined_frames: 0 });

        function.seal_entries();
        crate::cruby::with_rubyvm(|| {
            function.infer_types();
            assert_matches_err(function.validate_definite_assignment(), ValidationError::OperandNotDefined(exit, val2, v0));
        });
    }

    #[test]
    fn dominated_by_diamond() {
        // This tests that both branches with a definition succeeds.
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let side = function.new_block(0);
        let exit = function.new_block(0);
        let v0 = function.push_insn(entry, Insn::Const { val: Const::Value(VALUE::fixnum_from_usize(3)) });
        function.push_insn(side, Insn::Jump(BranchEdge { target: exit, args: vec![] }));
        let val = function.push_insn(entry, Insn::Const { val: Const::CBool(false) });
        function.push_insn(entry, Insn::CondBranch {
            val,
            if_true: BranchEdge { target: exit, args: vec![] },
            if_false: BranchEdge { target: side, args: vec![] }
        });
        let _val = function.push_insn(exit, Insn::ArrayDup { val: v0, state: v0 });
        let const_ = function.push_insn(exit, Insn::Const{val: Const::CBool(true)});
        function.push_insn(exit, Insn::Return { val: const_, pop_inlined_frames: 0 });
        function.seal_entries();
        crate::cruby::with_rubyvm(|| {
            function.infer_types();
            // Just checking that we don't panic.
            assert!(function.validate_definite_assignment().is_ok());
        });
    }

    #[test]
    fn instruction_appears_twice_in_same_block() {
        let mut function = Function::new(std::ptr::null());
        let block = function.new_block(0);
        function.push_insn(function.entry_block, Insn::Jump(BranchEdge { target: block, args: vec![] }));
        let val = function.push_insn(block, Insn::Const { val: Const::Value(Qnil) });
        function.push_insn_id(block, val);
        function.push_insn(block, Insn::Return { val, pop_inlined_frames: 0 });
        function.seal_entries();
        assert_matches_err(function.validate(), ValidationError::DuplicateInstruction(block, val));
    }

    #[test]
    fn instruction_appears_twice_with_different_ids() {
        let mut function = Function::new(std::ptr::null());
        let block = function.new_block(0);
        function.push_insn(function.entry_block, Insn::Jump(BranchEdge { target: block, args: vec![] }));
        let val0 = function.push_insn(block, Insn::Const { val: Const::Value(Qnil) });
        let val1 = function.push_insn(block, Insn::Const { val: Const::Value(Qnil) });
        function.make_equal_to(val1, val0);
        function.push_insn(block, Insn::Return { val: val0, pop_inlined_frames: 0 });
        function.seal_entries();
        assert_matches_err(function.validate(), ValidationError::DuplicateInstruction(block, val0));
    }

    #[test]
    fn instruction_appears_twice_in_different_blocks() {
        let mut function = Function::new(std::ptr::null());
        let block = function.new_block(0);
        function.push_insn(function.entry_block, Insn::Jump(BranchEdge { target: block, args: vec![] }));
        let val = function.push_insn(block, Insn::Const { val: Const::Value(Qnil) });
        let exit = function.new_block(0);
        function.push_insn(block, Insn::Jump(BranchEdge { target: exit, args: vec![] }));
        function.push_insn_id(exit, val);
        function.push_insn(exit, Insn::Return { val, pop_inlined_frames: 0 });
        function.seal_entries();
        assert_matches_err(function.validate(), ValidationError::DuplicateInstruction(exit, val));
    }

    // The heap-fields pointer (`as_heap`, a CPtr) and the first embedded
    // instance variable both live at ROBJECT_OFFSET_AS_HEAP_FIELDS ==
    // ROBJECT_OFFSET_AS_ARY == 0x10 on a Ruby object. They are distinct fields
    // with incompatible value types that happen to share a base and an offset.
    // Since we could end up with two `LoadField` on different shape types
    // (e.g., as the result of inlining), `optimize_load_store` must not satisfy
    // one load from another cached load with a different return type. The fault
    // surfaces here as the forwarded value flowing into a `Return` with the
    // wrong type (`CPtr` rather than `BasicObject`).
    #[test]
    fn optimize_load_store_does_not_alias_loads_with_incompatible_return_types() {
        assert_eq!(ROBJECT_OFFSET_AS_HEAP_FIELDS, ROBJECT_OFFSET_AS_ARY,
            "Conflicting field offsets changed, rendering the rest of this test incorrect");

        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let recv = function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        function.load_field(entry, recv, FieldName::as_heap, ROBJECT_OFFSET_AS_HEAP_FIELDS, types::CPtr);
        let ivar = function.load_field(entry, recv, FieldName::Id(ID(1)), ROBJECT_OFFSET_AS_ARY, types::BasicObject);
        function.push_insn(entry, Insn::Return { val: ivar, pop_inlined_frames: 0 });
        function.seal_entries();

        function.infer_types();
        function.optimize_load_store();

        assert!(
            function.validate().is_ok(),
            "optimize_load_store aliased two loads with different return types: {:?}",
            function.validate(),
        );
    }

    #[test]
    fn optimize_load_store_does_not_alias_loads_with_compatible_return_types() {
        assert_eq!(ROBJECT_OFFSET_AS_HEAP_FIELDS, ROBJECT_OFFSET_AS_ARY,
                   "Conflicting field offsets changed, rendering the rest of this test incorrect");

        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let recv = function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        function.load_field(entry, recv, FieldName::as_heap, ROBJECT_OFFSET_AS_HEAP_FIELDS, types::BasicObject);
        let ivar = function.load_field(entry, recv, FieldName::Id(ID(1)), ROBJECT_OFFSET_AS_ARY, types::Array);
        function.push_insn(entry, Insn::Return { val: ivar, pop_inlined_frames: 0 });
        function.seal_entries();

        function.infer_types();
        function.optimize_load_store();

        assert!(
            function.validate().is_ok(),
            "optimize_load_store failed to alias two loads with different, but compatible, return types: {:?}",
            function.validate(),
        );
    }
}

#[cfg(test)]
mod infer_tests {
    use super::*;

    #[track_caller]
    fn assert_subtype(left: Type, right: Type) {
        assert!(left.is_subtype(right), "{left} is not a subtype of {right}");
    }

    #[track_caller]
    fn assert_bit_equal(left: Type, right: Type) {
        assert!(left.bit_equal(right), "{left} != {right}");
    }

    #[test]
    fn test_const() {
        let mut function = Function::new(std::ptr::null());
        let val = function.push_insn(function.entry_block, Insn::Const { val: Const::Value(Qnil) });
        function.push_insn(function.entry_block, Insn::Unreachable);
        assert_bit_equal(function.infer_type(val), types::NilClass);
    }

    #[test]
    fn test_nil() {
        crate::cruby::with_rubyvm(|| {
            let mut function = Function::new(std::ptr::null());
            let nil = function.push_insn(function.entry_block, Insn::Const { val: Const::Value(Qnil) });
            let val = function.push_insn(function.entry_block, Insn::Test { val: nil });
            function.push_insn(function.entry_block, Insn::Unreachable);
            function.seal_entries();
            function.infer_types();
            assert_bit_equal(function.type_of(val), Type::from_cbool(false));
        });
    }

    #[test]
    fn test_false() {
        crate::cruby::with_rubyvm(|| {
            let mut function = Function::new(std::ptr::null());
            let false_ = function.push_insn(function.entry_block, Insn::Const { val: Const::Value(Qfalse) });
            let val = function.push_insn(function.entry_block, Insn::Test { val: false_ });
            function.push_insn(function.entry_block, Insn::Unreachable);
            function.seal_entries();
            function.infer_types();
            assert_bit_equal(function.type_of(val), Type::from_cbool(false));
        });
    }

    #[test]
    fn test_truthy() {
        crate::cruby::with_rubyvm(|| {
            let mut function = Function::new(std::ptr::null());
            let true_ = function.push_insn(function.entry_block, Insn::Const { val: Const::Value(Qtrue) });
            let val = function.push_insn(function.entry_block, Insn::Test { val: true_ });
            function.push_insn(function.entry_block, Insn::Unreachable);
            function.seal_entries();
            function.infer_types();
            assert_bit_equal(function.type_of(val), Type::from_cbool(true));
        });
    }

    #[test]
    fn newarray() {
        let mut function = Function::new(std::ptr::null());
        // Fake FrameState index of 0usize
        let val = function.push_insn(function.entry_block, Insn::NewArray { elements: vec![], state: InsnId(0) });
        assert_bit_equal(function.infer_type(val), types::ArrayExact);
    }

    #[test]
    fn arraydup() {
        let mut function = Function::new(std::ptr::null());
        // Fake FrameState index of 0usize
        let arr = function.push_insn(function.entry_block, Insn::NewArray { elements: vec![], state: InsnId(0) });
        let val = function.push_insn(function.entry_block, Insn::ArrayDup { val: arr, state: InsnId(0) });
        assert_bit_equal(function.infer_type(val), types::ArrayExact);
    }

    #[test]
    fn diamond_iffalse_merge_fixnum() {
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let side = function.new_block(0);
        let exit = function.new_block(0);
        let v0 = function.push_insn(side, Insn::Const { val: Const::Value(VALUE::fixnum_from_usize(3)) });
        function.push_insn(side, Insn::Jump(BranchEdge { target: exit, args: vec![v0] }));
        let val = function.push_insn(entry, Insn::Const { val: Const::CBool(false) });
        let v1 = function.push_insn(entry, Insn::Const { val: Const::Value(VALUE::fixnum_from_usize(4)) });
        function.push_insn(entry, Insn::CondBranch {
            val,
            if_true: BranchEdge { target: exit, args: vec![v1] },
            if_false: BranchEdge { target: side, args: vec![] },
        });
        let param = function.push_insn(exit, Insn::Param);
        function.push_insn(exit, Insn::Unreachable);
        function.seal_entries();
        crate::cruby::with_rubyvm(|| {
            function.infer_types();
        });
        assert_bit_equal(function.type_of(param), Type::fixnum(3));
    }

    #[test]
    fn self_loop_param_rotation_reaches_full_union() {
        // bb_entry:  jump bb_loop(c1, c2, c3, c4)   // 4 distinct types
        // bb_loop(p1, p2, p3, p4):
        //   jump bb_loop(p2, p3, p4, p1)            // 4-cycle rotation
        //
        // Every param transitively flows into every other across enough trips
        // around the loop, so the fixpoint for every param is the full union
        // of all four input types. The fixpoint loop must not exit while a
        // branch arm is still widening a param's type.
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let loop_block = function.new_block(0);

        let c1 = function.push_insn(entry, Insn::Const { val: Const::Value(Qtrue) });
        let c2 = function.push_insn(entry, Insn::Const { val: Const::Value(Qfalse) });
        let c3 = function.push_insn(entry, Insn::Const { val: Const::Value(Qnil) });
        let c4 = function.push_insn(entry, Insn::Const { val: Const::Value(VALUE::fixnum_from_usize(7)) });
        function.push_insn(entry, Insn::Jump(BranchEdge {
            target: loop_block,
            args: vec![c1, c2, c3, c4],
        }));

        let p1 = function.push_insn(loop_block, Insn::Param);
        let p2 = function.push_insn(loop_block, Insn::Param);
        let p3 = function.push_insn(loop_block, Insn::Param);
        let p4 = function.push_insn(loop_block, Insn::Param);
        function.push_insn(loop_block, Insn::Jump(BranchEdge {
            target: loop_block,
            args: vec![p2, p3, p4, p1],
        }));

        function.seal_entries();
        crate::cruby::with_rubyvm(|| {
            function.infer_types();
        });

        let full = types::TrueClass
            .union(types::FalseClass)
            .union(types::NilClass)
            .union(types::Fixnum);
        assert_bit_equal(function.type_of(p1), full);
        assert_bit_equal(function.type_of(p2), full);
        assert_bit_equal(function.type_of(p3), full);
        assert_bit_equal(function.type_of(p4), full);
    }

    #[test]
    fn diamond_iffalse_merge_bool() {
        let mut function = Function::new(std::ptr::null());
        let entry = function.entry_block;
        let side = function.new_block(0);
        let exit = function.new_block(0);
        let v0 = function.push_insn(side, Insn::Const { val: Const::Value(Qtrue) });
        function.push_insn(side, Insn::Jump(BranchEdge { target: exit, args: vec![v0] }));
        let val = function.push_insn(entry, Insn::Const { val: Const::CBool(false) });
        let v1 = function.push_insn(entry, Insn::Const { val: Const::Value(Qfalse) });
        function.push_insn(entry, Insn::CondBranch {
            val,
            if_true: BranchEdge { target: exit, args: vec![v1] },
            if_false: BranchEdge { target: side, args: vec![] },
        });
        let param = function.push_insn(exit, Insn::Param);
        function.push_insn(exit, Insn::Unreachable);
        function.seal_entries();
        crate::cruby::with_rubyvm(|| {
            function.infer_types();
            assert_bit_equal(function.type_of(param), types::TrueClass);
        });
    }
}
