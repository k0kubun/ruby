//! What a paused compilation assumed about the VM, and how it learns that an
//! assumption stopped holding.
//!
//! [`crate::bgcompile`] compiles the middle of each ISEQ without the GVL, so the
//! application thread runs arbitrary Ruby between the moment the compiler reads
//! the VM and the moment its code is installed. Anything it read can change in
//! that window, which the old single-critical-section design made impossible.
//!
//! The set of things that *can* change is not open-ended: it is exactly the set of
//! assumptions ZJIT already guards with patch points, because that is what a patch
//! point is for. Installed code survives a broken assumption by being rewritten to
//! side-exit; code that is not installed yet cannot be rewritten, so it has to be
//! thrown away instead.
//!
//! Rather than re-derive each assumption at install time -- which would mean
//! re-running method lookups and subclass walks, i.e. redoing the expensive part of
//! HIR construction -- the invalidation hooks tell us. Every hook already runs on a
//! GVL-holding thread and already knows the key it is invalidating (a CME, a method
//! name, a `(class, BOP)` pair). Each one now also calls [`note_invalidation`],
//! which poisons the in-flight snapshot if that key is one it recorded. The install
//! step then only has to check one boolean.
//!
//! This makes the check exact in the direction that matters: a compilation is
//! discarded only if something it actually assumed changed. A `def` of an unrelated
//! method, which `rb_clear_method_cache` reports on every method definition,
//! discards nothing.
//!
//! # What is recorded
//!
//! One [`Assumption`] per [`crate::hir::Invariant`] the HIR carries, taken from the
//! HIR rather than from the emitted patch points: the HIR is a superset (lowering
//! can drop a patch point, never add one), and being a superset is the safe
//! direction -- an extra key costs a discard we did not strictly need, a missing one
//! would install code whose guard was never armed.
//!
//! Four invariants are deliberately *not* recorded: `NoTracePoint`,
//! `NoNewObjHook`, `SingleRactorMode` and `RootBoxOnly`. Their hooks fire on events
//! that are global, rare, and broad enough that a paused compilation should be
//! discarded whether or not it named them, which is what
//! [`crate::bgcompile::note_invalidation_all`] does. Keying them would mean
//! trusting that a compilation always carries the matching patch point, and nothing
//! establishes that.
//!
//! Three keys are not invariants at all: [`Assumption::Iseq`],
//! [`Assumption::Cme`] and [`Assumption::Klass`] also stand for "this object must
//! still exist". A compilation holds raw pointers to ISEQs it inlined or calls, to
//! the CMEs behind those calls, and to classes it bakes into guards. Nothing marks
//! those for the duration of the GVL-free phase, so if the GC frees one we discard
//! rather than bake a dangling pointer into machine code.
//!
//! What *is* kept alive without a key: the ISEQ being compiled, which
//! [`crate::bgcompile::mark`] marks, and through it everything the GC reaches from
//! it -- its literal pool (every `putobject`/`putstring` constant), its
//! instruction sequence (the block and inlined-callee ISEQs its bytecode names),
//! and its ZJIT payload, whose profile marking keeps alive every class the compiler
//! specialized on. Compaction is handled separately and bluntly: it can move any of
//! those, so [`note_compaction`] poisons unconditionally.

use crate::cruby::{ID, IseqPtr, RedefinitionFlag, VALUE, rb_callable_method_entry_t, ruby_basic_operators};
use crate::hir::Invariant;

/// One thing a paused compilation needs to still be true when it installs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Assumption {
    /// A basic operator is not redefined for a class.
    Bop(RedefinitionFlag, ruby_basic_operators),
    /// A callable method entry is neither invalidated nor freed.
    Cme(*const rb_callable_method_entry_t),
    /// Nothing has changed what a method name resolves to, anywhere.
    MethodLookup(ID),
    /// A constant name has not been written to.
    Constant(ID),
    /// An ISEQ has still never escaped its environment pointer.
    NoEpEscape(IseqPtr),
    /// No object of a class has a singleton class.
    NoSingletonClass(VALUE),
    /// An ISEQ the compilation holds a pointer to has not been freed. Not an
    /// invariant of the compiled code -- see the module docs.
    Iseq(IseqPtr),
    /// A class the compilation holds a pointer to has not been freed.
    Klass(VALUE),
}

impl Assumption {
    /// Bit for [`Assumptions::kinds`]. Distinguishing kinds lets a hook reject a
    /// snapshot that assumed nothing of its kind without scanning the list, which
    /// matters for `rb_zjit_method_lookup_changed`: it runs on every `def`.
    fn kind_bit(&self) -> u16 {
        match self {
            Assumption::Bop(..) => 1 << 0,
            Assumption::Cme(_) => 1 << 1,
            Assumption::MethodLookup(_) => 1 << 2,
            Assumption::Constant(_) => 1 << 3,
            Assumption::NoEpEscape(_) => 1 << 4,
            Assumption::NoSingletonClass(_) => 1 << 5,
            Assumption::Iseq(_) => 1 << 6,
            Assumption::Klass(_) => 1 << 7,
        }
    }
}

/// Everything one paused compilation assumed, plus whether it still holds.
#[derive(Default, Debug)]
pub struct Assumptions {
    /// Deduplicated, in no particular order. A compiled ISEQ carries a handful to
    /// a few dozen of these, so a linear scan beats hashing: the scan runs on the
    /// invalidation hooks, which want to answer "not mine" as cheaply as possible.
    list: Vec<Assumption>,

    /// Union of [`Assumption::kind_bit`] over `list`.
    kinds: u16,

    /// Set once an invalidation landed that this snapshot assumed away. Never
    /// cleared: a poisoned compilation is discarded, not retried.
    poisoned: bool,
}

impl Assumptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    fn push(&mut self, assumption: Assumption) {
        if self.list.contains(&assumption) {
            return;
        }
        self.kinds |= assumption.kind_bit();
        self.list.push(assumption);
    }

    /// Record an ISEQ this compilation holds a pointer to.
    pub fn add_iseq(&mut self, iseq: IseqPtr) {
        if !iseq.is_null() {
            self.push(Assumption::Iseq(iseq));
        }
    }

    /// Record everything one HIR patch point assumes, including the objects it
    /// needs to outlive the compilation.
    pub fn add_invariant(&mut self, invariant: Invariant) {
        match invariant {
            Invariant::BOPRedefined { klass, bop } => {
                self.push(Assumption::Bop(klass, bop));
            }
            Invariant::MethodRedefined { klass, method: _, cme } => {
                self.push(Assumption::Cme(cme));
                self.push(Assumption::Klass(klass));
            }
            Invariant::NoMethodOverride { klass, method, cme } => {
                // Two separate assumptions: the resolved entry is still valid, and
                // nothing below `klass` has started overriding `method`. The second
                // is what `rb_zjit_method_lookup_changed` reports, keyed by name.
                self.push(Assumption::MethodLookup(method));
                self.push(Assumption::Cme(cme));
                self.push(Assumption::Klass(klass));
            }
            Invariant::StableConstantNames { idlist } => {
                // The list is NUL-terminated and lives in the ISEQ's constant
                // cache, so it is readable for as long as the ISEQ is.
                let mut idx = 0;
                loop {
                    let id = unsafe { *idlist.wrapping_add(idx) };
                    if id.0 == 0 {
                        break;
                    }
                    self.push(Assumption::Constant(id));
                    idx += 1;
                }
            }
            // The invariants whose hooks are global rather than keyed --
            // TracePoint being enabled, a second ractor, a non-root box, a NEWOBJ
            // hook -- need nothing recorded: [`crate::bgcompile::note_invalidation_all`]
            // discards every paused compilation, which is the right answer for an
            // event this broad and this rare.
            Invariant::NoTracePoint
            | Invariant::NoNewObjHook
            | Invariant::SingleRactorMode
            | Invariant::RootBoxOnly => {}
            Invariant::NoEPEscape(iseq) => {
                self.push(Assumption::NoEpEscape(iseq));
                self.add_iseq(iseq);
            }
            Invariant::NoSingletonClass { klass } => {
                self.push(Assumption::NoSingletonClass(klass));
                self.push(Assumption::Klass(klass));
            }
        }
    }

    /// Whether `assumption` is one of ours. The kind check is the fast path.
    fn assumes(&self, assumption: Assumption) -> bool {
        self.kinds & assumption.kind_bit() != 0 && self.list.contains(&assumption)
    }

    /// Poison if this snapshot assumed `assumption` away.
    pub fn note(&mut self, assumption: Assumption) {
        if self.assumes(assumption) {
            self.poisoned = true;
        }
    }
}
