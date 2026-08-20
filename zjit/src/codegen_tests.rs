#![cfg(test)]

use super::{gen_insn, JITState};
use crate::asm::CodeBlock;
use crate::backend::lir::Assembler;
use crate::codegen::max_iseq_versions;
use crate::cruby::*;
use crate::hir::{Insn, iseq_to_hir};
use crate::options::{get_option, rb_zjit_prepare_options, set_call_threshold, set_inline_budget, set_inline_threshold, set_max_versions, set_mem_bytes};
use crate::payload::IseqVersion;
use crate::hir::tests::hir_build_tests::assert_contains_opcode;
use crate::payload::*;
use insta::assert_snapshot;

/// Run the Ruby fragment with the inliner enabled with the default inline
/// threshold for tests. Most inliner tests should use this. `with_inlining_threshold`
/// exists if you need to customize the inline threshold for a given test.
#[track_caller]
fn with_inlining<T>(ruby_fragment: impl FnMut() -> T) -> T {
    // 30 will compile common, smaller methods while not compiling the whole world.
    with_inlining_threshold(30, ruby_fragment)
}

/// Run the Ruby fragment with the inliner enabled with the given inline `threshold`.
#[track_caller]
fn with_inlining_threshold<T>(threshold: usize, mut ruby_fragment: impl FnMut() -> T) -> T {
    with_rubyvm(|| {
        let old_inline_threshold = get_option!(inline_threshold);
        let old_call_threshold = unsafe { crate::options::rb_zjit_call_threshold };

        set_inline_threshold(threshold);
        set_call_threshold(2);
        let result = ruby_fragment();
        set_call_threshold(old_call_threshold);
        set_inline_threshold(old_inline_threshold);

        result
    })
}

/// Like `assert_compiles`, but also asserts that the program inlined at least one method
/// while running. Inliner tests must call the entry method enough times to cross the call
/// threshold, otherwise the method is never compiled and the test code ends up running in
/// interpreter. Asserting on `inline_method_count` fails the test in that case.
#[track_caller]
fn assert_inlines(program: &str) -> String {
    let counters = crate::state::ZJITState::get_counters();
    let inline_count_before = counters.inline_method_count;
    let result = assert_compiles(program);
    assert!(counters.inline_method_count > inline_count_before,
        "expected the program to inline at least one method, but inline_method_count did not increase");
    result
}

/// Like `assert_inlines`, but tolerates side exits. Use for inliner tests whose
/// inlined body legitimately exits at runtime, such as a `break` that unwinds out
/// of a literal block.
#[track_caller]
fn assert_inlines_allowing_exits(program: &str) -> String {
    let counters = crate::state::ZJITState::get_counters();
    let inline_count_before = counters.inline_method_count;
    let result = assert_compiles_allowing_exits(program);
    assert!(counters.inline_method_count > inline_count_before,
        "expected the program to inline at least one method, but inline_method_count did not increase");
    result
}

#[test]
fn test_breakpoint_hir_codegen() {
    rb_zjit_prepare_options();

    eval("def test_breakpoint_hir_codegen = nil");
    let iseq = crate::cruby::with_rubyvm(|| get_method_iseq("self", "test_breakpoint_hir_codegen"));
    unsafe { crate::cruby::rb_zjit_profile_disable(iseq) };
    let mut function = iseq_to_hir(iseq).unwrap();
    let breakpoint = function.push_insn(function.entries_block, Insn::BreakPoint);

    let mut jit = JITState::new(
        IseqVersion::new(iseq),
        function.num_insns(),
        function.num_blocks(),
        0,
    );
    let mut asm = Assembler::new();
    asm.new_block_without_id("test");
    let mut cb = CodeBlock::new_dummy();

    gen_insn(&mut cb, &mut jit, &mut asm, &function, breakpoint, &function.find(breakpoint)).unwrap();
    asm.compile_with_num_regs(&mut cb, 0);

    #[cfg(target_arch = "x86_64")]
    assert_eq!(cb.hexdump(), "cc");

    #[cfg(target_arch = "aarch64")]
    assert_eq!(cb.hexdump(), "00003ed4");
}

#[test]
fn test_call_itself() {
    assert_snapshot!(inspect("
        def test = 42.itself
        test
        test
    "), @"42");
}

#[test]
fn test_nil() {
    assert_snapshot!(inspect("
        def test = nil
        test
        test
    "), @"nil");
}

#[test]
fn test_function_stub_profiles_before_compiling() {
    rb_zjit_prepare_options();
    set_inline_threshold(0);
    let num_profiles = get_option!(num_profiles);
    let call_threshold = u32::from(num_profiles) + 2;
    set_call_threshold(call_threshold);

    eval(&format!("
        class Integer
          def zjit_profile_stub_target = self + 1
        end

        def zjit_profile_stub_entry(run)
          1.zjit_profile_stub_target if run
        end

        i = 0
        while i < {call_threshold}
          zjit_profile_stub_entry(false)
          i += 1
        end
    "));

    let entry_iseq = get_method_iseq("self", "zjit_profile_stub_entry");
    let entry_payload = get_or_create_iseq_payload(entry_iseq);
    let entry_version = unsafe { entry_payload.versions.last().unwrap().as_ref() };
    assert_eq!(1, entry_version.outgoing.len(), "expected a JIT-to-JIT function stub");

    let target_iseq = get_method_iseq("1", "zjit_profile_stub_target");
    assert!(get_or_create_iseq_payload(target_iseq).versions.is_empty());

    // Every stub hit in the profiling window should interpret the callee
    // without compiling it.
    for _ in 0..num_profiles {
        assert_eq!(VALUE::fixnum_from_usize(2), eval("zjit_profile_stub_entry(true)"));
        assert!(get_or_create_iseq_payload(target_iseq).versions.is_empty());
    }

    // Verify that the interpreted executions populated the profile for `+`.
    let mut insn_idx = 0;
    let iseq_size = unsafe { get_iseq_encoded_size(target_iseq) };
    let plus_idx = loop {
        assert!(insn_idx < iseq_size, "target ISEQ does not contain opt_plus");
        let opcode = iseq_opcode_at_idx(target_iseq, insn_idx);
        let bare_opcode = unsafe { rb_zjit_insn_to_bare_insn(opcode as i32) } as u32;
        if bare_opcode == YARVINSN_opt_plus {
            break insn_idx as usize;
        }
        insn_idx += insn_len(bare_opcode as usize);
    };
    assert_eq!(
        2,
        get_or_create_iseq_payload(target_iseq)
            .profile
            .get_operand_types(plus_idx)
            .unwrap()
            .len(),
    );

    // The following hit observes a completed profiling window and compiles.
    assert_eq!(VALUE::fixnum_from_usize(2), eval("zjit_profile_stub_entry(true)"));
    assert_eq!(1, get_or_create_iseq_payload(target_iseq).versions.len());
}

#[test]
fn test_putobject() {
    assert_snapshot!(inspect("
        def test = 1
        test
        test
    "), @"1");
}

#[test]
fn test_recompile_exit_waits_for_interpreter_profiles() {
    set_call_threshold(2);
    eval("
        def recompile_profile_window(a, b) = a + b
        recompile_profile_window(1, 2)
        recompile_profile_window(1, 2)
    ");

    let iseq = get_method_iseq("self", "recompile_profile_window");
    let num_profiles = get_option!(num_profiles);
    for _ in 0..num_profiles {
        eval("recompile_profile_window(1.5, 2.5)");
    }
    let payload = get_or_create_iseq_payload(iseq);
    assert!(!unsafe { payload.versions.last().unwrap().as_ref() }.is_invalidated());

    eval("recompile_profile_window(1.5, 2.5)");
    let payload = get_or_create_iseq_payload(iseq);
    assert!(unsafe { payload.versions.last().unwrap().as_ref() }.is_invalidated());
}

#[test]
fn test_dupstring() {
    eval(r##"
        def test = "#{""}"
        test
    "##);
    assert_contains_opcode("test", YARVINSN_dupstring);
    assert_snapshot!(assert_compiles(r##"test"##), @r#""""#);
}

#[test]
fn test_dupchilledstring() {
    eval(r#"
        def test = ""
        test
    "#);
    assert_contains_opcode("test", YARVINSN_dupchilledstring);
    assert_snapshot!(assert_compiles(r#"test"#), @r#""""#);
}

#[test]
fn test_leave_param() {
    assert_snapshot!(inspect("
        def test(n) = n
        test(5)
        test(5)
    "), @"5");
}

#[test]
fn test_getglobal_with_warning() {
    eval(r#"
        Warning[:deprecated] = true

        module Warning
          def warn(message)
            raise
          end
        end

        def test
          $=
        rescue
          "rescued"
        end
        $VERBOSE = true
        test
    "#);
    assert_contains_opcode("test", YARVINSN_getglobal);
    assert_snapshot!(assert_compiles(r#"test"#), @r#""rescued""#);
}

#[test]
fn test_setglobal() {
    eval("
        def test
          $a = 1
          $a
        end
        test
    ");
    assert_contains_opcode("test", YARVINSN_setglobal);
    assert_snapshot!(assert_compiles("test"), @"1");
}

#[test]
fn test_string_intern() {
    eval(r#"
        def test
          :"foo#{123}"
        end
        test
    "#);
    assert_contains_opcode("test", YARVINSN_intern);
    assert_snapshot!(assert_compiles(r#"test"#), @":foo123");
}

#[test]
fn test_duphash() {
    eval("
        def test
          {a: 1}
        end
        test
    ");
    assert_contains_opcode("test", YARVINSN_duphash);
    assert_snapshot!(assert_compiles("test"), @"{a: 1}");
}

#[test]
fn test_pushtoarray() {
    eval("
        def test
          [*[], 1, 2, 3]
        end
        test
    ");
    assert_contains_opcode("test", YARVINSN_pushtoarray);
    assert_snapshot!(assert_compiles("test"), @"[1, 2, 3]");
}

#[test]
fn test_splatarray_new_array() {
    eval("
        def test a
          [*a, 3]
        end
        test [1, 2]
    ");
    assert_contains_opcode("test", YARVINSN_splatarray);
    assert_snapshot!(assert_compiles("test [1, 2]"), @"[1, 2, 3]");
}

#[test]
fn test_splatarray_existing_array() {
    eval("
        def foo v
          [1, 2, v]
        end
        def test a
          foo(*a)
        end
        test [3]
    ");
    assert_contains_opcode("test", YARVINSN_splatarray);
    assert_snapshot!(assert_compiles("test [3]"), @"[1, 2, 3]");
}

#[test]
fn test_concattoarray() {
    eval("
        def test(*a)
          [1, 2, *a]
        end
        test 3
    ");
    assert_contains_opcode("test", YARVINSN_concattoarray);
    assert_snapshot!(assert_compiles("test 3"), @"[1, 2, 3]");
}

#[test]
fn test_definedivar() {
    eval("
        def test
          v0 = defined?(@a)
          @a = nil
          v1 = defined?(@a)
          remove_instance_variable :@a
          v2 = defined?(@a)
          [v0, v1, v2]
        end
        test
    ");
    assert_contains_opcode("test", YARVINSN_definedivar);
    assert_snapshot!(assert_compiles("test"), @r#"[nil, "instance-variable", nil]"#);
}

#[test]
fn test_setglobal_with_trace_var_exception() {
    eval(r#"
        def test
          $a = 1
        rescue
          "rescued"
        end
        trace_var(:$a) { raise }
        test
    "#);
    assert_contains_opcode("test", YARVINSN_setglobal);
    assert_snapshot!(assert_compiles(r#"test"#), @r#""rescued""#);
}

#[test]
fn test_getlocal_after_eval() {
    assert_snapshot!(inspect("
        def test
          a = 1
          eval('a = 2')
          a
        end
        test
        test
    "), @"2");
}

#[test]
fn test_getlocal_after_instance_eval() {
    assert_snapshot!(inspect("
        def test
          a = 1
          instance_eval('a = 2')
          a
        end
        test
        test
    "), @"2");
}

#[test]
fn test_getlocal_after_module_eval() {
    assert_snapshot!(inspect("
        def test
          a = 1
          Kernel.module_eval('a = 2')
          a
        end
        test
        test
    "), @"2");
}

#[test]
fn test_getlocal_after_class_eval() {
    assert_snapshot!(inspect("
        def test
          a = 1
          Kernel.class_eval('a = 2')
          a
        end
        test
        test
    "), @"2");
}

#[test]
fn test_setlocal() {
    assert_snapshot!(inspect("
        def test(n)
          m = n
          m
        end
        test(3)
        test(3)
    "), @"3");
}

#[test]
fn test_return_nonparam_local() {
    assert_snapshot!(inspect("
        def foo(a)
          if false
            x = nil
          end
          x
        end
        def test = foo(1)
        test
        test
    "), @"nil");
}

#[test]
fn test_nonparam_local_nil_in_jit_call() {
    assert_snapshot!(inspect(r#"
        def f(a)
          a ||= 1
          if false; b = 1; end
          eval("-> { p 'x#{b}' }")
        end

        4.times.map { f(1).call }
    "#), @r#"["x", "x", "x", "x"]"#);
}

#[test]
fn test_kwargs_with_exit_and_local_invalidation() {
    assert_snapshot!(inspect(r#"
        def a(b:, c:)
          if c == :b
            return -> {}
          end
          Class # invalidate locals

          raise "c is :b!" if c == :b
        end

        def test
          # note opposite order of kwargs
          a(c: :c, b: :b)
        end

        4.times { test }
        :ok
    "#), @":ok");
}

#[test]
fn test_kwargs_with_max_direct_send_arg_count() {
    assert_snapshot!(inspect("
        def kwargs(five, six, a:, b:, c:, d:, e:, f:)
          [a, b, c, d, five, six, e, f]
        end

        5.times.flat_map do
          [
            kwargs(5, 6, d: 4, c: 3, a: 1, b: 2, e: 7, f: 8),
            kwargs(5, 6, d: 4, c: 3, b: 2, a: 1, e: 7, f: 8)
          ]
        end.uniq
    "), @"[[1, 2, 3, 4, 5, 6, 7, 8]]");
}

#[test]
fn test_forwardable_callee_positional_args() {
    assert_snapshot!(inspect("
        def target(a, b) = a + b
        def fwd(...) = target(...)
        5.times.map { fwd(1, 2) }.uniq
    "), @"[3]");
}

#[test]
fn test_forwardable_callee_no_args() {
    assert_snapshot!(inspect("
        def target = :ok
        def fwd(...) = target(...)
        5.times.map { fwd }.uniq
    "), @"[:ok]");
}

#[test]
fn test_forwardable_callee_kwargs() {
    assert_snapshot!(inspect("
        def target(a, b:, c: 3) = [a, b, c]
        def fwd(...) = target(...)
        5.times.flat_map { [fwd(1, b: 2), fwd(1, c: 9, b: 2)] }.uniq
    "), @"[[1, 2, 3], [1, 2, 9]]");
}

#[test]
fn test_forwardable_callee_wrong_number_of_arguments() {
    assert_snapshot!(inspect(r#"
        def target(a, b) = a + b
        def fwd(...) = target(...)
        5.times.map { (fwd(1) rescue $!.message) }.uniq
    "#), @r#"["wrong number of arguments (given 1, expected 2)"]"#);
}

#[test]
fn test_forwardable_callee_unknown_keyword() {
    assert_snapshot!(inspect(r#"
        def target(a, b:) = [a, b]
        def fwd(...) = target(...)
        5.times.map { (fwd(1, z: 2) rescue $!.message) }.uniq
    "#), @r#"["missing keyword: :b"]"#);
}

#[test]
fn test_forwardable_callee_literal_block() {
    assert_snapshot!(inspect("
        def target(x) = yield(x)
        def fwd(...) = target(...)
        5.times.map { fwd(4) { |v| v * 2 } }.uniq
    "), @"[8]");
}

#[test]
fn test_forwardable_callee_block_arg_call_site_stays_dynamic() {
    // `vm_caller_setup_arg_block` pops a `&blk` argument off the stack before the forwardable
    // callee's frame is grown by `vm_ci_argc(ci)`, so the callinfo we would store in the `...`
    // local counts an argument that is no longer there. `can_direct_send_forwardable` lists
    // `VM_CALL_ARGS_BLOCKARG` in `FORWARDABLE_CALLEE_BLOCKERS` and declines the call site --
    // including when `&blk` holds a plain Proc, which would otherwise be handed straight to the
    // callee as its block handler by the block-arg passthrough.
    assert_snapshot!(inspect("
        def target(x) = [x, block_given? ? yield(x) : nil]
        def fwd(...) = target(...)
        def entry(x, p) = fwd(x, &p)
        doubler = ->(v) { v * 2 }
        out = nil
        200.times { |i| out = entry(i, doubler) }
        [out, entry(5, nil)]
    "), @"[[199, 398], [5, nil]]");
}

#[test]
fn test_forwardable_callee_splat_call_site_stays_dynamic() {
    assert_snapshot!(inspect("
        def target(*a, **k) = [a, k]
        def fwd(...) = target(...)
        args = [1, 2]
        opts = { x: 1 }
        5.times.flat_map { [fwd(*args), fwd(**opts), fwd(&nil)] }.uniq
    "), @"[[[1, 2], {}], [[], {x: 1}], [[], {}]]");
}

#[test]
fn test_forwardable_callee_ruby2_keywords_flag_survives() {
    assert_snapshot!(inspect("
        def target(*a, **k) = [a, k]
        def fwd(...) = target(...)
        ruby2_keywords def r2k(*a) = fwd(*a)
        5.times.map { r2k(1, k: 2) }.uniq
    "), @"[[[1], {k: 2}]]");
}

#[test]
fn test_forwardable_callee_chained_forwarding() {
    assert_snapshot!(inspect("
        def target(a, b:) = [a, b]
        def inner(...) = target(...)
        def outer(...) = inner(...)
        5.times.map { outer(1, b: 2) }.uniq
    "), @"[[1, 2]]");
}

#[test]
fn test_forwardable_callee_with_extra_locals() {
    assert_snapshot!(inspect("
        def target(a) = a * 2
        def fwd(...)
          extra = 10
          extra + target(...)
        end
        5.times.map { fwd(3) }.uniq
    "), @"[16]");
}

#[test]
fn test_forwardable_callee_super() {
    assert_snapshot!(inspect(r#"
        class Base
          def run(*a, **k) = ["base", a, k]
        end
        class Child < Base
          def run(...) = super
        end
        c = Child.new
        5.times.map { c.run(1, k: 2) }.uniq
    "#), @r#"[["base", [1], {k: 2}]]"#);
}

/// Run `program` and assert that `counter` moved by the end of it. Forwarding tests need this:
/// the dynamic `sendforward` computes the same answer as the expansion, so without an assertion
/// on what the compiler actually did they would all pass with the specialization switched off.
#[track_caller]
fn assert_forward_counter(program: &str, name: &str, counter: impl Fn(&crate::stats::Counters) -> u64) -> String {
    with_rubyvm(|| {
        let counters = crate::state::ZJITState::get_counters();
        let before = counter(counters);
        let result = assert_compiles_allowing_exits(program);
        let counters = crate::state::ZJITState::get_counters();
        assert!(counter(counters) > before, "expected {name} to increase, but it did not");
        result
    })
}

/// Assert that a `bar(...)` site in a standalone-compiled forwardable ISEQ expanded against a
/// profiled, guarded callinfo.
#[track_caller]
fn assert_expands_standalone_forward(program: &str) -> String {
    assert_forward_counter(program, "send_forward_expanded_profiled_count",
        |c| c.send_forward_expanded_profiled_count)
}

/// The inverse: the site had to stay on the dynamic `sendforward`, for the reason `counter`
/// names. Checking the reason and not just the absence of an expansion is what keeps these tests
/// from passing vacuously if the site stops being compiled at all.
#[track_caller]
fn assert_keeps_standalone_forward(program: &str, name: &str, counter: impl Fn(&crate::stats::Counters) -> u64) -> String {
    let expanded_before = with_rubyvm(||
        crate::state::ZJITState::get_counters().send_forward_expanded_profiled_count);
    let result = assert_forward_counter(program, name, counter);
    let expanded_after = with_rubyvm(||
        crate::state::ZJITState::get_counters().send_forward_expanded_profiled_count);
    assert_eq!(expanded_after, expanded_before,
        "expected the `bar(...)` site to stay dynamic, but it expanded");
    result
}

// A forwardable ISEQ that is *entered* megamorphically is never inlined, so its `bar(...)` has no
// compile-time callinfo. The profiler records the packed one the `...` local held every time and
// the compiled site guards it, reading the forwarded arguments back out of the frame extension.
#[test]
fn test_standalone_forwarder_megamorphic_entry() {
    assert_snapshot!(assert_expands_standalone_forward(r#"
        class Target; def bar(a, b) = a + b; end
        module Delegate
          def fwd(...) = @t.bar(...)
        end
        target = Target.new
        objs = 40.times.map do
          klass = Class.new do
            include Delegate
            define_method(:initialize) { |t| @t = t }
          end
          klass.new(target)
        end
        total = 0
        50.times { objs.each { |o| total += o.fwd(1, 2) } }
        total
    "#), @"6000");
}

// The same site reached with a different number of forwarded arguments fails the callinfo guard.
// The exit lands on the `sendforward` itself, where `vm_adjust_stack_forwarding` rebuilds the
// argument list from the frame extension, so the interpreter finishes the call unaided.
#[test]
fn test_standalone_forwarder_guard_miss_on_a_different_call_shape() {
    assert_snapshot!(assert_expands_standalone_forward(r#"
        class Target; def bar(*a) = a.sum; end
        class Fwd
          def initialize(t) = @t = t
          def fwd(...) = @t.bar(...)
        end
        f = Fwd.new(Target.new)
        200.times { f.fwd(1, 2) }
        [f.fwd(1, 2), f.fwd(1, 2, 3), f.fwd, f.fwd(4, 5)]
    "#), @"[3, 6, 0, 9]");
    assert!(crate::state::ZJITState::get_counters().exit_send_forward_callinfo_changed > 0,
        "expected the callinfo guard to have missed at least once");
}

// Same, but the callinfo changes because the *method name* did: two forwarders reaching the same
// packed callinfo would compare equal, so the guard has to see the whole word.
#[test]
fn test_standalone_forwarder_guard_miss_on_a_different_caller_name() {
    assert_snapshot!(assert_expands_standalone_forward(r#"
        class Target; def bar(a) = a * 2; end
        class Fwd
          def initialize(t) = @t = t
          def one(...) = @t.bar(...)
          def two(...) = one(...)
        end
        f = Fwd.new(Target.new)
        200.times { f.one(3) }
        [f.one(3), f.two(4)]
    "#), @"[6, 8]");
}

// A keyword-carrying caller needs a keyword table, which no packed callinfo has, so the `...`
// local holds a heap `imemo_callinfo`. Holding one across a compilation would need a GC root, so
// the site stays on the dynamic path.
#[test]
fn test_standalone_forwarder_kwargs_caller_stays_dynamic() {
    assert_snapshot!(assert_keeps_standalone_forward(r#"
        class Target; def bar(a, b:) = [a, b]; end
        class Fwd
          def initialize(t) = @t = t
          def fwd(...) = @t.bar(...)
        end
        f = Fwd.new(Target.new)
        200.times { f.fwd(1, b: 2) }
        f.fwd(1, b: 2)
    "#, "send_forward_reject_ci_not_packed", |c| c.send_forward_reject_ci_not_packed), @"[1, 2]");
}

// Chained forwarding: the target is itself a `def bar(...)`, whose `...` local has to *receive* a
// callinfo. No `rb_callinfo` describes the merged argument list, so the site is rejected outright
// rather than expanded into a call that could not fill the callee's `...`.
#[test]
fn test_standalone_forwarder_chained_forwarding_stays_dynamic() {
    assert_snapshot!(assert_forward_counter(r#"
        class Target; def bar(a, b) = a - b; end
        class Fwd
          def initialize(t) = @t = t
          def inner(...) = @t.bar(...)
          def outer(...) = inner(...)
        end
        f = Fwd.new(Target.new)
        200.times { f.outer(9, 4) }
        f.outer(9, 4)
    "#, "send_forward_reject_chained", |c| c.send_forward_reject_chained), @"5");
}

// `bh = VM_ENV_BLOCK_HANDLER(GET_LEP())`: a block given to the forwardable frame goes on to the
// target. A standalone site cannot know statically whether there is one, so it either passes the
// frame's handler through or -- when the target would reject or warn about an unused block --
// guards that there is none.
#[test]
fn test_standalone_forwarder_carries_a_block() {
    assert_snapshot!(assert_expands_standalone_forward(r#"
        class Target; def bar(x) = yield(x); end
        class Fwd
          def initialize(t) = @t = t
          def fwd(...) = @t.bar(...)
        end
        f = Fwd.new(Target.new)
        200.times { f.fwd(4) { |v| v * 2 } }
        f.fwd(4) { |v| v + 1 }
    "#), @"5");
}

// The blockless case takes the other branch: the target neither yields nor takes a block
// parameter, so passing the frame's handler on would keep it off the direct send. The site guards
// that the frame has no block instead, and a call that does have one exits to the interpreter.
//
// The block has to arrive through `public_send` for the guard to be the thing that catches it. A
// literal block written at the call site clears `VM_CALL_ARGS_SIMPLE` in that site's callinfo, so
// the *callinfo* guard already tells the two calls apart; `public_send` builds one callinfo for
// both and carries the block beside it.
#[test]
fn test_standalone_forwarder_block_guard_miss() {
    assert_snapshot!(assert_expands_standalone_forward(r#"
        class Target; def bar(a) = a * 3; end
        class Fwd
          def initialize(t) = @t = t
          def fwd(...) = @t.bar(...)
        end
        f = Fwd.new(Target.new)
        200.times { f.public_send(:fwd, 2) }
        [f.public_send(:fwd, 2), f.public_send(:fwd, 2) { :ignored }]
    "#), @"[6, 6]");
    assert!(crate::state::ZJITState::get_counters().exit_send_forward_block_given > 0,
        "expected the block-handler guard to have missed at least once");
}

// A literal block at the call site changes the caller's callinfo, so a forwarder warmed up
// without one and then called with one misses the callinfo guard rather than reaching the target
// with a block the expansion did not plan for.
#[test]
fn test_standalone_forwarder_literal_block_appearing_late() {
    assert_snapshot!(assert_expands_standalone_forward(r#"
        class Target; def bar(a) = block_given? ? yield(a) : a * 3; end
        class Fwd
          def initialize(t) = @t = t
          def fwd(...) = @t.bar(...)
        end
        f = Fwd.new(Target.new)
        200.times { f.fwd(2) }
        [f.fwd(2), f.fwd(2) { |v| v + 1 }]
    "#), @"[6, 3]");
}

// A guard *inside* the expansion -- here the receiver class guard on `@t` -- exits to the
// `sendforward` with the site's original stack, `[recv, ...]`, still on it. That is what
// `vm_adjust_stack_forwarding` expects: it clobbers the `...` slot with the forwarded arguments
// it copies back out of the frame extension.
#[test]
fn test_standalone_forwarder_side_exit_resumes_the_sendforward() {
    assert_snapshot!(assert_expands_standalone_forward(r#"
        class A; def m(a, b, c) = [:a, a, b, c]; end
        class B; def m(a, b, c) = [:b, a, b, c]; end
        class Fwd
          def initialize(t) = @t = t
          def m(...) = @t.m(...)
        end
        fa = Fwd.new(A.new)
        fb = Fwd.new(B.new)
        200.times { fa.m(1, 2, 3) }
        [fa.m(1, 2, 3), fb.m(4, 5, 6)]
    "#), @"[[:a, 1, 2, 3], [:b, 4, 5, 6]]");
}

// A `ruby2_keywords` frame splats a flagged Hash into the forwarder, which makes the caller's
// callinfo carry `VM_CALL_ARGS_SPLAT`. That is unspecializable, and the flag still has to reach
// the target as keywords.
#[test]
fn test_standalone_forwarder_ruby2_keywords() {
    assert_snapshot!(assert_keeps_standalone_forward(r#"
        class Target; def bar(*a, **k) = [a, k]; end
        class Fwd
          def initialize(t) = @t = t
          def fwd(...) = @t.bar(...)
          ruby2_keywords def r2k(*a) = fwd(*a)
        end
        f = Fwd.new(Target.new)
        200.times { f.r2k(1, k: 2) }
        f.r2k(1, k: 2)
    "#, "send_forward_reject_complex_args", |c| c.send_forward_reject_complex_args), @"[[1], {k: 2}]");
}

// The site's own arguments come first in the merged list, and the forwarded ones are read out of
// the frame extension in call order after them. (`def fwd(x, ...)` is not a forwardable ISEQ at
// all -- Ruby compiles it to `*rest, **kwrest, &block` -- so the site's own arguments have to be
// written at the call rather than declared as parameters.)
#[test]
fn test_standalone_forwarder_site_writes_its_own_args() {
    assert_snapshot!(assert_expands_standalone_forward(r#"
        class Target; def bar(a, b, c) = [a, b, c]; end
        class Fwd
          def initialize(t) = @t = t
          def fwd(...) = @t.bar(:first, ...)
        end
        f = Fwd.new(Target.new)
        200.times { f.fwd(2, 3) }
        f.fwd(2, 3)
    "#), @"[:first, 2, 3]");
}

// `vm_adjust_stack_forwarding` measures the frame extension from the *local* ISEQ's table, so a
// forwarder with body locals of its own has to read the arguments from further down.
#[test]
fn test_standalone_forwarder_with_extra_locals() {
    assert_snapshot!(assert_expands_standalone_forward(r#"
        class Target; def bar(a, b) = a + b; end
        class Fwd
          def initialize(t) = @t = t
          def fwd(...)
            extra = 10
            spare = 5
            extra + spare + @t.bar(...)
          end
        end
        f = Fwd.new(Target.new)
        200.times { f.fwd(1, 2) }
        f.fwd(1, 2)
    "#), @"18");
}

// The forwarded arguments stay live in the frame extension across a GC, which is what makes
// reading them back out of it after arbitrary work in the same frame safe.
#[test]
fn test_standalone_forwarder_arguments_survive_a_gc() {
    assert_snapshot!(assert_expands_standalone_forward(r#"
        class Target; def bar(a, b) = a + b; end
        class Fwd
          def initialize(t) = @t = t
          def fwd(...)
            GC.start
            @t.bar(...)
          end
        end
        f = Fwd.new(Target.new)
        200.times { f.fwd("x", "y") }
        f.fwd("a", "b")
    "#), @r#""ab""#);
}

#[test]
fn test_explicit_super_to_forwardable_callee() {
    assert_snapshot!(inspect(r#"
        class Base
          def run(...) = fin(...)
          def fin(a, b) = ["base", a, b]
        end
        class Child < Base
          def run(a, b) = super(a, b)
        end
        c = Child.new
        5.times.map { c.run(1, 2) }.uniq
    "#), @r#"[["base", 1, 2]]"#);
}

#[test]
fn test_zsuper_to_forwardable_callee() {
    assert_snapshot!(inspect(r#"
        class Base
          def run(...) = fin(...)
          def fin(a, b) = ["base", a, b]
        end
        class Child < Base
          def run(a, b) = super
        end
        c = Child.new
        5.times.map { c.run(3, 4) }.uniq
    "#), @r#"[["base", 3, 4]]"#);
}

#[test]
fn test_inlined_forwarder_positional_args() {
    // The forwarder is inlined into its caller, so the `bar(...)` inside it sees the caller's
    // callinfo at compile time and becomes a direct call to `target`.
    assert_snapshot!(with_inlining(|| assert_inlines("
        def target(a, b) = a - b
        def fwd(...) = target(...)
        def entry = fwd(7, 2)
        200.times { entry }
        entry
    ")), @"5");
}

#[test]
fn test_inlined_forwarder_keyword_args() {
    // `vm_caller_setup_fwd_args` gives the merged callinfo the *caller's* keyword table, so the
    // expanded call has to bind the trailing arguments as keywords, not as positionals.
    assert_snapshot!(with_inlining(|| assert_inlines(r#"
        def target(a, b:, c: 3) = [a, b, c]
        def fwd(...) = target(...)
        def entry = [fwd(1, b: 2), fwd(1, c: 9, b: 2)]
        200.times { entry }
        entry
    "#)), @"[[1, 2, 3], [1, 2, 9]]");
}

#[test]
fn test_inlined_forwarder_site_writes_its_own_args() {
    // `bar(x, ...)`: the merged argument list is the site's own arguments followed by the
    // caller's, in that order.
    assert_snapshot!(with_inlining(|| assert_inlines("
        def target(a, b, c) = [a, b, c]
        def fwd(x, ...) = target(x, ...)
        def entry = fwd(1, 2, 3)
        200.times { entry }
        entry
    ")), @"[1, 2, 3]");
}

#[test]
fn test_inlined_forwarder_carrying_a_literal_block() {
    // `bh = VM_ENV_BLOCK_HANDLER(GET_LEP())`: the forwarded call gets the forwarder frame's own
    // block handler, which the expanded call reads back out of the frame's EP. Re-deriving the
    // literal block instead would capture the wrong frame, since the block belongs to `entry`.
    assert_snapshot!(with_inlining(|| assert_inlines("
        def target(x) = yield(x)
        def fwd(...) = target(...)
        def entry = fwd(4) { |v| v * 2 }
        200.times { entry }
        entry
    ")), @"8");
}

#[test]
fn test_inlined_forwarder_block_present_on_some_calls_only() {
    // A `&blk` handed to the forwarder may be a Proc on one call and nothing on the next, so the
    // handler the expanded call passes on is only known at run time. `block_given?` in the target
    // has to see each call for what it was.
    assert_snapshot!(with_inlining(|| assert_inlines_allowing_exits(r#"
        def target(x) = [x, block_given? ? yield(x) : :none]
        def fwd(...) = target(...)
        def entry(i, &b) = fwd(i, &b)
        200.times { |i| i.even? ? entry(i) { |v| v } : entry(i) }
        [entry(1) { |v| v * 2 }, entry(1)]
    "#)), @"[[1, 2], [1, :none]]");
}

#[test]
fn test_inlined_forwarder_argument_error() {
    // The argument check belongs to the target, which the inlined forwarder now calls directly.
    assert_snapshot!(with_inlining(|| assert_inlines_allowing_exits(r#"
        def target(a, b) = a + b
        def fwd(...) = target(...)
        def entry = (fwd(1) rescue $!.message)
        200.times { entry }
        entry
    "#)), @r#""wrong number of arguments (given 1, expected 2)""#);
}

#[test]
fn test_inlined_forwarder_side_exit_resumes_the_sendforward() {
    // A guard inside the inlined forwarder exits to the `sendforward` instruction, and the
    // interpreter's `vm_adjust_stack_forwarding` rebuilds the argument list by reading below the
    // frame at `lep - (local_table_size + argc + 2)`. That only works because the inlined frame
    // push copied the arguments into those slots and put the callinfo above them, the way
    // `vm_call_iseq_forwardable` does.
    assert_snapshot!(with_inlining(|| assert_inlines_allowing_exits(r#"
        class A; def m(a, b, c) = [:a, a, b, c]; end
        class B; def m(a, b, c) = [:b, a, b, c]; end
        class Fwd
          def initialize(t) = @t = t
          def m(...) = @t.m(...)
        end
        fa = Fwd.new(A.new)
        fb = Fwd.new(B.new)
        # Warm up on A alone so the expanded call guards on A.
        200.times { fa.m(1, 2, 3) }
        # B fails that guard mid-forwarder.
        [fa.m(1, 2, 3), fb.m(4, 5, 6)]
    "#)), @"[[:a, 1, 2, 3], [:b, 4, 5, 6]]");
}

#[test]
fn test_inlined_forwarder_chained_forwarding_falls_back() {
    // The inner target is itself a `def bar(...)`, whose `...` local has to receive a real
    // callinfo. No `rb_callinfo` describes the merged call, so the site keeps its `sendforward`.
    assert_snapshot!(with_inlining(|| assert_inlines("
        def target(a, b:) = [a, b]
        def inner(...) = target(...)
        def outer(...) = inner(...)
        def entry = outer(1, b: 2)
        200.times { entry }
        entry
    ")), @"[1, 2]");
}

#[test]
fn test_inlined_forwarder_ruby2_keywords() {
    // A `ruby2_keywords` frame splats into the forwarder, which keeps the call site off the
    // direct send entirely; the flagged Hash still has to reach the target as keywords.
    assert_snapshot!(with_inlining(|| assert_inlines_allowing_exits("
        def target(*a, **k) = [a, k]
        def fwd(...) = target(...)
        ruby2_keywords def r2k(*a) = fwd(*a)
        def entry = r2k(1, k: 2)
        200.times { entry }
        entry
    ")), @"[[1], {k: 2}]");
}

#[test]
fn test_inlined_forwarder_super_is_unaffected() {
    // `super` out of a forwardable frame goes through `invokesuperforward`, which
    // `vm_search_super_method` rebuilds the callinfo for at run time. Inlining the frame must
    // not disturb it.
    assert_snapshot!(with_inlining(|| assert_inlines_allowing_exits(r#"
        class Base
          def run(*a, **k) = ["base", a, k]
        end
        class Child < Base
          def run(...) = super
        end
        c = Child.new
        def call_it(c) = c.run(1, k: 2)
        200.times { call_it(c) }
        call_it(c)
    "#)), @r#"["base", [1], {k: 2}]"#);
}

#[test]
fn test_inlined_forwarder_with_extra_locals() {
    // The `...` local is local 0 and the frame extension sits below the whole local table, so a
    // forwarder with locals of its own still finds its arguments where the interpreter left them.
    assert_snapshot!(with_inlining(|| assert_inlines("
        def target(a) = a * 2
        def fwd(...)
          extra = 10
          extra + target(...)
        end
        def entry = fwd(3)
        200.times { entry }
        entry
    ")), @"16");
}

#[test]
fn test_kwrest_only_no_caller_keywords() {
    assert_snapshot!(inspect("
        def target(a, **opts) = [a, opts]
        5.times.map { target(1) }.uniq
    "), @"[[1, {}]]");
}

#[test]
fn test_kwrest_only_with_caller_keywords() {
    assert_snapshot!(inspect("
        def target(a, **opts) = [a, opts]
        5.times.map { target(1, x: 2, y: 3) }.uniq
    "), @"[[1, {x: 2, y: 3}]]");
}

#[test]
fn test_kwrest_with_named_keywords() {
    assert_snapshot!(inspect("
        def target(a, b: 1, **opts) = [a, b, opts]
        5.times.flat_map { [target(1), target(1, b: 2), target(1, z: 3, b: 2)] }.uniq
    "), @"[[1, 1, {}], [1, 2, {}], [1, 2, {z: 3}]]");
}

#[test]
fn test_kwrest_with_required_keyword_missing() {
    assert_snapshot!(inspect(r#"
        def target(a, b:, **opts) = [a, b, opts]
        5.times.map { (target(1, q: 5) rescue $!.message) }.uniq
    "#), @r#"["missing keyword: :b"]"#);
}

#[test]
fn test_kwrest_with_rest_and_optional() {
    assert_snapshot!(inspect("
        def target(a, b = 9, *r, c:, d: 4, **opts) = [a, b, r, c, d, opts]
        5.times.map { target(1, 2, 3, 4, c: 5, e: 6) }.uniq
    "), @"[[1, 2, [3, 4], 5, 4, {e: 6}]]");
}

#[test]
fn test_kwrest_only_kwrest_param() {
    assert_snapshot!(inspect("
        def target(**opts) = opts
        5.times.flat_map { [target, target(k: 1)] }.uniq
    "), @"[{}, {k: 1}]");
}

#[test]
fn test_kwrest_anonymous() {
    assert_snapshot!(inspect("
        def target(**) = :anon
        5.times.flat_map { [target, target(k: 1)] }.uniq
    "), @"[:anon]");
}

#[test]
fn test_kwrest_anonymous_forwarded() {
    assert_snapshot!(inspect("
        def sink(*x, **k) = [x, k]
        def anon_only(**) = sink(**)
        def anon_with_lead(a, **) = sink(a, **)
        def anon_with_rest(*, **) = sink(*, **)
        5.times.flat_map do
          [anon_only, anon_only(k: 1), anon_with_lead(1), anon_with_lead(1, z: 2), anon_with_rest(1, 2)]
        end.uniq
    "), @"[[[], {}], [[], {k: 1}], [[1], {}], [[1], {z: 2}], [[1, 2], {}]]");
}

#[test]
fn test_kwrest_anonymous_named_keyword_still_allocates_hash() {
    assert_snapshot!(inspect("
        def target(a, b: 5, **) = [a, b]
        5.times.flat_map { [target(1), target(1, b: 3, q: 4)] }.uniq
    "), @"[[1, 5], [1, 3]]");
}

#[test]
fn test_kwrest_splat_and_kwrest() {
    assert_snapshot!(inspect("
        def target(*a, **opts) = [a, opts]
        5.times.flat_map { [target, target(1, 2, k: 3)] }.uniq
    "), @"[[[], {}], [[1, 2], {k: 3}]]");
}

#[test]
fn test_setlocal_on_eval() {
    assert_snapshot!(inspect("
        @b = binding
        eval('a = 1', @b)
        eval('a', @b)
    "), @"1");
}

#[test]
fn test_optional_arguments() {
    assert_snapshot!(inspect("
        def test(a, b = 2, c = 3)
          [a, b, c]
        end
        [test(1), test(10, 20), test(100, 200, 300)]
    "), @"[[1, 2, 3], [10, 20, 3], [100, 200, 300]]");
}

#[test]
fn test_optional_arguments_setlocal() {
    assert_snapshot!(inspect("
        def test(a = (b = 2))
          [a, b]
        end
        [test, test(1)]
    "), @"[[2, 2], [1, nil]]");
}

#[test]
fn test_optional_arguments_cyclic() {
    assert_snapshot!(inspect("
        test = proc { |a=a| a }
        [test.call, test.call(1)]
    "), @"[nil, 1]");
}

#[test]
fn test_getblockparamproxy() {
    eval("
        def test(&block)
          0.then(&block)
        end
        test { 1 }
    ");
    assert_contains_opcode("test", YARVINSN_getblockparamproxy);
    // `Kernel#then` is Ruby-level and its `yield(self)` compiles to a direct dispatch to the
    // block it profiled, which is not the one this assertion passes. The guard misses once and
    // the site respecializes; the getblockparamproxy code under test is unaffected.
    assert_snapshot!(assert_compiles_allowing_exits("test { 1 }"), @"1");
}

#[test]
fn test_getblockparamproxy_modified() {
    eval("
        def test(&block)
          b = block
          0.then(&block)
        end
        test { 1 }
    ");
    assert_contains_opcode("test", YARVINSN_getblockparamproxy);
    assert_snapshot!(inspect("test { 1 }"), @"1");
}

#[test]
fn test_getblockparamproxy_modified_nested_block() {
    eval("
        def test(&block)
          proc do
            b = block
            0.then(&block)
          end
        end
        test { 1 }.call
    ");
    assert_snapshot!(inspect("test { 1 }.call"), @"1");
}

#[test]
fn test_getblockparamproxy_polymorphic_none_and_iseq() {
    set_call_threshold(3);
    eval("
        def test(&block)
          0.then(&block)
        end
        test
        test { 1 }
    ");
    assert_contains_opcode("test", YARVINSN_getblockparamproxy);
    // See test_getblockparamproxy: the `yield(self)` inside `Kernel#then` respecializes on the
    // block this assertion passes.
    assert_snapshot!(assert_compiles_allowing_exits("test { 2 }"), @"2");
}

#[test]
fn test_getblockparamproxy_proc() {
    eval("
        val = proc { 1 }
        def test(&block)
          0.then(&block)
        end
        test(&val)
    ");
    assert_contains_opcode("test", YARVINSN_getblockparamproxy);
    assert_snapshot!(assert_compiles("val = proc { 2 }; test(&val)"), @"2");
}

#[test]
fn test_getblockparamproxy_polymorphic_none_and_iseq_and_proc() {
    set_call_threshold(4);
    eval("
        val = proc { 3 }
        def test(&block)
          0.then(&block)
        end
        test
        test { 1 }
        test(&val)
    ");
    assert_contains_opcode("test", YARVINSN_getblockparamproxy);
    assert_snapshot!(assert_compiles("val = proc { 2 }; test(&val)"), @"2");
}

#[test]
fn test_yield_inline_self_is_captured_self() {
    // The inlined frame's self must be the block's captured self, not the yielding receiver.
    set_call_threshold(2);
    eval("
        class Yielder
          def run = yield
        end
        class C
          def initialize(v) = @v = v
          def go(y) = y.run { @v * 2 }
        end
        Y = Yielder.new
        C.new(21).go(Y)
        C.new(21).go(Y)
    ");
    assert_snapshot!(assert_compiles("C.new(21).go(Y)"), @"42");
}

#[test]
fn test_yield_iseq_guard_miss_recompiles() {
    set_call_threshold(2);
    eval("
        def invoke = yield(41)
        invoke { |x| x * 2 }
        invoke { |x| x * 2 }
    ");
    assert_snapshot!(assert_compiles_allowing_exits("[invoke { |x| x + 1 }, invoke { |x| x * 2 }]"), @"[42, 82]");
}

#[test]
fn test_yield_polymorphic_blocks_dispatch_directly() {
    // A yield site shared by two call sites recompiles with a polymorphic ISEQ dispatch
    // chain after the monomorphic guard miss. Once the polymorphic version is installed,
    // both blocks must dispatch directly with no side exits.
    set_call_threshold(2);
    eval("
        def invoke = yield(10)
        def add_one = invoke { |x| x + 1 }
        def double = invoke { |x| x * 2 }
        add_one; double
        add_one; double
    ");
    // Drive the re-profile window so the invalidated monomorphic version is replaced.
    let num_profiles = get_option!(num_profiles);
    for _ in 0..num_profiles + 2 {
        eval("add_one; double");
    }
    assert_snapshot!(assert_compiles("[add_one, double]"), @"[11, 20]");
}

#[test]
fn test_yield_polymorphic_non_iseq_handler_falls_back() {
    // A proc handler at a polymorphic yield site fails the ISEQ tag check and takes the
    // generic InvokeBlock fallback in-line, without a side exit or another recompile.
    set_call_threshold(2);
    eval("
        def invoke = yield(10)
        def add_one = invoke { |x| x + 1 }
        def double = invoke { |x| x * 2 }
        def via_proc(l) = invoke(&l)
        add_one; double
        add_one; double
    ");
    let num_profiles = get_option!(num_profiles);
    for _ in 0..num_profiles + 2 {
        eval("add_one; double; via_proc(proc { |x| x * 3 })");
    }
    assert_snapshot!(assert_compiles("[add_one, double, via_proc(proc { |x| x * 3 })]"), @"[11, 20, 30]");
}

#[test]
fn test_yield_polymorphic_symbol_handler_falls_back() {
    // A symbol handler at a polymorphic yield site fails the ISEQ tag check and takes the
    // generic InvokeBlock fallback in-line, without a side exit or another recompile.
    set_call_threshold(2);
    eval("
        def invoke = yield(10)
        def add_one = invoke { |x| x + 1 }
        def double = invoke { |x| x * 2 }
        def via_sym = invoke(&:to_s)
        add_one; double
        add_one; double
    ");
    let num_profiles = get_option!(num_profiles);
    for _ in 0..num_profiles + 2 {
        eval("add_one; double; via_sym");
    }
    assert_snapshot!(assert_compiles("[add_one, double, via_sym]"), @r#"[11, 20, "10"]"#);
}

#[test]
fn test_yield_polymorphic_ifunc_handler_falls_back() {
    // An ifunc handler (Enumerator#each yields to the enumerator's C block) at a polymorphic
    // yield site fails the ISEQ tag check and takes the generic InvokeBlock fallback in-line.
    // Threshold 4 keeps calls 1-3 in the profile window (num_profiles defaults to 5), so
    // invoke's first compile already sees both blocks and installs the polymorphic dispatch;
    // the standalone version matters here because the Enumerator calls invoke from C.
    set_call_threshold(4);
    eval("
        def invoke = yield(10)
        def add_one = invoke { |x| x + 1 }
        def double = invoke { |x| x * 2 }
        def via_enum = to_enum(:invoke).to_a
        add_one; double
        add_one; double
    ");
    assert_snapshot!(assert_compiles("[add_one, double, via_enum]"), @"[11, 20, [10]]");
}

#[test]
fn test_yield_repeated_ifunc_handlers_dispatch_directly() {
    // Every Enumerator call allocates a fresh ifunc, so profiling block handlers by object
    // identity used to make a yield site that only ever yields to C blocks look megamorphic.
    // Handlers are profiled by kind instead, so the site stays monomorphic and takes the
    // ifunc fast path while still returning the right result.
    set_call_threshold(2);
    eval("
        def invoke = yield(10)
        def via_enum = to_enum(:invoke).to_a
        via_enum; via_enum
    ");
    let num_profiles = get_option!(num_profiles);
    for _ in 0..num_profiles + 2 {
        eval("via_enum");
    }
    assert_snapshot!(assert_compiles("via_enum"), @"[10]");
}

#[test]
fn test_yield_cold_iseq_candidates_skip_dispatch_chain() {
    // A yield site whose executions are dominated by proc handlers must not build an ISEQ
    // dispatch chain out of the cold ISEQ blocks in its profile: the chain would miss on
    // nearly every call. Results must stay correct for every handler either way.
    set_call_threshold(2);
    eval("
        def invoke = yield(10)
        def add_one = invoke { |x| x + 1 }
        def via_proc(l) = invoke(&l)
        PROCS = [proc { |x| x * 3 }, proc { |x| x * 4 }]
        add_one; via_proc(PROCS[0])
        add_one; via_proc(PROCS[0])
    ");
    let num_profiles = get_option!(num_profiles);
    for _ in 0..num_profiles + 2 {
        eval("via_proc(PROCS[0]); via_proc(PROCS[1])");
    }
    assert_snapshot!(assert_compiles("[add_one, via_proc(PROCS[0]), via_proc(PROCS[1])]"), @"[11, 30, 40]");
}

#[test]
fn test_yield_megamorphic_mixed_block_handlers() {
    // A yield site that sees ISEQ, proc, symbol, and ifunc handlers mixed together goes
    // megamorphic (each to_enum call profiles a distinct ifunc), so it compiles to the
    // generic InvokeBlock and must return the right result for every handler kind.
    set_call_threshold(2);
    eval("
        def invoke = yield(10)
        def add_one = invoke { |x| x + 1 }
        def double = invoke { |x| x * 2 }
        def via_proc(l) = invoke(&l)
        def via_sym = invoke(&:to_s)
        def via_enum = to_enum(:invoke).to_a
        PR = proc { |x| x * 3 }
        add_one; double
        add_one; double
    ");
    let num_profiles = get_option!(num_profiles);
    for _ in 0..num_profiles + 2 {
        eval("add_one; double; via_proc(PR); via_sym; via_enum");
    }
    assert_snapshot!(assert_compiles("[add_one, double, via_proc(PR), via_sym, via_enum]"), @r#"[11, 20, 30, "10", [10]]"#);
}

#[test]
fn test_yield_inline_invocation_with_args() {
    // Plain yield with two args to a matching-arity block inlines and returns correctly.
    set_call_threshold(2);
    eval("
        def foo = yield(3, 4)
        def test = foo { |a, b| a + b }
        test
        test
    ");
    assert_snapshot!(assert_compiles("test"), @"7");
}

#[test]
fn test_yield_with_more_args_than_abi_registers() {
    // `self` + eight yield args don't fit in C argument registers (6 on x86_64, 8 on
    // arm64), so the direct block invocation passes the overflow arguments on the
    // native stack.
    set_call_threshold(2);
    eval("
        def foo = yield(1, 2, 3, 4, 5, 6, 7, 8)
        def test = foo { |a, b, c, d, e, f, g, h| a + b + c + d + e + f + g + h }
        test
        test
    ");
    assert_snapshot!(assert_compiles("test"), @"36");
}

#[test]
fn test_send_direct_with_more_args_than_abi_registers() {
    // `self` + ten args don't fit in C argument registers (6 on x86_64, 8 on arm64),
    // so the JIT-to-JIT call passes the overflow arguments on the native stack, and
    // the callee's JIT entry loads them from above its frame.
    set_call_threshold(2);
    eval("
        def callee(a, b, c, d, e, f, g, h, i, j) = [a, b, c, d, e, f, g, h, i, j]
        def test = callee(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        test
        test
    ");
    assert_snapshot!(assert_compiles("test"), @"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
}

#[test]
fn test_send_direct_with_equal_args_beyond_abi_registers() {
    // A pair of adjacent stack-passed arguments that are the same zero immediate
    // (false) or the same register lowers to an STP with an identical register
    // pair on arm64, e.g. `stp xzr, xzr`, which the assembler used to reject.
    // On arm64, c_args[8] and c_args[9] (arguments h and i below) form a pair.
    set_call_threshold(2);
    eval("
        def callee(a, b, c, d, e, f, g, h, i, j) = [a, b, c, d, e, f, g, h, i, j]
        def test(x = 9) = [callee(1, 2, 3, 4, 5, 6, 7, false, false, 10), callee(1, 2, 3, 4, 5, 6, 7, x, x, 10)]
        test
        test
    ");
    assert_snapshot!(assert_compiles("test"), @"[[1, 2, 3, 4, 5, 6, 7, false, false, 10], [1, 2, 3, 4, 5, 6, 7, 9, 9, 10]]");
}

#[test]
fn test_yield_inline_invocation_live_stack_below_args() {
    // A live value sits on the stack below the yield args; the no-receiver-slot SP math
    // must preserve it so `x +` sees the right operand.
    set_call_threshold(2);
    eval("
        def foo(x) = x + yield(1, 2)
        def test = foo(10) { |a, b| a + b }
        test
        test
    ");
    assert_snapshot!(assert_compiles("test"), @"13");
}

#[test]
fn test_yield_inlined_caller_block_dispatches_without_guards() {
    // When the yielding method is inlined into a caller that passes a literal block, the block
    // handler is written into the inlined frame's EP from a compile-time constant, so the yield
    // dispatches with no tag/iseq guards. assert_inlines requires the method to actually inline
    // and to run with no side exits, exercising the guard-free InvokeBlockIseqDirect machine code.
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def two_yields = (yield 1) + (yield 2)
            def test = two_yields { |x| x * 10 }
            test
            test
        "), @"30");
    });
}

#[test]
fn test_yield_with_lambda_arg() {
    // A lambda passed via &l is a proc handler (not imemo_iseq): yield falls back but runs.
    set_call_threshold(2);
    eval("
        def foo = yield(5)
        def test = foo(&L)
        L = ->(x) { x * 10 }
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"50");
}

#[test]
fn test_yield_break() {
    set_call_threshold(2);
    eval("
        def foo = yield
        def test = foo { break 5 }
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"5");
}

/// `eval` the setup code and then run `program`, asserting that an exception
/// handler entry (`body->jit_exception`) was compiled along the way. Returns the
/// `#inspect` of `program`.
#[track_caller]
fn assert_compiles_exception_entry(setup: &str, program: &str) -> String {
    // The setup runs the program often enough to reach the call threshold, so
    // the entry is compiled here and `program` below runs the compiled code.
    eval(setup);
    assert!(crate::state::ZJITState::get_counters().compiled_exception_entry_count > 0,
        "expected the program to compile an exception handler entry, but none was compiled");
    assert_compiles_allowing_exits(program)
}

#[test]
fn test_exception_entry_at_break_continuation() {
    set_call_threshold(2);
    // The frame that catches the `break` resumes at the catch-table
    // continuation, which is compiled as an exception handler entry.
    assert_snapshot!(assert_compiles_exception_entry("
        def find_it(arr, target)
          n = 0
          arr.each do |x|
            n += 1
            break x * 10 if x == target
          end
          n + 100
        end
        def test = find_it([1, 2, 3], 2)
        test
        test
    ", "test"), @"102");
}

#[test]
fn test_exception_entry_at_break_continuation_with_live_stack() {
    set_call_threshold(2);
    // The value 7 is live on the VM stack across the send, so the exception
    // entry has to read it back from the VM stack.
    assert_snapshot!(assert_compiles_exception_entry("
        def find_it(arr, target)
          [7, arr.each { |x| break x * 10 if x == target }, 9]
        end
        def test = find_it([1, 2, 3], 2)
        test
        test
    ", "test"), @"[7, 20, 9]");
}

#[test]
fn test_exception_entry_in_rescue_iseq() {
    set_call_threshold(2);
    // The rescue ISEQ is entered by the interpreter with the exception in its
    // first local, so the entry must read locals from the frame rather than
    // initialize them to nil like an ordinary call entry does.
    assert_snapshot!(assert_compiles_exception_entry("
        def risky(n)
          x = n
          begin
            raise 'boom'
          rescue => e
            x += e.message.length
          end
          x
        end
        def test = risky(1)
        test
        test
    ", "test"), @"5");
}

#[test]
fn test_exception_entry_at_retry_continuation() {
    set_call_threshold(2);
    assert_snapshot!(assert_compiles_exception_entry("
        def attempt(limit)
          tries = 0
          begin
            tries += 1
            raise 'again' if tries < limit
          rescue
            retry
          end
          tries
        end
        def test = attempt(3)
        test
        test
    ", "test"), @"3");
}

#[test]
fn test_exception_entry_with_non_local_return() {
    set_call_threshold(2);
    assert_snapshot!(assert_compiles_exception_entry("
        def inner(a)
          a.each { |x| yield x }
          :fell_through
        end
        def middle(a)
          inner(a) { |x| return x + 1000 if x == 2 }
          :no_return
        end
        def test = [middle([1, 2, 3]), middle([1])]
        test
        test
    ", "test"), @"[1002, :no_return]");
}

#[test]
fn test_yield_non_local_return() {
    set_call_threshold(2);
    eval("
        def inner = yield
        def test
          inner { return 42 }
          99
        end
        test
        test
    ");
    // The block's body is inlined into `test` and its `return` becomes a plain return, so
    // this compiles with no throw and no side exit at all.
    assert_snapshot!(assert_compiles("test"), @"42");
}

/// A lone yielded Array is destructured into a multi-parameter block, and the direct dispatch
/// takes it. Without the expansion the arity mismatch leaves this on the generic `invokeblock`
/// and the block's `return` throws, which `assert_compiles` catches as a side exit.
#[test]
fn test_block_autosplat_direct_dispatch() {
    set_call_threshold(2);
    eval("
        def test(pairs)
          pairs.each { |a, b| return a + b if a > 0 }
          -1
        end
        test([[1, 2]])
        test([[1, 2]])
    ");
    assert_snapshot!(assert_compiles("test([[3, 4]])"), @"7");
}

/// The auto-splat expansion joins the generic `invokeblock` rather than side-exiting, so a
/// site that sees an Array of the wrong length, a non-Array, or nil still gets the
/// interpreter's nil-filling and truncation, and keeps running compiled code.
#[test]
fn test_block_autosplat_length_mismatch_joins_fallback() {
    set_call_threshold(2);
    eval("
        def each_of(vals)
          out = []
          vals.each { |a, b| out << [a, b] }
          out
        end
        each_of([[1, 2]])
        each_of([[1, 2]])
    ");
    // exact length, too long, too short, empty, non-Array, nil
    assert_snapshot!(assert_compiles("each_of([[1, 2], [3, 4, 5], [6], [], 7, nil]).inspect"),
        @r#""[[1, 2], [3, 4], [6, nil], [nil, nil], [7, nil], [nil, nil]]""#);
}

/// An Array subclass and a `to_ary` duck both take the fallback arm, where the interpreter's
/// `rb_check_array_type` destructures them the same way it always has.
#[test]
fn test_block_autosplat_non_exact_array_joins_fallback() {
    set_call_threshold(2);
    eval("
        class AutosplatSub < Array; end
        class AutosplatDuck; def to_ary = [:d1, :d2]; end
        def each_of(vals)
          out = []
          vals.each { |a, b| out << [a, b] }
          out
        end
        each_of([[1, 2]])
        each_of([[1, 2]])
    ");
    assert_snapshot!(assert_compiles("each_of([AutosplatSub.new([1, 2]), AutosplatDuck.new]).inspect"),
        @r#""[[1, 2], [:d1, :d2]]""#);
}

/// The profiled-monomorphic dispatch guards the block ISEQ *after* the expansion has replaced
/// the one yielded Array with its elements. That guard has to side-exit to the interpreter's
/// own stack, which still holds just the Array, not to the expanded one.
#[test]
fn test_block_autosplat_iseq_guard_failure_restores_caller_stack() {
    set_call_threshold(2);
    eval("
        def each_of(vals)
          i = 0
          while i < vals.size
            yield vals[i]
            i += 1
          end
          nil
        end
        def run(vals, which)
          out = []
          if which == 0
            each_of(vals) { |a, b| out << [a, b, :first] }
          else
            each_of(vals) { |a, b| out << [a, b, :second] }
          end
          out
        end
        # Warm up only the first block so the yield site profiles it as monomorphic.
        4.times { run([[1, 2]], 0) }
    ");
    // Switching to the second block makes the ISEQ guard fail after the expansion.
    assert_snapshot!(assert_compiles_allowing_exits("run([[1, 2], [3], 4], 1).inspect"),
        @r#""[[1, 2, :second], [3, nil, :second], [4, nil, :second]]""#);
}

/// A single plain parameter does not auto-splat: `ambiguous_param0` is set and the block
/// receives the Array whole. The expansion must not fire.
#[test]
fn test_block_single_param_does_not_autosplat() {
    set_call_threshold(2);
    eval("
        def each_of(vals)
          out = []
          vals.each { |a| out << a }
          out
        end
        each_of([[1, 2]])
        each_of([[1, 2]])
    ");
    assert_snapshot!(assert_compiles("each_of([[1, 2], 3]).inspect"), @r#""[[1, 2], 3]""#);
}

/// Blocks with optional, rest, or post parameters are not `rb_simple_iseq_p`, so they must
/// not take the expansion: their auto-splat nil-fills and packs in ways it does not model.
#[test]
fn test_block_autosplat_skips_non_simple_params() {
    set_call_threshold(2);
    eval("
        def each_opt(vals)
          out = []
          vals.each { |a, b = :dflt| out << [a, b] }
          out
        end
        def each_rest(vals)
          out = []
          vals.each { |a, *r| out << [a, r] }
          out
        end
        def each_post(vals)
          out = []
          vals.each { |a, *m, z| out << [a, m, z] }
          out
        end
        2.times do
          each_opt([[1, 2]])
          each_rest([[1, 2]])
          each_post([[1, 2, 3]])
        end
    ");
    assert_snapshot!(assert_compiles("each_opt([[1, 2], [3]]).inspect"), @r#""[[1, 2], [3, :dflt]]""#);
    assert_snapshot!(assert_compiles("each_rest([[1, 2, 3]]).inspect"), @r#""[[1, [2, 3]]]""#);
    assert_snapshot!(assert_compiles("each_post([[1, 2, 3, 4]]).inspect"), @r#""[[1, [2, 3], 4]]""#);
}

/// Nested destructuring (`|a, (b, c)|`) has an extra `expandarray` in the block body but the
/// outer arity is still simple, so the expansion applies and the body does the rest.
#[test]
fn test_block_autosplat_nested_destructuring() {
    set_call_threshold(2);
    eval("
        def each_of(vals)
          out = []
          vals.each { |a, (b, c)| out << [a, b, c] }
          out
        end
        each_of([[1, [2, 3]]])
        each_of([[1, [2, 3]]])
    ");
    assert_snapshot!(assert_compiles("each_of([[1, [2, 3]], [4, [5, 6]]]).inspect"),
        @r#""[[1, 2, 3], [4, 5, 6]]""#);
}

/// The real-world shape the larger threshold is for: `Array#each` is 41 instructions, so the
/// ordinary threshold leaves `ary.each { return ... }` throwing on every call.
#[test]
fn test_inline_array_each_to_erase_block_non_local_return() {
    set_call_threshold(2);
    eval("
        def test(ary)
          ary.each { |x| return x * 2 if x > 2 }
          -1
        end
        test([1, 2, 3])
        test([1, 2, 3])
    ");
    assert_snapshot!(assert_compiles("test([1, 5, 3])"), @"10");
}

/// The relaxation is not limited to blocks with a non-local `return`: a plain
/// `ary.each { ... }` gets the oversized `Array#each` inlined too, so its `yield` reaches
/// the direct block dispatch instead of `rb_vm_invokeblock()`. `assert_inlines` is what
/// pins the relaxation down -- at the plain threshold `Array#each` is too large and the
/// program inlines nothing.
#[test]
fn test_inline_array_each_to_dispatch_yield_directly() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def test(ary)
              out = 0
              ary.each { |x| out += x }
              out
            end
            test([1, 2, 3])
            test([1, 2, 3])
            test([4, 5, 6])
        "), @"15");
    });
}

/// A caller already over its cumulative inlining budget still gets the iterator inlined, which
/// is what puts its `yield` on the direct block dispatch. A budget of 0 stands in for the big
/// Rails methods that spend the real one long before they reach their `.each`.
#[test]
fn test_inline_iterator_past_exhausted_budget() {
    with_inlining(|| {
        let old_budget = get_option!(inline_budget);
        set_inline_budget(1);
        let result = assert_inlines("
            def test(ary)
              out = 0
              ary.each { |x| out += x }
              out
            end
            test([1, 2, 3])
            test([1, 2, 3])
            test([4, 5, 6])
        ");
        set_inline_budget(old_budget);
        assert_snapshot!(result, @"15");
    });
}

/// The allowance is not unlimited: a caller over budget gets a few iterator bodies, not one per
/// `.each` it contains. The `.each` calls past the cap keep their out-of-line dispatch, which
/// respecializes on the block it sees, so exits are expected here.
#[test]
fn test_inline_iterator_past_budget_is_capped() {
    with_inlining(|| {
        let old_budget = get_option!(inline_budget);
        set_inline_budget(1);
        let result = assert_inlines_allowing_exits("
            def test(ary)
              out = 0
              ary.each { |x| out += x }
              ary.each { |x| out += x }
              ary.each { |x| out += x }
              ary.each { |x| out += x }
              ary.each { |x| out += x }
              out
            end
            test([1, 2, 3])
            test([1, 2, 3])
            test([4, 5, 6])
        ");
        set_inline_budget(old_budget);
        assert_snapshot!(result, @"75");
    });
}

/// A multi-parameter block over pairs, which needs the yielded Array destructured on top of
/// the oversized-iterator relaxation.
#[test]
fn test_inline_array_each_to_dispatch_autosplat_yield_directly() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def test(pairs)
              out = []
              pairs.each { |a, b| out << [b, a] }
              out
            end
            test([[1, 2]])
            test([[1, 2]])
            test([[1, 2], [3, 4]]).inspect
        "), @r#""[[2, 1], [4, 3]]""#);
    });
}

/// The relaxation only applies to a callee whose `yield` can benefit from it: an oversized
/// method with no `yield` in it stays out of line however it is called.
///
/// `inline_method_count` is global, so the padding below deliberately avoids calling any
/// method that would itself be inlined -- indexing the array rather than calling
/// `Array#last`, which is a leaf builtin the general inliner would inline and count.
#[test]
fn test_oversized_callee_without_yield_is_not_relaxed() {
    with_inlining(|| {
        let counters = crate::state::ZJITState::get_counters();
        let before = counters.inline_method_count;
        assert_snapshot!(assert_compiles("
            def no_yield(a)
              pad0 = a + 1
              pad1 = pad0 + 2
              pad2 = pad1 + 3
              pad3 = pad2 + 4
              pad4 = pad3 + 5
              pad5 = pad4 + 6
              pad6 = pad5 + 7
              [pad0, pad1, pad2, pad3, pad4, pad5, pad6][6]
            end
            def test(a) = no_yield(a) { :unused_block }
            test(1)
            test(1)
            test(2)
        "), @"30");
        assert_eq!(before, counters.inline_method_count,
            "an oversized callee without a `yield` must not get the relaxed threshold");
    });
}

#[test]
fn test_inline_block_non_local_return_with_args() {
    set_call_threshold(2);
    eval("
        def inner(a, b) = yield(a, b)
        def test(a, b)
          inner(a, b) { |x, y| return x + y }
          99
        end
        test(1, 2)
        test(1, 2)
    ");
    assert_snapshot!(assert_compiles("test(3, 4)"), @"7");
}

#[test]
fn test_inline_block_conditional_non_local_return() {
    set_call_threshold(2);
    eval("
        def inner = yield
        def test(x)
          y = inner { return :early if x }
          [y, :late]
        end
        test(true)
        test(false)
    ");
    assert_snapshot!(assert_compiles("test(true)"), @":early");
    assert_snapshot!(assert_compiles("test(false)"), @"[nil, :late]");
}

#[test]
fn test_inline_block_non_local_return_runs_ensure_in_caller() {
    // An `ensure` in the frame the `return` unwinds to must still run, so this block is
    // not inlined and falls back to the throw.
    set_call_threshold(2);
    eval("
        $ran = false
        def inner = yield
        def test
          inner { return 1 }
        ensure
          $ran = true
        end
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("[test, $ran]"), @"[1, true]");
}

#[test]
fn test_inline_block_non_local_return_runs_ensure_in_block() {
    // Likewise for an `ensure` inside the block itself.
    set_call_threshold(2);
    eval("
        $ran = false
        def inner = yield
        def test
          inner { begin; return 1; ensure; $ran = true; end }
        end
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("[test, $ran]"), @"[1, true]");
}

#[test]
fn test_inline_block_non_local_return_from_nested_block_owner() {
    // `return` inside a block nested in another block escapes to the enclosing method, not
    // to the block frame ZJIT would be returning from, so the outer block is not eligible.
    set_call_threshold(2);
    eval("
        def inner = yield
        def test
          [1, 2].each do
            inner { return :deep }
          end
          :shallow
        end
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test"), @":deep");
}

#[test]
fn test_inline_block_non_local_return_keeps_frames_walkable() {
    set_call_threshold(2);
    eval("
        def inner = yield
        def test
          inner { return caller_locations(0, 3).map(&:label) }
        end
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test"), @r#"["block in Object#test", "Object#inner", "Object#test"]"#);
}

#[test]
fn test_inline_block_non_local_return_restores_the_callers_sp() {
    // A non-local `return` out of an inlined block returns with the inlined frames still
    // pushed, and it has to hand the SP register back the way it received it: a direct
    // JIT-to-JIT caller restores its own SP with a fixed `sub` after the call. `find`
    // below is padded past the inline threshold so `test` really does call it directly,
    // and `test` then spills `hit` and `count` for the block to read out of its EP, which
    // only lands in the right slots if SP survived the call.
    set_call_threshold(2);
    eval("
        def each3
          yield 1
          yield 2
        end
        def find(key)
          pad0 = key.to_s
          pad1 = pad0.size
          pad2 = pad1 + 1
          pad3 = pad2 * 2
          pad4 = pad3 - 1
          each3 { |v| return [v, key, pad4] if v == 2 }
          [pad0, pad1, pad2, pad3, pad4]
        end
        def test(key)
          hit = find(key)
          count = 0
          [1, 2].each { count += hit.size }
          [hit, count]
        end
        test(:a)
        test(:a)
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test(:ab)"), @r#"[[2, :ab, 5], 6]"#);
}

#[test]
fn test_inline_block_non_local_return_keeps_the_callers_frame_intact() {
    // Distilled from tool/lib/leakchecker.rb, which crashed with `vm_get_cref:
    // unreachable`. `find_fds` returns non-locally out of a block inlined at the `yield`
    // in `each2`, so it left its direct JIT-to-JIT caller's SP too high by the inlined
    // frames' offsets. `check` then wrote the frame for its own next call over its frame's
    // flags, zeroing the frame magic, and `setclassvariable` walked the resulting ep chain
    // looking for a cref that was no longer there. `find_fds` is padded past the inline
    // threshold so that `check` really does call it directly rather than inline it, and
    // the `each` block below both reads a local `check` spilled for it and writes the
    // class variable through its cref.
    set_call_threshold(2);
    eval("
        class Leaks
          @@leaked = nil
          def each2
            yield 1
            yield 2
          end
          def find_fds(dirs)
            pad0 = dirs.to_s
            pad1 = pad0.size
            pad2 = pad1 + 1
            pad3 = pad2 * 2
            pad4 = pad3 - 1
            each2 { |v| return [v, pad4] if v == 2 }
            [pad0, pad1, pad2, pad3, pad4]
          end
          def check(name)
            live = find_fds(name)
            @@leaked = 0
            [1, 2].each { @@leaked += live.size }
            [live, @@leaked]
          end
        end
        Leaks.new.check('ab')
        Leaks.new.check('ab')
    ");
    assert_snapshot!(assert_compiles_allowing_exits("Leaks.new.check('abc')"), @r#"[[2, 7], 4]"#);
}

#[test]
fn test_inline_block_raise_unwinds_through_inlined_frames() {
    set_call_threshold(2);
    eval("
        def inner = yield
        def test
          inner { raise 'boom' }
        rescue => e
          e.message
        end
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test"), @r#""boom""#);
}

#[test]
fn test_throw_break_with_value_from_each() {
    set_call_threshold(2);
    eval("
        def test(a) = a.each { |x| break x * 10 if x == 3 }
        test([1, 2, 3, 4])
        test([1, 2, 3, 4])
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test([1, 2, 3, 4])"), @"30");
}

#[test]
fn test_throw_no_break_returns_receiver() {
    set_call_threshold(2);
    eval("
        def test(a) = a.each { |x| break x if x == 99 }
        test([1, 2])
        test([1, 2])
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test([1, 2])"), @"[1, 2]");
}

#[test]
fn test_throw_break_across_jit_to_jit_call() {
    set_call_threshold(2);
    eval("
        def inner = yield
        def outer = inner { break 7 }
        def test = outer
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"7");
}

#[test]
fn test_throw_break_three_frames_deep() {
    set_call_threshold(2);
    eval("
        def innermost(a) = a.each { |x| break x if x.even? }
        def middle(a) = innermost(a)
        def test(a) = middle(a)
        test([1, 2, 3])
        test([1, 2, 3])
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test([1, 2, 3])"), @"2");
}

#[test]
fn test_throw_break_value_used_by_caller() {
    set_call_threshold(2);
    eval("
        def test(a)
          v = a.each { |x| break x + 100 if x > 1 }
          v.to_s
        end
        test([1, 2, 3])
        test([1, 2, 3])
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test([1, 2, 3])"), @r#""102""#);
}

#[test]
fn test_throw_break_search_loop() {
    set_call_threshold(2);
    eval("
        def test(a) = a.each_with_index { |x, i| break i if x == :b }
        test([:a, :b, :c])
        test([:a, :b, :c])
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test([:a, :b, :c])"), @"1");
}

#[test]
fn test_throw_break_runs_ensure() {
    set_call_threshold(2);
    eval("
        def test(a)
          log = []
          r = a.each do |x|
            begin
              break x if x == 2
            ensure
              log << x
            end
          end
          [r, log]
        end
        test([1, 2, 3])
        test([1, 2, 3])
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test([1, 2, 3])"), @"[2, [1, 2]]");
}

#[test]
fn test_throw_return_from_proc() {
    set_call_threshold(2);
    eval("
        def test
          p = proc { return 5 }
          p.call
          99
        end
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"5");
}

#[test]
fn test_throw_return_from_lambda() {
    set_call_threshold(2);
    eval("
        def test
          l = lambda { return 5 }
          l.call + 1
        end
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"6");
}

#[test]
fn test_throw_orphan_break_raises_local_jump_error() {
    set_call_threshold(2);
    eval("
        def test
          pr = proc { break 1 }
          begin
            pr.call
          rescue LocalJumpError => e
            e.class
          end
        end
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"LocalJumpError");
}

#[test]
fn test_throw_retry_in_rescue() {
    set_call_threshold(2);
    eval("
        def test
          tries = 0
          begin
            tries += 1
            raise 'boom' if tries < 3
            tries
          rescue
            retry
          end
        end
        test
        test
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"3");
}

#[test]
fn test_throw_next_with_ensure() {
    set_call_threshold(2);
    eval("
        def test(a)
          a.map do |x|
            begin
              next x * 2
            ensure
              nil
            end
          end
        end
        test([1, 2, 3])
        test([1, 2, 3])
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test([1, 2, 3])"), @"[2, 4, 6]");
}

#[test]
fn test_throw_break_inner_loop_repeatedly() {
    set_call_threshold(2);
    eval("
        def test(a)
          sum = 0
          a.each do |x|
            a.each do |y|
              break if y > 2
              sum += x * y
            end
          end
          sum
        end
        test([1, 2, 3])
        test([1, 2, 3])
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test([1, 2, 3])"), @"18");
}

#[test]
fn test_yield_autosplat() {
    // {|a, b|} auto-splats a single Array arg for yield (falls back).
    set_call_threshold(2);
    eval("
        def via_yield = yield([3, 4])
        def test_yield = via_yield { |a, b| a + b }
        test_yield; test_yield
    ");
    assert_snapshot!(assert_compiles_allowing_exits("test_yield"), @"7");
}

#[test]
fn test_yield_next() {
    // next(val) compiles to leave (not throw), so yield inlines invocation and returns val.
    set_call_threshold(2);
    eval("
        def via_yield = yield
        def test_yield = via_yield { next 7 }
        test_yield; test_yield
    ");
    assert_snapshot!(assert_compiles("test_yield"), @"7");
}

#[test]
fn test_yield_inline_ensure_runs() {
    // The ensure body must run on the normal inlined invocation yield path.
    set_call_threshold(2);
    eval("
        def foo = yield
        $log = []
        def driver
          foo do
            begin
              42
            ensure
              $log << :ensured
            end
          end
        end
        driver
        driver
    ");
    assert_snapshot!(assert_compiles_allowing_exits("$log.clear; [driver, $log]"), @"[42, [:ensured]]");
}

#[test]
fn test_getblockparam() {
    eval("
        def test(&blk)
          blk
        end
        test { 2 }.call
    ");
    assert_contains_opcode("test", YARVINSN_getblockparam);
    assert_snapshot!(assert_compiles("test { 2 }.call"), @"2");
}

#[test]
fn test_setblockparam() {
    eval("
        def test(&block)
          block = proc { 3 }
          blk = block
          blk.call
        end
        test { 1 }
    ");
    assert_contains_opcode("test", YARVINSN_setblockparam);
    assert_snapshot!(assert_compiles("test { 1 }"), @"3");
}

#[test]
fn test_setblockparam_nested_block() {
    eval("
        def test(&block)
          proc do
            block = proc { 3 }
            blk = block
            blk.call
          end.call
        end
        test { 1 }
    ");
    assert_snapshot!(assert_compiles("test { 1 }"), @"3");
}

#[test]
fn test_getblockparamproxy_after_setblockparam() {
    eval("
        def test(&block)
          block = proc { 3 }
          block.call
        end
        test { 1 }
    ");
    assert_contains_opcode("test", YARVINSN_setblockparam);
    assert_snapshot!(assert_compiles("test { 1 }"), @"3");
}

#[test]
fn test_getblockparam_used_twice_in_args() {
    eval("
        def f(*args) = args
        def test(&blk)
          b = blk
          f(*[1], blk)
          blk
        end
        test {1}.call
    ");
    assert_contains_opcode("test", YARVINSN_getblockparam);
    assert_snapshot!(assert_compiles("test {1}.call"), @"1");
}

#[test]
fn test_optimized_method_call_proc_call() {
    eval("
        def test(p)
          p.call(1)
        end
        test(proc { |x| x * 2 })
    ");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
    assert_snapshot!(assert_compiles("test(proc { |x| x * 2 })"), @"2");
}

#[test]
fn test_optimized_method_call_proc_aref() {
    eval("
        def test(p)
          p[2]
        end
        test(proc { |x| x * 2 })
    ");
    assert_contains_opcode("test", YARVINSN_opt_aref);
    assert_snapshot!(assert_compiles("test(proc { |x| x * 2 })"), @"4");
}

#[test]
fn test_optimized_method_call_proc_yield() {
    eval("
        def test(p)
          p.yield(3)
        end
        test(proc { |x| x * 2 })
    ");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
    assert_snapshot!(assert_compiles("test(proc { |x| x * 2 })"), @"6");
}

#[test]
fn test_optimized_method_call_proc_kw_splat() {
    eval("
        def test(p, h)
          p.call(**h)
        end
        test(proc { |**kw| kw[:a] + kw[:b] }, { a: 1, b: 2 })
    ");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
    assert_snapshot!(assert_compiles("test(proc { |**kw| kw[:a] + kw[:b] }, { a: 1, b: 2 })"), @"3");
}

#[test]
fn test_optimized_method_call_proc_call_splat() {
    assert_snapshot!(inspect("
        p = proc { |x| x + 1 }
        def test(p)
          ary = [42]
          p.call(*ary)
        end
        test(p)
        test(p)
    "), @"43");
}

#[test]
fn test_optimized_method_call_proc_call_kwarg() {
    assert_snapshot!(inspect("
        p = proc { |a:| a }
        def test(p)
          p.call(a: 1)
        end
        test(p)
        test(p)
    "), @"1");
}

#[test]
fn test_setlocal_on_eval_with_spill() {
    assert_snapshot!(inspect("
        @b = binding
        eval('a = 1; itself', @b)
        eval('a', @b)
    "), @"1");
}

#[test]
fn test_nested_local_access() {
    assert_snapshot!(inspect("
        1.times do |l2|
          1.times do |l1|
            define_method(:test) do
              l1 = 1
              l2 = 2
              l3 = 3
              [l1, l2, l3]
            end
          end
        end

        test
        test
        test
    "), @"[1, 2, 3]");
}

#[test]
fn test_send_with_local_written_by_blockiseq() {
    assert_snapshot!(inspect("
        def test
          l1 = nil
          l2 = nil
          tap do |_|
            l1 = 1
            tap do |_|
              l2 = 2
            end
          end

          [l1, l2]
        end

        test
        test
    "), @"[1, 2]");
}

#[test]
fn test_send_does_not_reload_local_untouched_by_blockiseq() {
    // https://github.com/Shopify/ruby/issues/976: a call with a block must not
    // reload locals the block never assigns, otherwise it reads a stale stack
    // slot and clobbers the correct SSA value (here, `a`).
    eval("
        def foo(&block) = 1

        def test
          a = 1
          foo {}
          a
        end

        test
    ");
    assert_contains_opcode("test", YARVINSN_send);
    assert_snapshot!(assert_compiles("test"), @"1");
}

#[test]
fn test_no_ep_escape_patch_point_after_send_does_not_repeat_send() {
    eval(r#"
        $send_count = 0

        def test
          captured = nil
          tap do |_|
            $send_count += 1
            -> { captured } if $send_count == 2
          end
          $send_count
        end
    "#);
    assert_contains_opcode("test", YARVINSN_send);
    assert_snapshot!(assert_compiles_allowing_exits("[test, test, test]"), @"[1, 2, 3]");
}

#[test]
fn test_no_ep_escape_side_exit_restores_locals_while_oom() {
    // A regression test for stub compilation failures on OOM. Functions patched by NoEPEscape
    // is unsafe to enter (FrameState uses without_locals() and doesn't spill the entry state),
    // so even under OOM, the re-stub after invalidation must succeed.
    set_mem_bytes(2 * 1024 * 1024);
    set_inline_threshold(0);
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        class Foo
          def initialize = @perm = 7
          def callee(esc, local_to_spill = "spilled", perm = @perm)
            binding if esc
            local_to_spill
          end
        end
        def kaller(foo, esc) = foo.callee(esc)

        foo = Foo.new
        300.times { kaller(foo, false) } # compile callee (with its NoEPEscape patch point) and the kaller->callee edge

        # Fill the code region so the re-stub after the EP escape fails with OutOfMemory.
        1000.times do |i|
          body = (0...25).map { |k| "u#{k} = #{i} + #{k}; s += u#{k}" }.join("; ")
          eval "def big#{i}(a = 1); s = 0; #{body}; s; end"
        end
        1000.times { |i| 2.times { send(:"big#{i}") } }

        kaller(foo, true) # escape callee's EP; the kaller->callee re-stub OOMs
        # Re-enter the patched callee. Each call must still return "spilled"; on the buggy
        # build local_to_spill is read from a stale stack slot and comes back as junk.
        300.times.all? { kaller(foo, false) == "spilled" }
    "#), @"true");
}

#[test]
fn test_send_without_block() {
    assert_snapshot!(inspect("
        def foo = 1
        def bar(a) = a - 1
        def baz(a, b) = a - b

        def test1 = foo
        def test2 = bar(3)
        def test3 = baz(4, 1)

        [test1, test2, test3]
    "), @"[1, 2, 3]");
}

#[test]
fn test_send_with_six_args() {
    assert_snapshot!(inspect("
        def foo(a1, a2, a3, a4, a5, a6)
          [a1, a2, a3, a4, a5, a6]
        end

        def test
          foo(1, 2, 3, 4, 5, 6)
        end

        test # profile send
        test
    "), @"[1, 2, 3, 4, 5, 6]");
}

#[test]
fn test_send_optional_arguments() {
    assert_snapshot!(inspect("
        def test(a, b = 2) = [a, b]
        def entry = [test(1), test(3, 4)]
        entry
        entry
    "), @"[[1, 2], [3, 4]]");
}

#[test]
fn test_send_rest_arguments() {
    eval("
        def test(*args) = args
        def entry = test(1, 2, 3)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"[1, 2, 3]");
}

#[test]
fn test_send_many_rest_arguments() {
    eval("
        def test(*args) = args.length
        def entry = test(1, 2, 3, 4, 5, 6, 7)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"7");
}

#[test]
fn test_send_rest_arguments_with_post() {
    eval("
        def test(a, *args, z) = [a, args, z]
        def entry = test(1, 2, 3, 4)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"[1, [2, 3], 4]");
}

#[test]
fn test_send_rest_arguments_with_keyword() {
    eval("
        def test(*args, k:) = [args, k]
        def entry = test(1, 2, k: 40)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"[[1, 2], 40]");
}

#[test]
fn test_send_rest_arguments_with_optional_keyword_default() {
    eval("
        def test(*args, k: 40) = [args, k]
        def entry = test(1, 2)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"[[1, 2], 40]");
}

#[test]
fn test_send_optional_and_rest_arguments() {
    eval("
        def test(a, b = 2, *rest) = [a, b, rest]
        def entry = [test(1), test(3, 4), test(5, 6, 7, 8)]
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"[[1, 2, []], [3, 4, []], [5, 6, [7, 8]]]");
}

#[test]
fn test_send_optional_return_default_without_argument() {
    eval("
        def test(arg = nil || (return :default)) = arg
        def entry = test
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @":default");
}

#[test]
fn test_send_optional_return_default_with_argument() {
    eval("
        def test(arg = nil || (return :default)) = arg
        def entry = test(1)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"1");
}

#[test]
fn test_send_keyword_to_positional_hash() {
    eval("
        def test(arg) = arg
        def entry = test(k: 1)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"{k: 1}");
}

#[test]
fn test_send_multiple_keywords_to_positional_hash() {
    eval("
        def test(arg) = arg
        def entry = test(k: 1, v: 2)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"{k: 1, v: 2}");
}

#[test]
fn test_send_positional_and_keyword_to_positional_hash() {
    eval("
        def test(a, b) = [a, b]
        def entry = test(1, k: 2)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"[1, {k: 2}]");
}

#[test]
fn test_send_optional_and_keyword_to_positional_hash() {
    eval("
        def test(a, b = 2) = [a, b]
        def entry = test(k: 1)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"[{k: 1}, 2]");
}

#[test]
fn test_send_rest_arguments_with_keyword_to_positional_hash() {
    eval("
        def test(*args) = args
        def entry = test(k: 1)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"[{k: 1}]");
}

#[test]
fn test_send_optional_and_rest_arguments_with_keyword_to_positional_hash() {
    eval("
        def test(a, b = 2, *rest) = [a, b, rest]
        def entry = test(1, k: 3)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"[1, {k: 3}, []]");
}

#[test]
fn test_send_rest_and_post_arguments_with_keyword_to_positional_hash() {
    eval("
        def test(a, *rest, b) = [a, rest, b]
        def entry = test(1, 2, k: 3)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"[1, [2], {k: 3}]");
}

#[test]
fn test_send_keyword_splat_to_positional_hash_fallback() {
    eval("
        def test(arg) = arg
        def entry = test(**{ k: 1 })
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"{k: 1}");
}

#[test]
fn test_send_no_kwarg_to_positional_hash_fallback() {
    eval("
        def test(arg, **nil) = arg
        def entry
          test(k: 1)
        rescue ArgumentError
          :argument_error
        end
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @":argument_error");
}

#[test]
fn test_send_ruby2_keywords_to_positional_hash_fallback() {
    eval("
        def target(k:) = k
        ruby2_keywords def forward(*args) = target(*args)
        def entry = forward(k: 1)
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"1");
}

#[test]
fn test_send_splat_expanded_to_positional_args() {
    eval("
        def target(a, b) = a - b
        def entry(args) = target(*args)
        entry([1, 2])
    ");
    assert_snapshot!(assert_compiles("entry([10, 3])"), @"7");
}

#[test]
fn test_send_splat_expanded_with_leading_positional_arg() {
    eval("
        def target(a, b, c) = [a, b, c]
        def entry(args) = target(1, *args)
        entry([2, 3])
    ");
    assert_snapshot!(assert_compiles("entry([4, 5])"), @"[1, 4, 5]");
}

#[test]
fn test_send_splat_expanded_into_rest_parameter() {
    eval("
        def target(*args) = args
        def entry(args) = target(*args)
        entry([1, 2])
    ");
    assert_snapshot!(assert_compiles("entry([3, 4])"), @"[3, 4]");
}

#[test]
fn test_send_splat_with_changed_length_side_exits() {
    eval("
        def target(*args) = args.sum
        def entry(args) = target(*args)
        entry([1, 2])
        entry([1, 2])
    ");
    // The guarded length no longer matches, so the call falls back to the interpreter.
    assert_snapshot!(inspect("entry([1, 2, 3])"), @"6");
}

#[test]
fn test_send_splat_of_ruby2_keywords_hash_side_exits() {
    // `forward` is not itself ruby2_keywords, so it speculates on the splat, but the array it
    // receives ends in a flagged Hash that the interpreter turns back into keywords.
    eval("
        def target(k: 0) = k
        def forward(args) = target(*args)
        ruby2_keywords def outer(*args) = forward(args)
        outer(k: 1)
        outer(k: 1)
    ");
    assert_snapshot!(inspect("outer(k: 2)"), @"2");
}

#[test]
fn test_send_rest_arguments_with_block_literal() {
    eval("
        def test(*args) = yield args.length
        def entry = test(1, 2, 3) { |n| n + 4 }
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"7");
}

#[test]
fn test_send_rest_arguments_with_block_param() {
    eval("
        def test(*args, &block) = block.call(args.length)
        def entry = test(1, 2, 3) { |n| n + 5 }
        entry
    ");
    assert_snapshot!(assert_compiles("entry"), @"8");
}

#[test]
fn test_send_nil_block_arg() {
    assert_snapshot!(inspect("
        def test = block_given?
        def entry = test(&nil)
        test
        test
    "), @"false");
}

#[test]
fn test_attr_reader_two_shapes_per_class_in_polymorphic_arm() {
    // A polymorphic call site branches on the receiver's class, so one arm can be entered by
    // objects of that class with different shapes. Each shape the profile saw gets its own ivar
    // load; the rest still have to read correctly through the C fallback.
    assert_snapshot!(inspect("
        class C
          attr_reader :foo
          def early = (@foo = 1; @bar = 2)
          def late  = (@bar = 3; @foo = 4)
          def third = (@baz = 5; @qux = 6; @foo = 7)
        end
        class D
          attr_reader :foo
          def initialize = @foo = :d
        end
        objs = []
        200.times do |i|
          c = C.new
          case i % 3
          when 0 then c.early
          when 1 then c.late
          else c.third
          end
          objs << c
          objs << D.new
        end
        def read(o) = o.foo
        out = objs.map { |o| read(o) }
        [out.tally.sort_by(&:to_s), read(C.new)]
    "), @"[[[1, 67], [4, 67], [7, 66], [:d, 200]], nil]");
}

#[test]
fn test_send_mixed_nil_and_non_nil_block_arg() {
    // A `&block` forwarding site that sees both nil and non-nil blocks is split on nil, so the
    // no-block calls become direct sends. Both branches must still produce the right answer.
    assert_snapshot!(inspect("
        def callee(n, &block)
          block ? block.call(n) : n
        end
        def forward(n, &block) = callee(n, &block)
        results = []
        100.times do |i|
          results << forward(i)
          results << forward(i) { |n| n * 2 }
        end
        [forward(7), forward(7) { |n| n + 1 }, results.last(2)]
    "), @"[7, 8, [99, 198]]");
}

#[test]
fn test_send_nil_block_arg_split_polymorphic_receiver() {
    // The nil branch of the split dispatches on the receiver type, so a forwarding site shared by
    // several receiver classes still has to pick the right method for each.
    assert_snapshot!(inspect("
        class A; def value(&block) = block ? block.call(1) : 1; end
        class B; def value(&block) = block ? block.call(2) : 2; end
        def forward(obj, &block) = obj.value(&block)
        out = []
        200.times do |i|
          obj = i.even? ? A.new : B.new
          out << forward(obj)
          out << forward(obj) { |n| n * 10 } if i % 3 == 0
        end
        [forward(A.new), forward(B.new), forward(A.new) { |n| n * 10 }, out.sum]
    "), @"[1, 2, 10, 1300]");
}

#[test]
fn test_send_forwards_block_param_proxy() {
    // `bar(&blk)` where `blk` comes from `getblockparamproxy` passes this frame's own block
    // handler to the callee, so the callee's `yield` and `&b` parameter have to see the block
    // the outermost caller gave. Every kind of handler goes through the same site: a literal
    // block, no block at all, a Proc, and a symbol-to-proc.
    assert_snapshot!(inspect("
        def callee(n, &b)
          [n, block_given? ? yield(n) : nil, b ? b.call(n) : nil]
        end
        def forward(n, &blk) = callee(n, &blk)
        def pass_proc(n, p) = forward(n, &p)
        out = []
        200.times do |i|
          out << forward(i) { |x| x + 1 }
          out << forward(i)
          out << pass_proc(i, ->(x) { x * 2 })
          out << pass_proc(i, nil)
        end
        out.last(4)
    "), @"[[199, 200, 200], [199, nil, nil], [199, 398, 398], [199, nil, nil]]");
}

#[test]
fn test_send_forwards_block_param_proxy_after_setblockparam() {
    // Assigning the block parameter makes `getblockparamproxy` hand back the materialized Proc
    // instead of the proxy, so the branch on the proxy has to send those calls the other way.
    assert_snapshot!(inspect("
        def callee(n, &b) = [n, b ? b.call(n) : nil]
        def forward(n, replace, &blk)
          blk = ->(x) { x * 100 } if replace
          callee(n, &blk)
        end
        out = []
        200.times do |i|
          out << forward(i, false) { |x| x + 1 }
          out << forward(i, true) { |x| x + 1 }
        end
        out.last(2)
    "), @"[[199, 200], [199, 19900]]");
}

#[test]
fn test_send_proc_block_arg_passes_through() {
    // A `&blk` argument holding a plain Proc is its own block handler in the interpreter, so the
    // direct send installs it as the callee frame's specval.
    assert_snapshot!(inspect("
        def callee(n, &b) = [n, block_given?, b.call(n)]
        def entry(n, p) = callee(n, &p)
        doubler = ->(x) { x * 2 }
        out = nil
        200.times { |i| out = entry(i, doubler) }
        [out, entry(5, proc { |x| x + 1 })]
    "), @"[[199, true, 398], [5, true, 6]]");
}

#[test]
fn test_send_proc_block_arg_guard_rejects_other_block_args() {
    // The Proc guard has to send a `&:sym` or a Method through the interpreter, which converts it
    // with `to_proc` rather than using it as the block handler directly.
    assert_snapshot!(inspect("
        def callee(n, &b) = b.call(n)
        def entry(n, p) = callee(n, &p)
        doubler = ->(x) { x * 2 }
        200.times { |i| entry(i, doubler) }
        [entry(5, doubler), entry(-6, :abs), entry(7, 2.method(:+))]
    "), @"[10, 6, 9]");
}

#[test]
fn test_send_block_param_proxy_from_block_body() {
    // `foo(&blk)` inside a block reads the block parameter of the enclosing method, and
    // `VM_CF_BLOCK_HANDLER` resolves through the local EP to that same frame.
    assert_snapshot!(inspect("
        def callee(n, &b) = [n, b ? b.call(n) : nil]
        def forward(n, &blk)
          [1].map { callee(n, &blk) }.first
        end
        out = nil
        200.times { |i| out = forward(i) { |x| x + 3 } }
        [out, forward(9)]
    "), @"[[199, 202], [9, nil]]");
}

#[test]
fn test_send_forwards_block_arg_to_cfunc() {
    // A C method takes its block from the frame's specval too, so `&blk` forwarding reaches
    // `Hash#fetch` (variadic) and `Array#bsearch_index` (fixed arity) as a direct C call. The
    // no-block calls must still see no block.
    assert_snapshot!(inspect("
        def fetch(h, k, &b) = h.fetch(k, &b)
        def search(a, &b) = a.bsearch_index(&b)
        def each_with_proc(a, p) = a.each(&p)
        out = []
        300.times do
          out << fetch({ a: 1 }, :b) { |k| \"no #{k}\" }
          out << fetch({ a: 1 }, :a) { |k| \"no #{k}\" }
          out << search([1, 3, 5, 7]) { |x| x >= 5 }
          out << (fetch({ a: 1 }, :b) rescue :keyerror)
        end
        seen = []
        each_with_proc([1, 2], ->(x) { seen << x })
        [out.last(4), seen]
    "), @r#"[["no b", 1, 2, :keyerror], [1, 2]]"#);
}

#[test]
fn test_send_symbol_block_arg() {
    assert_snapshot!(inspect("
        def test = [1, 2].map(&:to_s)
        test
        test
    "), @r#"["1", "2"]"#);
}

#[test]
fn test_send_variadic_with_block() {
    assert_snapshot!(inspect("
        A = [1, 2, 3]
        B = [\"a\", \"b\", \"c\"]

        def test
          result = []
          A.zip(B) { |x, y| result << [x, y] }
          result
        end

        test; test
    "), @r#"[[1, "a"], [2, "b"], [3, "c"]]"#);
}

#[test]
fn test_send_kwarg_optional() {
    assert_snapshot!(inspect("
        def test(a: 1, b: 2) = [a, b]
        def entry = test
        entry
        entry
    "), @"[1, 2]");
}

#[test]
fn test_send_kwarg_optional_too_many() {
    assert_snapshot!(inspect("
        def test(a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7, h: 8, i: 9, j: 10) = [a, b, c, d, e, f, g, h, i, j]
        def entry = test
        entry
        entry
    "), @"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
}

#[test]
fn test_send_kwarg_required_and_optional() {
    assert_snapshot!(inspect("
        def test(a:, b: 2) = [a, b]
        def entry = test(a: 3)
        entry
        entry
    "), @"[3, 2]");
}

#[test]
fn test_send_kwarg_to_hash() {
    assert_snapshot!(inspect("
        def test(hash) = hash
        def entry = test(a: 3)
        entry
        entry
    "), @"{a: 3}");
}

#[test]
fn test_send_kwarg_to_ccall() {
    assert_snapshot!(inspect(r#"
        def test(s) = s.each_line(chomp: true).to_a
        def entry = test(%(a\nb\nc))
        entry
        entry
    "#), @r#"["a", "b", "c"]"#);
}

#[test]
fn test_send_kwarg_and_block_to_ccall() {
    assert_snapshot!(inspect(r#"
        def test(s)
          a = []
          s.each_line(chomp: true) { |l| a << l }
          a
        end
        def entry = test(%(a\nb\nc))
        entry
        entry
    "#), @r#"["a", "b", "c"]"#);
}

#[test]
fn test_send_kwarg_with_too_many_args_to_c_call() {
    assert_snapshot!(inspect(r#"
        def test(a:, b:, c:, d:, e:) = sprintf("%s %s %s %s %s", a, b, c, d, kwargs: e)
        def entry = test(e: :e, d: :d, c: :c, a: :a, b: :b)
        entry
        entry
    "#), @r#""a b c d {kwargs: :e}""#);
}

#[test]
fn test_send_kwsplat() {
    assert_snapshot!(inspect("
        def test(a:) = a
        def entry = test(**{a: 3})
        entry
        entry
    "), @"3");
}

#[test]
fn test_send_kwrest() {
    assert_snapshot!(inspect("
        def test(**kwargs) = kwargs
        def entry = test(a: 3)
        entry
        entry
    "), @"{a: 3}");
}

#[test]
fn test_send_req_kwreq() {
    assert_snapshot!(inspect("
        def test(a, c:) = [a, c]
        def entry = test(1, c: 3)
        entry
        entry
    "), @"[1, 3]");
}

#[test]
fn test_send_req_opt_kwreq() {
    assert_snapshot!(inspect("
        def test(a, b = 2, c:) = [a, b, c]
        def entry = [test(1, c: 3), test(-1, -2, c: -3)]
        entry
        entry
    "), @"[[1, 2, 3], [-1, -2, -3]]");
}

#[test]
fn test_send_req_opt_kwreq_kwopt() {
    assert_snapshot!(inspect("
        def test(a, b = 2, c:, d: 4) = [a, b, c, d]
        def entry = [test(1, c: 3), test(-1, -2, d: -4, c: -3)]
        entry
        entry
    "), @"[[1, 2, 3, 4], [-1, -2, -3, -4]]");
}

#[test]
fn test_send_unexpected_keyword() {
    assert_snapshot!(inspect("
        def test(a: 1) = a*2
        def entry
          test(z: 2)
        rescue ArgumentError
          :error
        end

        entry
        entry
    "), @":error");
}

#[test]
fn test_pos_optional_with_maybe_too_many_args() {
    // The last call passes 8 args, which together with self exceed the C argument
    // registers (6 on x86_64, 8 on arm64), so it runs through the dynamic send path.
    assert_snapshot!(inspect("
        def target(a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h:) = [a, b, c, d, e, f, g, h]
        def test = [target(h: 8), target(10, 20, 30, h: 8), target(10, 20, 30, 40, 50, 60, 70, h: 80)]
        test
        test
    "), @"[[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]]");
}

#[test]
fn test_send_kwarg_partial_optional() {
    assert_snapshot!(inspect("
        def test(a: 1, b: 2, c: 3) = [a, b, c]
        def entry = [test, test(b: 20), test(c: 30, a: 10)]
        entry
        entry
    "), @"[[1, 2, 3], [1, 20, 3], [10, 2, 30]]");
}

#[test]
fn test_send_kwarg_optional_a_lot() {
    assert_snapshot!(inspect("
        def test(a: 1, b: 2, c: 3, d: 4, e: 5, f: 6) = [a, b, c, d, e, f]
        def entry = [test, test(d: 7, f: 9, e: 8), test(f: 12, e: 10, d: 8, c: 6, b: 4, a: 2)]
        entry
        entry
    "), @"[[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9], [2, 4, 6, 8, 10, 12]]");
}

#[test]
fn test_send_kwarg_non_constant_default() {
    assert_snapshot!(inspect("
        def make_default = 2
        def test(a: 1, b: make_default) = [a, b]
        def entry = [test, test(a: 10)]
        entry
        entry
    "), @"[[1, 2], [10, 2]]");
}

#[test]
fn test_send_kwarg_optional_static_with_side_exit() {
    assert_snapshot!(inspect("
        def callee(a: 1, b: 2)
          x = binding.local_variable_get(:a)
          [a, b, x]
        end

        def entry
          callee(a: 10)
        end

        entry
        entry
    "), @"[10, 2, 10]");
}

#[test]
fn test_send_hash_to_kwarg_only_method() {
    assert_snapshot!(inspect(r#"
        def callee(a:) = a

        def entry
          callee({a: 1})
        rescue ArgumentError
          "ArgumentError"
        end

        entry
        entry
    "#), @r#""ArgumentError""#);
}

#[test]
fn test_send_hash_to_optional_kwarg_only_method() {
    assert_snapshot!(inspect(r#"
        def callee(a: nil) = a

        def entry
          callee({a: 1})
        rescue ArgumentError
          "ArgumentError"
        end

        entry
        entry
    "#), @r#""ArgumentError""#);
}

#[test]
fn test_send_all_arg_types() {
    assert_snapshot!(inspect("
        def test(a, b = :opt, c, d:, e: :kwo) = [a, b, c, d, e, block_given?]
        def entry = test(:req, :post, d: :kwr) {}
        entry
        entry
    "), @"[:req, :opt, :post, :kwr, :kwo, true]");
}

#[test]
fn test_send_ccall_variadic_with_different_receiver_classes() {
    assert_snapshot!(inspect(r#"
        def test(obj) = obj.start_with?("a")
        [test("abc"), test(:abc)]
    "#), @"[true, true]");
}

#[test]
fn test_forwardable_iseq() {
    assert_snapshot!(inspect("
        def test(...) = 1
        test
        test
    "), @"1");
}

#[test]
fn test_sendforward() {
    eval("
        def callee(a, b) = [a, b]
        def test(...) = callee(...)
        test(1, 2)
    ");
    assert_contains_opcode("test", YARVINSN_sendforward);
    assert_snapshot!(assert_compiles("test(1, 2)"), @"[1, 2]");
}

#[test]
fn test_iseq_with_optional_arguments() {
    assert_snapshot!(inspect("
        def test(a, b = 2) = [a, b]
        [test(1), test(3, 4)]
    "), @"[[1, 2], [3, 4]]");
}

#[test]
fn test_invokesuper() {
    assert_snapshot!(inspect("
        class Foo
          def foo(a) = a + 1
          def bar(a) = a + 10
        end

        class Bar < Foo
          def foo(a) = super(a) + 2
          def bar(a) = super + 20
        end

        bar = Bar.new
        [bar.foo(3), bar.bar(30)]
    "), @"[6, 60]");
}

#[test]
fn test_invokesuper_to_iseq() {
    assert_snapshot!(inspect(r#"
        class A
          def foo
            "A"
          end
        end

        class B < A
          def foo
            ["B", super]
          end
        end

        def test
          B.new.foo
        end

        test  # profile invokesuper
        test  # compile + run compiled code
    "#), @r#"["B", "A"]"#);
}

#[test]
fn test_invokesuper_with_args() {
    assert_snapshot!(inspect(r#"
        class A
          def foo(x)
            x * 2
          end
        end

        class B < A
          def foo(x)
            ["B", super(x) + 1]
          end
        end

        def test
          B.new.foo(5)
        end

        test  # profile invokesuper
        test  # compile + run compiled code
    "#), @r#"["B", 11]"#);
}

#[test]
fn test_invokesuper_with_args_to_rest_param() {
    assert_snapshot!(inspect(r#"
        class A
          def foo(x, *rest)
            [x, rest]
          end
        end

        class B < A
          def foo(x, y, z)
            ["B", *super(x, y, z)]
          end
        end

        def test
          B.new.foo("a", "b", "c")
        end

        test  # profile invokesuper
        test  # compile + run compiled code
    "#), @r#"["B", "a", ["b", "c"]]"#);
}

#[test]
fn test_invokesuper_with_block() {
    assert_snapshot!(inspect(r#"
        class A
          def foo
            block_given? ? yield : "no_block"
          end
        end

        class B < A
          def foo
            ["B", super { "from_block" }]
          end
        end

        def test
          B.new.foo
        end

        test  # profile invokesuper
        test  # compile + run compiled code
    "#), @r#"["B", "from_block"]"#);
}

#[test]
fn test_invokesuper_to_cfunc_no_args() {
    assert_snapshot!(inspect(r#"
        class MyString < String
          def length
            ["MyString", super]
          end
        end

        def test
          MyString.new("abc").length
        end

        test  # profile invokesuper
        test  # compile + run compiled code
    "#), @r#"["MyString", 3]"#);
}

#[test]
fn test_invokesuper_to_cfunc_simple_args() {
    assert_snapshot!(inspect(r#"
        class MyString < String
          def include?(other)
            ["MyString", super(other)]
          end
        end

        def test
          MyString.new("abc").include?("bc")
        end

        test  # profile invokesuper
        test  # compile + run compiled code
    "#), @r#"["MyString", true]"#);
}

#[test]
fn test_invokesuper_to_cfunc_with_optional_arg() {
    assert_snapshot!(inspect(r#"
        class MyString < String
          def byteindex(needle, offset = 0)
            ["MyString", super(needle, offset)]
          end
        end

        def test
          MyString.new("hello world").byteindex("world")
        end

        test  # profile invokesuper
        test  # compile + run compiled code
    "#), @r#"["MyString", 6]"#);
}

#[test]
fn test_invokesuper_to_cfunc_varargs() {
    assert_snapshot!(inspect(r#"
        class MyString < String
          def end_with?(str)
            ["MyString", super(str)]
          end
        end

        def test
          MyString.new("abc").end_with?("bc")
        end

        test  # profile invokesuper
        test  # compile + run compiled code
    "#), @r#"["MyString", true]"#);
}

#[test]
fn test_invokesuper_to_cfunc_with_too_many_args_exits() {
    // `self` + eight args don't fit in C argument registers (6 on x86_64, 8 on arm64),
    // so the invokesuper to the cfunc must side-exit instead of emitting a CCall.
    unsafe extern "C" fn test_super_eight_args(
        _self: VALUE,
        a: VALUE,
        b: VALUE,
        c: VALUE,
        d: VALUE,
        e: VALUE,
        f: VALUE,
        g: VALUE,
        h: VALUE,
    ) -> VALUE {
        unsafe { rb_ary_new_from_args(8, a, b, c, d, e, f, g, h) }
    }

    with_rubyvm(|| {
        let superclass = define_class("ZJITSuperEightArgs", unsafe { rb_cObject });
        unsafe {
            rb_define_method(
                superclass,
                c"eight".as_ptr(),
                Some(std::mem::transmute::<
                    unsafe extern "C" fn(VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE) -> VALUE,
                    unsafe extern "C" fn(VALUE) -> VALUE,
                >(test_super_eight_args)),
                8,
            );
        }
    });

    assert_snapshot!(assert_compiles_allowing_exits(r#"
        class ZJITSuperEightArgsSubclass < ZJITSuperEightArgs
          def eight(a, b, c, d, e, f, g, h)
            super
          end
        end

        def test
          ZJITSuperEightArgsSubclass.new.eight(1, 2, 3, 4, 5, 6, 7, 8)
        end

        test
        test
        test
    "#), @"[1, 2, 3, 4, 5, 6, 7, 8]");
}

// Repro for the production "Failed to get_opnd(vN)" panic
// (PriceRs::PricingService#build_rust_adjustment_from_row, introduced by #17186).
//
// A regular send to a C method with 8 fixed args is reduced to a CCallWithFrame
// with recv + 8 = 9 operands, which exceeds C_ARG_OPNDS.len() (6 on x86_64, 8 on
// arm64). gen_insn bails with `return Err(*state)`; the caller emits a side exit and
// `break`s out of the block. But the call's *result* is stored in a local and used in
// a *later* basic block (the `if` arm here). Because codegen bailed before assigning
// a LIR operand to the result, compiling that later block calls get_opnd(result) on a
// None entry and panics. The existing `test_invokesuper_to_cfunc_with_too_many_args_exits`
// does not catch this because there the call result is the method's tail value and is
// not referenced past the bailed block.
//
// NOTE: This currently ABORTS with `Failed to get_opnd(vN)` (the bug). The snapshot
// below is the expected behavior once the backend exits cleanly: `flag` is true so
// `test` returns the cfunc's result, the array [1, 2, 3, 4, 5, 6, 7, 8].
#[test]
fn test_ccall_with_frame_too_many_args_result_used_in_later_block() {
    unsafe extern "C" fn test_eight_args(
        _self: VALUE,
        a: VALUE,
        b: VALUE,
        c: VALUE,
        d: VALUE,
        e: VALUE,
        f: VALUE,
        g: VALUE,
        h: VALUE,
    ) -> VALUE {
        unsafe { rb_ary_new_from_args(8, a, b, c, d, e, f, g, h) }
    }

    with_rubyvm(|| {
        let klass = define_class("ZJITEightArgs", unsafe { rb_cObject });
        unsafe {
            rb_define_method(
                klass,
                c"eight".as_ptr(),
                Some(std::mem::transmute::<
                    unsafe extern "C" fn(VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE) -> VALUE,
                    unsafe extern "C" fn(VALUE) -> VALUE,
                >(test_eight_args)),
                8,
            );
        }
    });

    assert_snapshot!(assert_compiles_allowing_exits(r#"
        def test(obj, flag)
          priceable = obj.eight(1, 2, 3, 4, 5, 6, 7, 8)
          if flag
            priceable
          else
            nil
          end
        end

        obj = ZJITEightArgs.new
        test(obj, true)  # profile receiver class
        test(obj, true)  # compile -> currently panics: Failed to get_opnd(vN)
        test(obj, true)
    "#), @"[1, 2, 3, 4, 5, 6, 7, 8]");
}

// Assert that a C method defined with rb_define_method() observes exactly the
// argument values that were passed at the Ruby level. The 7-arg method plus
// the receiver fills all 8 AAPCS64 argument registers on arm64 (x86_64 falls
// back to a dynamic send); the 10-arg method exceeds the argument registers
// on both platforms.
#[test]
fn test_cfunc_asserts_argument_values() {
    unsafe extern "C" fn assert_seven_args(
        _self: VALUE,
        a: VALUE,
        b: VALUE,
        c: VALUE,
        d: VALUE,
        e: VALUE,
        f: VALUE,
        g: VALUE,
    ) -> VALUE {
        assert_eq!(a, VALUE::fixnum_from_usize(1));
        assert_eq!(b, VALUE::fixnum_from_usize(2));
        assert_eq!(c, VALUE::fixnum_from_usize(3));
        assert_eq!(d, VALUE::fixnum_from_usize(4));
        assert_eq!(e, VALUE::fixnum_from_usize(5));
        assert_eq!(f, VALUE::fixnum_from_usize(6));
        assert_eq!(g, VALUE::fixnum_from_usize(7));
        Qtrue
    }

    unsafe extern "C" fn assert_ten_args(
        _self: VALUE,
        a: VALUE,
        b: VALUE,
        c: VALUE,
        d: VALUE,
        e: VALUE,
        f: VALUE,
        g: VALUE,
        h: VALUE,
        i: VALUE,
        j: VALUE,
    ) -> VALUE {
        assert_eq!(a, VALUE::fixnum_from_usize(1));
        assert_eq!(b, VALUE::fixnum_from_usize(2));
        assert_eq!(c, VALUE::fixnum_from_usize(3));
        assert_eq!(d, VALUE::fixnum_from_usize(4));
        assert_eq!(e, VALUE::fixnum_from_usize(5));
        assert_eq!(f, VALUE::fixnum_from_usize(6));
        assert_eq!(g, VALUE::fixnum_from_usize(7));
        assert_eq!(h, VALUE::fixnum_from_usize(8));
        assert_eq!(i, VALUE::fixnum_from_usize(9));
        assert_eq!(j, VALUE::fixnum_from_usize(10));
        Qtrue
    }

    with_rubyvm(|| {
        let klass = define_class("ZJITArgValues", unsafe { rb_cObject });
        unsafe {
            rb_define_method(
                klass,
                c"seven".as_ptr(),
                Some(std::mem::transmute::<
                    unsafe extern "C" fn(VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE) -> VALUE,
                    unsafe extern "C" fn(VALUE) -> VALUE,
                >(assert_seven_args)),
                7,
            );
            rb_define_method(
                klass,
                c"ten".as_ptr(),
                Some(std::mem::transmute::<
                    unsafe extern "C" fn(VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE, VALUE) -> VALUE,
                    unsafe extern "C" fn(VALUE) -> VALUE,
                >(assert_ten_args)),
                10,
            );
        }
    });

    assert_snapshot!(assert_compiles_allowing_exits(r#"
        def test(obj)
          [obj.seven(1, 2, 3, 4, 5, 6, 7), obj.ten(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
        end

        obj = ZJITArgValues.new
        test(obj)
        test(obj)
        test(obj)
    "#), @"[true, true]");
}

#[test]
fn test_string_new_preserves_string_arg() {
    assert_snapshot!(inspect(r#"
        def test
          str = "hello"
          String.new(str)
          :ok
        end

        test
        test
    "#), @":ok");
}

#[test]
fn test_invokesuper_multilevel() {
    assert_snapshot!(inspect(r#"
        class A
          def foo
            "A"
          end
        end

        class B < A
          def foo
            ["B", super]
          end
        end

        class C < B
          def foo
            ["C", super]
          end
        end

        def test
          C.new.foo
        end

        test  # profile invokesuper
        test  # compile + run compiled code
    "#), @r#"["C", ["B", "A"]]"#);
}

#[test]
fn test_invokesuper_forwards_block_implicitly() {
    assert_snapshot!(inspect(r#"
        class A
          def foo
            block_given? ? yield : "no_block"
          end
        end

        class B < A
          def foo
            ["B", super]  # should forward the block from caller
          end
        end

        def test
          B.new.foo { "forwarded_block" }
        end

        test  # profile invokesuper
        test  # compile + run compiled code
    "#), @r#"["B", "forwarded_block"]"#);
}

#[test]
fn test_invokesuper_forwards_block_implicitly_with_args() {
    assert_snapshot!(inspect(r#"
        class A
          def foo(x)
            [x, (block_given? ? yield : "no_block")]
          end
        end

        class B < A
          def foo(x)
            ["B", super(x)]  # explicit args, but block should still be forwarded
          end
        end

        def test
          B.new.foo("arg_value") { "forwarded" }
        end

        test  # profile
        test  # compile + run compiled code
    "#), @r#"["B", ["arg_value", "forwarded"]]"#);
}

#[test]
fn test_invokesuper_forwards_block_implicitly_no_block_given() {
    assert_snapshot!(inspect(r#"
        class A
          def foo
            block_given? ? yield : "no_block"
          end
        end

        class B < A
          def foo
            ["B", super]  # no block given by caller
          end
        end

        def test
          B.new.foo  # called without a block
        end

        test  # profile
        test  # compile + run compiled code
    "#), @r#"["B", "no_block"]"#);
}

#[test]
fn test_invokesuper_forwards_block_implicitly_multilevel() {
    assert_snapshot!(inspect(r#"
        class A
          def foo
            block_given? ? yield : "no_block"
          end
        end

        class B < A
          def foo
            ["B", super]  # forwards block to A
          end
        end

        class C < B
          def foo
            ["C", super]  # forwards block to B, which forwards to A
          end
        end

        def test
          C.new.foo { "deep_block" }
        end

        test  # profile
        test  # compile + run compiled code
    "#), @r#"["C", ["B", "deep_block"]]"#);
}

#[test]
fn test_invokesuper_forwards_block_param() {
    assert_snapshot!(inspect(r#"
        class A
          def foo
            block_given? ? yield : "no_block"
          end
        end

        class B < A
          def foo(&block)
            ["B", super]  # should forward &block implicitly
          end
        end

        def test
          B.new.foo { "block_param_forwarded" }
        end

        test  # profile
        test  # compile + run compiled code
    "#), @r#"["B", "block_param_forwarded"]"#);
}

#[test]
fn test_invokesuper_with_blockarg() {
    assert_snapshot!(inspect(r#"
        class A
          def foo
            block_given? ? yield : "no block"
          end
        end

        class B < A
          def foo(&blk)
            other_block = proc { "different block" }
            ["B", super(&other_block)]
          end
        end

        def test
          B.new.foo { "passed block" }
        end

        test  # profile
        test  # compile + run compiled code
    "#), @r#"["B", "different block"]"#);
}

#[test]
fn test_invokesuper_with_symbol_to_proc() {
    assert_snapshot!(inspect(r#"
        class A
          def foo(items, &blk)
            items.map(&blk)
          end
        end

        class B < A
          def foo(items)
            ["B", super(items, &:succ)]
          end
        end

        def test
          B.new.foo([2, 4, 6])
        end

        test  # profile
        test  # compile + run compiled code
    "#), @r#"["B", [3, 5, 7]]"#);
}

#[test]
fn test_invokesuper_with_splat() {
    assert_snapshot!(inspect(r#"
        class A
          def foo(a, b, c)
            a + b + c
          end
        end

        class B < A
          def foo(*args)
            ["B", super(*args)]
          end
        end

        def test
          B.new.foo(1, 2, 3)
        end

        test  # profile
        test  # compile + run compiled code
    "#), @r#"["B", 6]"#);
}

#[test]
fn test_invokesuper_with_kwargs() {
    assert_snapshot!(inspect(r#"
        class A
          def foo(x:, y:)
            "x=#{x}, y=#{y}"
          end
        end

        class B < A
          def foo(x:, y:)
            ["B", super(x: x, y: y)]
          end
        end

        def test
          B.new.foo(x: 1, y: 2)
        end

        test  # profile
        test  # compile + run compiled code
    "#), @r#"["B", "x=1, y=2"]"#);
}

#[test]
fn test_invokesuper_with_kw_splat() {
    assert_snapshot!(inspect(r#"
        class A
          def foo(x:, y:)
            "x=#{x}, y=#{y}"
          end
        end

        class B < A
          def foo(**kwargs)
            ["B", super(**kwargs)]
          end
        end

        def test
          B.new.foo(x: 1, y: 2)
        end

        test  # profile
        test  # compile + run compiled code
    "#), @r#"["B", "x=1, y=2"]"#);
}

#[test]
fn test_invokesuper_with_include() {
    assert_snapshot!(inspect(r#"
        class A
          def foo
            "A"
          end
        end

        class B < A
          def foo
            ["B", super]
          end
        end

        def test
          B.new.foo
        end

        test  # profile invokesuper (super -> A#foo)
        test  # compile with super -> A#foo

        # Now include a module in B that defines foo - super should go to M#foo instead
        module M
          def foo
            "M"
          end
        end
        B.include(M)

        test  # should call M#foo, not A#foo
    "#), @r#"["B", "M"]"#);
}

#[test]
fn test_invokesuper_with_prepend() {
    assert_snapshot!(inspect(r#"
        class A
          def foo
            "A"
          end
        end

        class B < A
          def foo
            ["B", super]
          end
        end

        def test
          B.new.foo
        end

        test  # profile invokesuper (super -> A#foo)
        test  # compile with super -> A#foo

        # Now prepend a module that defines foo - super should go to M#foo instead
        module M
          def foo
            "M"
          end
        end
        A.prepend(M)

        test  # should call M#foo, not A#foo
    "#), @r#"["B", "M"]"#);
}

/// A monomorphic `super` should stay specialized: no side exits, no dynamic dispatch.
#[test]
fn test_invokesuper_monomorphic_does_not_exit() {
    eval("
        class MonoSuperA
          def foo(x) = x + 1
        end
        class MonoSuperB < MonoSuperA
          def foo(x) = super(x) * 10
        end
        def test = MonoSuperB.new.foo(1)
        test
        test
    ");
    assert_snapshot!(assert_compiles("[test, test, test]"), @"[20, 20, 20]");
}

/// The VM replaces a frame's `ep[VM_ENV_DATA_INDEX_ME_CREF]` with an `imemo_svar` wrapping the
/// frame's method entry as soon as the frame touches a special variable, which a regexp match
/// does. The `super` guard has to read through the svar; otherwise it misses on every single
/// call and the site side-exits forever.
#[test]
fn test_invokesuper_after_regexp_match_does_not_exit() {
    eval(r#"
        class SvarSuperA
          def foo(x) = x + 1
        end
        class SvarSuperB < SvarSuperA
          def foo(x)
            x.to_s =~ /(\d+)/
            [$1, super(x)]
          end
        end
        def test = SvarSuperB.new.foo(3)
        test
        test
    "#);
    assert_snapshot!(assert_compiles("[test, test, test]"), @r#"[["3", 4], ["3", 4], ["3", 4]]"#);
}

/// A `super` inside a module body resolves through a different complemented CME for each
/// including class, so no single CME can be guarded. The site must converge on dispatching
/// `super` dynamically rather than side-exiting once per call.
#[test]
fn test_invokesuper_polymorphic_converges_without_repeated_exits() {
    let exits = || crate::state::ZJITState::get_counters().exit_guard_super_method_entry;
    // `run` keeps the calls in one ISEQ so that the second phase does not compile anything new.
    assert_snapshot!(inspect(r#"
        class PolySuperBase1
          def foo(x) = [:base1, x]
        end
        class PolySuperBase2
          def foo(x) = [:base2, x]
        end
        module PolySuperM
          def foo(x) = super(x) << :m
        end
        class PolySuperA < PolySuperBase1
          include PolySuperM
        end
        class PolySuperB < PolySuperBase2
          include PolySuperM
        end
        def test(o) = o.foo(1)
        def run(a, b, n) = n.times { test(a); test(b) }

        $poly_super_a = PolySuperA.new
        $poly_super_b = PolySuperB.new
        run($poly_super_a, $poly_super_b, 200)
        [test($poly_super_a), test($poly_super_b)]
    "#), @"[[:base1, 1, :m], [:base2, 1, :m]]");

    // The site has converged, so exits are now bounded by the recompile budget (a handful per
    // compiled ISEQ) rather than one per call: without the dynamic fallback, all 1000 super
    // calls below would exit.
    let before = exits();
    assert_snapshot!(inspect("
        run($poly_super_a, $poly_super_b, 500)
        [test($poly_super_a), test($poly_super_b)]
    "), @"[[:base1, 1, :m], [:base2, 1, :m]]");
    let delta = exits() - before;
    assert!(delta < 25, "guard_super_method_entry exits still scale with call count: {delta} exits over 1000 super calls");
}

/// A `super` in a method that is always called with a block can never pass the specialized
/// call's block-handler guard. The exits must stop once the site gives up and dispatches
/// `super` dynamically.
#[test]
fn test_invokesuper_always_with_block_converges_without_repeated_exits() {
    let exits = || crate::state::ZJITState::get_counters().exit_unhandled_block_arg;
    assert_snapshot!(inspect(r#"
        class BlockSuperA
          def foo(x) = x + 1
        end
        class BlockSuperB < BlockSuperA
          def foo(x) = super(x) * 10
        end
        def block_super_test(o) = o.foo(1) { }
        def block_super_run(o, n) = n.times { block_super_test(o) }
        $block_super_o = BlockSuperB.new
        block_super_run($block_super_o, 500)
        block_super_test($block_super_o)
    "#), @"20");

    // Exits are bounded by the recompile budget now, not one per call: without the dynamic
    // fallback all 1000 calls below would exit.
    let before = exits();
    assert_snapshot!(inspect("
        block_super_run($block_super_o, 1000)
        block_super_test($block_super_o)
    "), @"20");
    let delta = exits() - before;
    assert!(delta < 25, "unhandled_block_arg exits still scale with call count: {delta} exits over 1000 super calls");
}

/// Redefining the target of a specialized `super` after it is compiled must take effect.
#[test]
fn test_invokesuper_with_target_redefined_after_compile() {
    assert_snapshot!(inspect(r#"
        class RedefSuperA
          def foo = "a1"
        end
        class RedefSuperB < RedefSuperA
          def foo = ["b", super]
        end
        def test = RedefSuperB.new.foo
        before = [test, test, test]
        class RedefSuperA
          def foo = "a2"
        end
        [before.last, test]
    "#), @r#"[["b", "a1"], ["b", "a2"]]"#);
}

/// A `super` reached through a prepended module, an included module and a singleton class all
/// at once runs under a different frame method entry each time, so the site dispatches on the
/// entry and each arm resolves `super` from its own defining class.
#[test]
fn test_invokesuper_chain_over_prepend_include_and_singleton() {
    assert_snapshot!(inspect(r#"
        module ChainSuperM
          def tag(x) = super(x) + [:m]
        end
        class ChainSuperBase
          def tag(x) = [x]
        end
        class ChainSuperPrepend < ChainSuperBase
          prepend ChainSuperM
        end
        class ChainSuperInclude
          include ChainSuperM
          def self.new_with_singleton
            o = allocate
            def o.extra = :sing
            o
          end
        end
        class ChainSuperIncludeBase
          def tag(x) = [x, :incbase]
        end
        class ChainSuperInclude2 < ChainSuperIncludeBase
          include ChainSuperM
        end
        $chain_super = [ChainSuperPrepend.new, ChainSuperInclude2.new, ChainSuperInclude2.new.tap { |o| def o.z = 1 }]
        def chain_super_run(n) = n.times { $chain_super.each { |o| o.tag(1) } }
        chain_super_run(300)
        $chain_super.map { |o| o.tag(2) }
    "#), @"[[2, :m], [2, :incbase, :m], [2, :incbase, :m]]");
    assert!(crate::state::ZJITState::get_counters().super_chain_sites > 0,
        "the super site never got a method-entry dispatch chain");
}

/// Redefining the target of one arm of a `super` dispatch chain has to take effect, the same way
/// it does for a single guarded `super`.
#[test]
fn test_invokesuper_chain_with_target_redefined_after_compile() {
    assert_snapshot!(inspect(r#"
        module ChainRedefM
          def val = ["m", super]
        end
        class ChainRedefA
          def val = "a1"
        end
        class ChainRedefB
          def val = "b1"
        end
        class ChainRedefSubA < ChainRedefA
          prepend ChainRedefM
        end
        class ChainRedefSubB < ChainRedefB
          prepend ChainRedefM
        end
        $chain_redef = [ChainRedefSubA.new, ChainRedefSubB.new]
        def chain_redef_run(n) = n.times { $chain_redef.each(&:val) }
        chain_redef_run(300)
        before = $chain_redef.map(&:val)
        class ChainRedefA
          def val = "a2"
        end
        [before, $chain_redef.map(&:val)]
    "#), @r#"[[["m", "a1"], ["m", "b1"]], [["m", "a2"], ["m", "b1"]]]"#);
}

/// A zsuper (`super` with no argument list) forwards the caller's arguments, which the
/// specialized call cannot reproduce, so it keeps a dynamic dispatch -- including inside a
/// dispatch chain's arms, where the arm still has to produce the right answer.
#[test]
fn test_zsuper_stays_dynamic_but_correct() {
    assert_snapshot!(inspect(r#"
        module ZSuperM
          def calc(a, b) = super * 10
        end
        class ZSuperBase1
          def calc(a, b) = a + b
        end
        class ZSuperBase2
          def calc(a, b) = a * b
        end
        class ZSuperA < ZSuperBase1
          prepend ZSuperM
        end
        class ZSuperB < ZSuperBase2
          prepend ZSuperM
        end
        $zsuper = [ZSuperA.new, ZSuperB.new]
        def zsuper_run(n) = n.times { $zsuper.each { |o| o.calc(2, 3) } }
        zsuper_run(300)
        $zsuper.map { |o| o.calc(2, 3) }
    "#), @"[50, 60]");
}

/// A protected method called with an explicit receiver from an instance of a subclass is
/// permitted, and one called from an unrelated class still raises.
#[test]
fn test_protected_call_permitted_from_subclass_and_refused_elsewhere() {
    assert_snapshot!(inspect(r#"
        class ProtBase
          def initialize(v) = @v = v
          def combine(other) = secret + other.secret
          protected
          def secret = @v
        end
        class ProtSub < ProtBase; end
        class ProtStranger
          def peek(o) = o.secret
        end
        $prot_a = ProtBase.new(1)
        $prot_b = ProtSub.new(2)
        def prot_run(n) = n.times { $prot_a.combine($prot_b); $prot_b.combine($prot_a) }
        prot_run(300)
        stranger = (ProtStranger.new.peek($prot_a) rescue $!.class)
        [$prot_a.combine($prot_b), $prot_b.combine($prot_a), stranger]
    "#), @"[3, 3, NoMethodError]");
    assert!(crate::state::ZJITState::get_counters().send_protected_guard_sites > 0,
        "the protected call never got a caller-self guard");
}

/// A refinement's protected method is defined in the refinement's ICLASS, which is not a class
/// the caller's `self` can be checked against, so the call keeps its dynamic dispatch -- and
/// keeps refusing callers the interpreter would refuse.
#[test]
fn test_protected_call_under_refinement() {
    assert_snapshot!(inspect(r#"
        class RefProt
          def initialize(v) = @v = v
        end
        module RefProtM
          refine RefProt do
            def combine(other) = secret + other.secret
            protected def secret = @v * 2
          end
        end
        using RefProtM
        $ref_prot_a = RefProt.new(1)
        $ref_prot_b = RefProt.new(2)
        def ref_prot_run(n) = n.times { $ref_prot_a.combine($ref_prot_b) }
        ref_prot_run(300)
        [$ref_prot_a.combine($ref_prot_b), ($ref_prot_a.secret rescue $!.class)]
    "#), @"[6, NoMethodError]");
}

/// `self.foo = x` is a legal call to a private writer, and the bytecode marks it FCALL, so it
/// specializes with no visibility guard at all.
#[test]
fn test_private_writer_through_self_receiver_compiles() {
    assert_snapshot!(inspect(r#"
        class PrivWriter
          def set(x)
            self.value = x
            self.value
          end
          private
          attr_accessor :value
        end
        $priv_writer = PrivWriter.new
        def priv_writer_run(n) = n.times { |i| $priv_writer.set(i) }
        priv_writer_run(300)
        $priv_writer.set(7)
    "#), @"7");
}

/// A private method reached through a receiver that is not literally `self` is still refused,
/// even when the receiver happens to be the same object at runtime.
#[test]
fn test_private_call_with_non_self_receiver_still_raises() {
    assert_snapshot!(inspect(r#"
        class PrivRecv
          def call_through(o) = o.hidden rescue $!.class
          private
          def hidden = :nope
        end
        $priv_recv = PrivRecv.new
        def priv_recv_run(n) = n.times { $priv_recv.call_through($priv_recv) }
        priv_recv_run(300)
        $priv_recv.call_through($priv_recv)
    "#), @"NoMethodError");
}

/// `respond_to?` reports visibility, not callability, and nothing here changes that.
#[test]
fn test_respond_to_visibility_unaffected() {
    assert_snapshot!(inspect(r#"
        class RespondVis
          def pub = 1
          protected def prot = 2
          private def priv = 3
        end
        $respond_vis = RespondVis.new
        def respond_vis_test(o) = [o.respond_to?(:pub), o.respond_to?(:prot), o.respond_to?(:priv),
                                   o.respond_to?(:prot, true), o.respond_to?(:priv, true)]
        def respond_vis_run(n) = n.times { respond_vis_test($respond_vis) }
        respond_vis_run(300)
        respond_vis_test($respond_vis)
    "#), @"[true, false, false, true, true]");
}

#[test]
fn test_invokesuper_with_keyword_args() {
    assert_snapshot!(inspect(r#"
        class A
          def foo(attributes = {})
            @attributes = attributes
          end
        end

        class B < A
          def foo(content = '')
            super(content: content)
          end
        end

        def test
          B.new.foo("image data")
        end

        test
        test
    "#), @r#"{content: "image data"}"#);
}

#[test]
fn test_invokesuper_with_optional_keyword_args() {
    assert_snapshot!(inspect("
        class Parent
          def foo(a, b: 2, c: 3) = [a, b, c]
        end

        class Child < Parent
          def foo(a) = super(a)
        end

        def test = Child.new.foo(1)

        test
        test
    "), @"[1, 2, 3]");
}

#[test]
fn test_invokesuperforward() {
    assert_snapshot!(inspect("
        class A
          def foo(a,b,c) = [a,b,c]
        end

        class B < A
          def foo(...) = super
        end

        def test
          B.new.foo(1, 2, 3)
        end

        test
        test
    "), @"[1, 2, 3]");
}

#[test]
fn test_invokesuperforward_with_args_kwargs_and_block() {
    assert_snapshot!(inspect("
        class A
          def foo(*args, **kwargs, &block)
            [args, kwargs, block&.call]
          end
        end

        class B < A
          def foo(...) = super
        end

        def test
          B.new.foo(1, 2, x: 3) { 4 }
        end

        test
        test
    "), @"[[1, 2], {x: 3}, 4]");
}

#[test]
fn test_send_with_non_constant_keyword_default() {
    assert_snapshot!(inspect("
        def dbl(x = 1) = x * 2

        def foo(a: dbl, b: dbl(2), c: dbl(2 ** 3))
          [a, b, c]
        end

        def test
          [
            foo,
            foo(a: 10),
            foo(b: 20),
            foo(c: 30),
            foo(a: 10, b: 20, c: 30)
          ]
        end

        test
        test
    "), @"[[2, 4, 16], [10, 4, 16], [2, 20, 16], [2, 4, 30], [10, 20, 30]]");
}

#[test]
fn test_send_with_non_constant_keyword_default_not_evaluated_when_provided() {
    assert_snapshot!(inspect("
        def foo(a: raise, b: raise, c: raise)
          [a, b, c]
        end

        def test
          foo(a: 1, b: 2, c: 3)
        end

        test
        test
    "), @"[1, 2, 3]");
}

#[test]
fn test_send_with_non_constant_keyword_default_evaluated_when_not_provided() {
    assert_snapshot!(inspect(r#"
        def raise_a = raise "a"
        def raise_b = raise "b"
        def raise_c = raise "c"

        def foo(a: raise_a, b: raise_b, c: raise_c)
          [a, b, c]
        end

        def test_a
          foo(b: 2, c: 3)
        rescue RuntimeError => e
          e.message
        end

        def test_b
          foo(a: 1, c: 3)
        rescue RuntimeError => e
          e.message
        end

        def test_c
          foo(a: 1, b: 2)
        rescue RuntimeError => e
          e.message
        end

        def test
          [test_a, test_b, test_c]
        end

        test
        test
    "#), @r#"["a", "b", "c"]"#);
}

#[test]
fn test_send_with_non_constant_keyword_default_jit_to_jit() {
    assert_snapshot!(inspect("
        def make_default(x) = x * 2

        def callee(a: make_default(1), b: make_default(2), c: make_default(3))
          [a, b, c]
        end

        def caller_method
          callee
        end

        # Warm up callee first so it gets JITted
        callee
        callee

        # Now warm up caller - this creates JIT-to-JIT call
        caller_method
        caller_method
    "), @"[2, 4, 6]");
}

#[test]
fn test_send_with_non_constant_keyword_default_side_exit() {
    assert_snapshot!(inspect("
        def make_b = 2

        def callee(a: 1, b: make_b, c: 3)
          x = binding.local_variable_get(:a)
          y = binding.local_variable_get(:b)
          z = binding.local_variable_get(:c)
          [x, y, z]
        end

        def test
          callee(a: 10, c: 30)
        end

        test
        test
    "), @"[10, 2, 30]");
}

#[test]
fn test_send_with_non_constant_keyword_default_evaluation_order() {
    assert_snapshot!(inspect(r#"
        def log(x)
          $order << x
          x
        end

        def foo(a: log("a"), b: log("b"), c: log("c"))
          [a, b, c]
        end

        def test
          results = []

          $order = []
          foo
          results << $order.dup

          $order = []
          foo(a: "A")
          results << $order.dup

          $order = []
          foo(b: "B")
          results << $order.dup

          $order = []
          foo(c: "C")
          results << $order.dup

          results
        end

        test
        test
    "#), @r#"[["a", "b", "c"], ["b", "c"], ["a", "c"], ["a", "b"]]"#);
}

#[test]
fn test_send_with_too_many_non_constant_keyword_defaults() {
    assert_snapshot!(inspect("
        def many_kwargs( k1: 1, k2: 2, k3: 3, k4: 4, k5: 5, k6: 6, k7: 7, k8: 8, k9: 9, k10: 10, k11: 11, k12: 12, k13: 13, k14: 14, k15: 15, k16: 16, k17: 17, k18: 18, k19: 19, k20: 20, k21: 21, k22: 22, k23: 23, k24: 24, k25: 25, k26: 26, k27: 27, k28: 28, k29: 29, k30: 30, k31: 31, k32: 32, k33: 33, k34: k33 + 1) = k1 + k34
        def t = many_kwargs
        t
        t
    "), @"35");
}

#[test]
fn test_invokebuiltin_delegate() {
    assert_snapshot!(inspect("
        def test = [].clone(freeze: true)
        r = test
        r2 = test
        [r2, r2.frozen?]
    "), @"[[], true]");
}

#[test]
fn test_invokebuiltin_many_args() {
    // Time#initialize calls the time_init_args builtin with 7 arguments
    // (9 C arguments including ec and self), which don't fit in argument
    // registers and exercise stack arguments in CCall.
    assert_snapshot!(inspect("
        def test = Time.new(1992, 9, 23, 23, 0, 0, 3600)
        test
        test
    "), @"1992-09-23 23:00:00 +0100");
}

#[test]
fn test_kernel_integer_exception_false_returns_nil() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def test = Integer('x', exception: false) ? 1 : 0;
            test
            test
        "), @"0");
    });
}

#[test]
fn test_kernel_float_exception_false_returns_nil() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def test = Float('x', exception: false) ? 1 : 0;
            test
            test
        "), @"0");
    });
}

#[test]
fn test_opt_plus_const() {
    assert_snapshot!(inspect("
        def test = 1 + 2
        test # profile opt_plus
        test
    "), @"3");
}

#[test]
fn test_opt_plus_fixnum() {
    assert_snapshot!(inspect("
        def test(a, b) = a + b
        test(0, 1) # profile opt_plus
        test(1, 2)
    "), @"3");
}

#[test]
fn test_opt_plus_chain() {
    assert_snapshot!(inspect("
        def test(a, b, c) = a + b + c
        test(0, 1, 2) # profile opt_plus
        test(1, 2, 3)
    "), @"6");
}

#[test]
fn test_opt_plus_left_imm() {
    assert_snapshot!(inspect("
        def test(a) = 1 + a
        test(1) # profile opt_plus
        test(2)
    "), @"3");
}

#[test]
fn test_opt_plus_type_guard_exit() {
    assert_snapshot!(inspect("
        def test(a) = 1 + a
        test(1) # profile opt_plus
        [test(2), test(2.0)]
    "), @"[3, 3.0]");
}

#[test]
fn test_opt_plus_type_guard_exit_with_locals() {
    assert_snapshot!(inspect("
        def test(a)
          local = 3
          1 + a + local
        end
        test(1) # profile opt_plus
        [test(2), test(2.0)]
    "), @"[6, 6.0]");
}

#[test]
fn test_opt_plus_type_guard_nested_exit() {
    assert_snapshot!(inspect("
        def side_exit(n) = 1 + n
        def jit_frame(n) = 1 + side_exit(n)
        def entry(n) = jit_frame(n)
        entry(2) # profile send
        [entry(2), entry(2.0)]
    "), @"[4, 4.0]");
}

#[test]
fn test_opt_plus_type_guard_nested_exit_with_locals() {
    assert_snapshot!(inspect("
        def side_exit(n)
          local = 2
          1 + n + local
        end
        def jit_frame(n)
          local = 3
          1 + side_exit(n) + local
        end
        def entry(n) = jit_frame(n)
        entry(2) # profile send
        [entry(2), entry(2.0)]
    "), @"[9, 9.0]");
}

#[test]
fn test_opt_minus() {
    assert_snapshot!(inspect("
        def test(a, b) = a - b
        test(2, 1) # profile opt_minus
        test(6, 4)
    "), @"2");
}

#[test]
fn test_opt_mult() {
    assert_snapshot!(inspect("
        def test(a, b) = a * b
        test(1, 2) # profile opt_mult
        test(2, 3)
    "), @"6");
}

#[test]
fn test_opt_mult_overflow() {
    assert_snapshot!(inspect("
        def test(a, b)
          a * b
        end
        test(1, 1) # profile opt_mult

        r1 = test(2, 3)
        r2 = test(2, -3)
        r3 = test(2 << 40, 2 << 41)
        r4 = test(2 << 40, -2 << 41)
        r5 = test(1 << 62, 1 << 62)

        [r1, r2, r3, r4, r5]
    "), @"[6, -6, 9671406556917033397649408, -9671406556917033397649408, 21267647932558653966460912964485513216]");
}

#[test]
fn test_opt_plus_overflow() {
    assert_snapshot!(inspect("
        def test(a, b)
          a + b
        end
        test(1, 2) # profile opt_plus

        r1 = test(2, 3)
        r2 = test(4611686018427387903, 1)    # FIXNUM_MAX + 1 overflows
        r3 = test(-4611686018427387904, -1)  # FIXNUM_MIN - 1 overflows

        [r1, r2, r3]
    "), @"[5, 4611686018427387904, -4611686018427387905]");
}

#[test]
fn test_opt_minus_overflow() {
    assert_snapshot!(inspect("
        def test(a, b)
          a - b
        end
        test(6, 4) # profile opt_minus

        r1 = test(6, 4)
        r2 = test(4611686018427387903, -1)   # FIXNUM_MAX - (-1) overflows
        r3 = test(-4611686018427387904, 1)   # FIXNUM_MIN - 1 overflows

        [r1, r2, r3]
    "), @"[2, 4611686018427387904, -4611686018427387905]");
}

#[test]
fn test_fixnum_lshift() {
    assert_snapshot!(inspect("
        def test(a) = a << 3
        test(1) # profile opt_ltlt

        [test(5), test(0), test(-5)]
    "), @"[40, 0, -40]");
}

#[test]
fn test_fixnum_lshift_overflow() {
    assert_snapshot!(inspect("
        def test(a) = a << 3
        test(1) # profile opt_ltlt

        r1 = test(1 << 60)
        r2 = test(-(1 << 60))

        [r1, r2]
    "), @"[9223372036854775808, -9223372036854775808]");
}

#[test]
fn test_opt_eq() {
    eval("
        def test(a, b) = a == b
        test(0, 2) # profile opt_eq
    ");
    assert_contains_opcode("test", YARVINSN_opt_eq);
    assert_snapshot!(assert_compiles("[test(1, 1), test(0, 1)]"), @"[true, false]");
}

#[test]
fn test_opt_eq_with_minus_one() {
    eval("
        def test(a) = a == -1
        test(1) # profile opt_eq
    ");
    assert_contains_opcode("test", YARVINSN_opt_eq);
    assert_snapshot!(assert_compiles("[test(0), test(-1)]"), @"[false, true]");
}

#[test]
fn test_opt_neq_dynamic() {
    eval("
        def test(a, b) = a != b
        test(0, 2) # profile opt_neq
    ");
    assert_contains_opcode("test", YARVINSN_opt_neq);
    assert_snapshot!(assert_compiles("[test(1, 1), test(0, 1)]"), @"[false, true]");
}

#[test]
fn test_opt_neq_fixnum() {
    assert_snapshot!(inspect("
        def test(a, b) = a != b
        test(0, 2) # profile opt_neq
        [test(1, 1), test(0, 1)]
    "), @"[false, true]");
}

#[test]
fn test_opt_neq_string_nil() {
    assert_snapshot!(inspect(r#"
        def test(str) = str != nil
        test("x") # profile opt_neq
        [test("x"), test(nil)]
    "#), @"[true, false]");
}

#[test]
fn test_opt_neq_string_same_operand() {
    assert_snapshot!(inspect(r#"
        def test(s) = s != s
        test("x") # profile opt_neq
        [test("x"), test("y")]
    "#), @"[false, false]");
    assert_contains_opcode("test", YARVINSN_opt_neq);
}

#[test]
fn test_opt_neq_string_distinct_literals() {
    assert_snapshot!(inspect(r#"
        def test = "a" != "b"
        test # profile opt_neq
        [test, test]
    "#), @"[true, true]");
    assert_contains_opcode("test", YARVINSN_opt_neq);
}

#[test]
fn test_opt_neq_string_one_side_known_literal() {
    assert_snapshot!(inspect(r#"
        def test(s) = "a" != s
        test("a") # profile opt_neq
        [test("a"), test("b")]
    "#), @"[false, true]");
    assert_contains_opcode("test", YARVINSN_opt_neq);
}

#[test]
fn test_opt_neq_string_distinct_objects() {
    assert_snapshot!(inspect(r#"
        def test(s, t) = s != t
        test("x", "x") # profile opt_neq
        [test("x", "x"), test("x", "y")]
    "#), @"[false, true]");
    assert_contains_opcode("test", YARVINSN_opt_neq);
}

#[test]
fn test_opt_eq_string_same_operand() {
    assert_snapshot!(inspect(r#"
        def test(s) = s == s
        test("x") # profile opt_eq
        [test("x"), test("y")]
    "#), @"[true, true]");
    assert_contains_opcode("test", YARVINSN_opt_eq);
}

#[test]
fn test_opt_eq_string_distinct_literals() {
    assert_snapshot!(inspect(r#"
        def test = "a" == "b"
        test # profile opt_eq
        [test, test]
    "#), @"[false, false]");
    assert_contains_opcode("test", YARVINSN_opt_eq);
}

#[test]
fn test_opt_eq_string_one_side_known_literal() {
    assert_snapshot!(inspect(r#"
        def test(s) = "a" == s
        test("a") # profile opt_eq
        [test("a"), test("b")]
    "#), @"[true, false]");
    assert_contains_opcode("test", YARVINSN_opt_eq);
}

#[test]
fn test_opt_eq_string_distinct_objects() {
    assert_snapshot!(inspect(r#"
        def test(s, t) = s == t
        test("x", "x") # profile opt_eq
        [test("x", "x"), test("x", "y")]
    "#), @"[true, false]");
    assert_contains_opcode("test", YARVINSN_opt_eq);
}

#[test]
fn test_opt_eqq_string_same_operand() {
    assert_snapshot!(inspect(r#"
        def test(s) = s === s
        test("x") # profile opt_send_without_block
        [test("x"), test("y")]
    "#), @"[true, true]");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
}

#[test]
fn test_opt_lt() {
    eval("
        def test(a, b) = a < b
        test(2, 3) # profile opt_lt
    ");
    assert_contains_opcode("test", YARVINSN_opt_lt);
    assert_snapshot!(assert_compiles("[test(0, 1), test(0, 0), test(1, 0)]"), @"[true, false, false]");
}

#[test]
fn test_opt_lt_with_literal_lhs() {
    eval("
        def test(n) = 2 < n
        test(2) # profile opt_lt
    ");
    assert_contains_opcode("test", YARVINSN_opt_lt);
    assert_snapshot!(assert_compiles("[test(1), test(2), test(3)]"), @"[false, false, true]");
}

#[test]
fn test_opt_le() {
    eval("
        def test(a, b) = a <= b
        test(2, 3) # profile opt_le
    ");
    assert_contains_opcode("test", YARVINSN_opt_le);
    assert_snapshot!(assert_compiles("[test(0, 1), test(0, 0), test(1, 0)]"), @"[true, true, false]");
}

#[test]
fn test_opt_gt() {
    eval("
        def test(a, b) = a > b
        test(2, 3) # profile opt_gt
    ");
    assert_contains_opcode("test", YARVINSN_opt_gt);
    assert_snapshot!(assert_compiles("[test(0, 1), test(0, 0), test(1, 0)]"), @"[false, false, true]");
}

#[test]
fn test_opt_empty_p() {
    eval("
        def test(x) = x.empty?
    ");
    assert_contains_opcode("test", YARVINSN_opt_empty_p);
    assert_snapshot!(assert_compiles_allowing_exits("[test([1]), test(\"1\"), test({})]"), @"[false, false, true]");
}

#[test]
fn test_opt_succ() {
    eval("
        def test(obj) = obj.succ
    ");
    assert_contains_opcode("test", YARVINSN_opt_succ);
    assert_snapshot!(assert_compiles_allowing_exits(r#"[test(-1), test("A")]"#), @r#"[0, "B"]"#);
}

#[test]
fn test_opt_and() {
    eval("
        def test(x, y) = x & y
    ");
    assert_contains_opcode("test", YARVINSN_opt_and);
    assert_snapshot!(assert_compiles_allowing_exits("[test(0b1101, 3), test([3, 2, 1, 4], [8, 1, 2, 3])]"), @"[1, [3, 2, 1]]");
}

#[test]
fn test_opt_or() {
    eval("
        def test(x, y) = x | y
    ");
    assert_contains_opcode("test", YARVINSN_opt_or);
    assert_snapshot!(assert_compiles_allowing_exits("[test(0b1000, 3), test([3, 2, 1], [1, 2, 3])]"), @"[11, [3, 2, 1]]");
}

#[test]
fn test_fixnum_and() {
    eval("
        def test(a, b) = a & b
    ");
    assert_contains_opcode("test", YARVINSN_opt_and);
    assert_snapshot!(assert_compiles("
        [
                  test(5, 3),
                  test(0b011, 0b110),
                  test(-0b011, 0b110)
                ]
    "), @"[1, 2, 4]");
}

#[test]
fn test_fixnum_and_side_exit() {
    eval("
        def test(a, b) = a & b
    ");
    assert_contains_opcode("test", YARVINSN_opt_and);
    assert_snapshot!(assert_compiles_allowing_exits("
        [
                  test(2, 2),
                  test(0b011, 0b110),
                  test(true, false)
                ]
    "), @"[2, 2, false]");
}

#[test]
fn test_fixnum_or() {
    eval("
        def test(a, b) = a | b
    ");
    assert_contains_opcode("test", YARVINSN_opt_or);
    assert_snapshot!(assert_compiles("
        [
                  test(5, 3),
                  test(1, 2),
                  test(1, -4)
                ]
    "), @"[7, 3, -3]");
}

#[test]
fn test_fixnum_or_side_exit() {
    eval("
        def test(a, b) = a | b
    ");
    assert_contains_opcode("test", YARVINSN_opt_or);
    assert_snapshot!(assert_compiles_allowing_exits("
        [
                  test(1, 2),
                  test(2, 2),
                  test(true, false)
                ]
    "), @"[3, 2, true]");
}

#[test]
fn test_fixnum_xor() {
    assert_snapshot!(inspect("
        def test(a, b) = a ^ b
        [
          test(5, 3),
          test(-5, 3),
          test(1, 2)
        ]
    "), @"[6, -8, 3]");
}

#[test]
fn test_fixnum_xor_side_exit() {
    assert_snapshot!(inspect("
        def test(a, b) = a ^ b
        [
          test(5, 3),
          test(5, 3),
          test(true, false)
        ]
    "), @"[6, 6, true]");
}

#[test]
fn test_fixnum_mul() {
    eval("
        C = 3
        def test(n) = C * n
        test(4)
        test(4)
    ");
    assert_contains_opcode("test", YARVINSN_opt_mult);
    assert_snapshot!(assert_compiles("test(4)"), @"12");
}

#[test]
fn test_fixnum_div() {
    eval("
        C = 48
        def test(n) = C / n
        test(4)
    ");
    assert_contains_opcode("test", YARVINSN_opt_div);
    assert_snapshot!(assert_compiles("test(4)"), @"12");
}

#[test]
fn test_fixnum_floor() {
    eval("
        C = 3
        def test(n) = C / n
        test(4)
    ");
    assert_contains_opcode("test", YARVINSN_opt_div);
    assert_snapshot!(assert_compiles("test(4)"), @"0");
}

#[test]
fn test_fixnum_mod() {
    eval("
        def test(a, b) = a % b
        test(13, 4) # profile opt_mod
    ");
    assert_contains_opcode("test", YARVINSN_opt_mod);
    assert_snapshot!(assert_compiles("[test(13, 4), test(13, 13), test(5, 7)]"), @"[1, 0, 5]");
}

#[test]
fn test_fixnum_mod_negative() {
    eval("
        def test(a, b) = a % b
        test(7, 3) # profile opt_mod
    ");
    assert_contains_opcode("test", YARVINSN_opt_mod);
    assert_snapshot!(assert_compiles("[test(-7, 3), test(7, -3), test(-7, -3)]"), @"[2, -2, -1]");
}

#[test]
fn test_fixnum_mod_pow2_constant() {
    // Modulo by a positive power-of-two constant is strength-reduced to FixnumAnd
    eval("
        def test(a) = a % 8
        test(13) # profile opt_mod
    ");
    assert_contains_opcode("test", YARVINSN_opt_mod);
    assert_snapshot!(assert_compiles("[test(13), test(8), test(0), test(-1), test(-8), test(4611686018427387903), test(-4611686018427387904)]"), @"[5, 0, 0, 7, 0, 7, 0]");
}

#[test]
fn test_fixnum_mod_one_constant() {
    eval("
        def test(a) = a % 1
        test(13) # profile opt_mod
    ");
    assert_contains_opcode("test", YARVINSN_opt_mod);
    assert_snapshot!(assert_compiles("[test(13), test(-13)]"), @"[0, 0]");
}

#[test]
fn test_fixnum_mod_negative_pow2_constant() {
    // Only positive power-of-two divisors are strength-reduced
    eval("
        def test(a) = a % -8
        test(13) # profile opt_mod
    ");
    assert_contains_opcode("test", YARVINSN_opt_mod);
    assert_snapshot!(assert_compiles("[test(13), test(-13)]"), @"[-3, -5]");
}

#[test]
fn test_fixnum_div_pow2_constant() {
    // Division by a positive power-of-two constant is strength-reduced to FixnumRShift
    eval("
        def test(a) = a / 8
        test(13) # profile opt_div
    ");
    assert_contains_opcode("test", YARVINSN_opt_div);
    assert_snapshot!(assert_compiles("[test(13), test(-13), test(0), test(-1), test(4611686018427387903), test(-4611686018427387904)]"), @"[1, -2, 0, -1, 576460752303423487, -576460752303423488]");
}

#[test]
fn test_fixnum_div_negative_pow2_constant() {
    // Only positive power-of-two divisors are strength-reduced
    eval("
        def test(a) = a / -8
        test(13) # profile opt_div
    ");
    assert_contains_opcode("test", YARVINSN_opt_div);
    assert_snapshot!(assert_compiles("[test(13), test(-13)]"), @"[-2, 1]");
}

#[test]
fn test_fixnum_aref_constant_index() {
    eval("
        def test(a) = a[12]
        test(4096) # profile opt_aref
    ");
    assert_contains_opcode("test", YARVINSN_opt_aref);
    assert_snapshot!(assert_compiles("[test(4096), test(4095), test(0), test(-1), test(-4096)]"), @"[1, 0, 0, 1, 1]");
}

#[test]
fn test_fixnum_aref_constant_index_beyond_fixnum_width() {
    // An index beyond the fixnum width is not strength-reduced; FixnumAref handles it
    eval("
        def test(a) = a[100]
        test(1) # profile opt_aref
    ");
    assert_contains_opcode("test", YARVINSN_opt_aref);
    assert_snapshot!(assert_compiles("[test(1), test(-1), test(4611686018427387903), test(-4611686018427387904)]"), @"[0, 1, 0, 1]");
}

#[test]
fn test_fixnum_aref_constant_index_bignum_receiver() {
    // A Bignum receiver fails the Fixnum guard and side-exits to the correct result
    eval("
        def test(a) = a[1]
        test(5) # profile opt_aref
    ");
    assert_contains_opcode("test", YARVINSN_opt_aref);
    assert_snapshot!(assert_compiles_allowing_exits("[test(5), test(2**100 + 2)]"), @"[0, 1]");
}

#[test]
fn test_fixnum_mod_by_zero() {
    eval("
        def test(a, b) = a % b rescue :zero_div
        test(13, 4) # profile opt_mod
    ");
    assert_contains_opcode("test", YARVINSN_opt_mod);
    assert_snapshot!(assert_compiles_allowing_exits("test(13, 0)"), @":zero_div");
}

#[test]
fn test_fixnum_div_min_by_neg_one() {
    // FIXNUM_MIN / -1 overflows to a Bignum: the JIT must side exit, not return a mistyped Fixnum.
    eval("
        def test(a, b) = a / b
        test(10, 3) # profile opt_div
    ");
    assert_contains_opcode("test", YARVINSN_opt_div);
    assert_snapshot!(assert_compiles_allowing_exits("test(-4611686018427387904, -1)"), @"4611686018427387904");
}

#[test]
fn test_fixnum_div_overflow_propagation() {
    // The div must side exit before its Bignum result reaches the specialized (a / b) & 1 op.
    eval("
        def test(a, b) = (a / b) & 1
        test(10, 3) # profile opt_div
    ");
    assert_contains_opcode("test", YARVINSN_opt_div);
    assert_snapshot!(assert_compiles_allowing_exits("test(-4611686018427387904, -1)"), @"0");
}

#[test]
fn test_fixnum_div_by_neg_one_is_fine() {
    // x / -1 (x != FIXNUM_MIN) is a normal Fixnum and must NOT trip the overflow guard.
    eval("
        def test(a, b) = a / b
        test(10, 3) # profile opt_div
    ");
    assert_contains_opcode("test", YARVINSN_opt_div);
    assert_snapshot!(assert_compiles("test(10, -1)"), @"-10");
}

#[test]
fn test_opt_not() {
    eval("
        def test(obj) = !obj
    ");
    assert_contains_opcode("test", YARVINSN_opt_not);
    assert_snapshot!(assert_compiles_allowing_exits("[test(nil), test(false), test(0)]"), @"[true, true, false]");
}

#[test]
fn test_opt_regexpmatch2() {
    eval("
        def test(haystack) = /needle/ =~ haystack
    ");
    assert_contains_opcode("test", YARVINSN_opt_regexpmatch2);
    assert_snapshot!(assert_compiles(r#"[test("kneedle"), test("")]"#), @"[1, nil]");
}

#[test]
fn test_opt_ge() {
    eval("
        def test(a, b) = a >= b
        test(2, 3) # profile opt_ge
    ");
    assert_contains_opcode("test", YARVINSN_opt_ge);
    assert_snapshot!(assert_compiles("[test(0, 1), test(0, 0), test(1, 0)]"), @"[false, true, true]");
}

#[test]
fn test_opt_new_does_not_push_frame() {
    eval("
        class Foo
          attr_reader :backtrace
          def initialize
            @backtrace = caller
          end
        end
        def test = Foo.new
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_new);
    assert_snapshot!(assert_compiles("
        foo = test
        foo.backtrace.find { |frame| frame.include?('Class#new') }
    "), @"nil");
}

#[test]
fn test_opt_new_with_redefined() {
    eval(r#"
        class Foo
          def self.new = "foo"
          def initialize = raise("unreachable")
        end
        def test = Foo.new
        test
    "#);
    assert_contains_opcode("test", YARVINSN_opt_new);
    assert_snapshot!(assert_compiles(r#"test"#), @r#""foo""#);
}

#[test]
fn test_opt_new_invalidate_new() {
    eval(r#"
        class Foo; end
        def test = Foo.new
        test
    "#);
    assert_contains_opcode("test", YARVINSN_opt_new);
    assert_snapshot!(assert_compiles(r#"
        result = [test.class.name]
        def Foo.new = "foo"
        result << test
    "#), @r#"["Foo", "foo"]"#);
}

#[test]
fn test_opt_newarray_send_include_p() {
    eval("
        def test(x)
          [:y, 1, Object.new].include?(x)
        end
        test(1)
    ");
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles("[test(1), test(\"n\")]"), @"[true, false]");
}

#[test]
fn test_opt_newarray_send_include_p_redefined() {
    eval("
        class Array
          alias_method :old_include?, :include?
          def include?(x)
            old_include?(x) ? :true : :false
          end
        end
        def test(x)
          [:y, 1, Object.new].include?(x)
        end
    ");
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles_allowing_exits("
        def test(x)
          [:y, 1, Object.new].include?(x)
        end
        test(1)
        [test(1), test(\"n\")]
    "), @"[:true, :false]");
}

#[test]
fn test_opt_duparray_send_include_p() {
    eval("
        def test(x)
          [:y, 1].include?(x)
        end
        test(1)
    ");
    assert_contains_opcode("test", YARVINSN_opt_duparray_send);
    assert_snapshot!(assert_compiles("[test(1), test(\"n\")]"), @"[true, false]");
}

#[test]
fn test_opt_duparray_send_include_p_redefined() {
    eval("
        class Array
          alias_method :old_include?, :include?
          def include?(x)
            old_include?(x) ? :true : :false
          end
        end
        def test(x)
          [:y, 1].include?(x)
        end
    ");
    assert_contains_opcode("test", YARVINSN_opt_duparray_send);
    assert_snapshot!(assert_compiles_allowing_exits("
        def test(x)
          [:y, 1].include?(x)
        end
        test(1)
        [test(1), test(\"n\")]
    "), @"[:true, :false]");
}

#[test]
fn test_opt_newarray_send_pack() {
    eval(r#"
        def test(num)
          [num].pack('C')
        end
        test(65)
    "#);
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles(r#"
        [test(65), test(66), test(67)]
    "#), @r#"["A", "B", "C"]"#);
}

#[test]
fn test_opt_newarray_send_pack_redefined() {
    eval(r#"
        class Array
          alias_method :old_pack, :pack
          def pack(fmt, buffer: nil)
            "override:#{old_pack(fmt, buffer: buffer)}"
          end
        end
        def test(num)
          [num].pack('C')
        end
    "#);
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles_allowing_exits(r#"
        [test(65), test(66), test(67)]
    "#), @r#"["override:A", "override:B", "override:C"]"#);
}

#[test]
fn test_opt_newarray_send_pack_buffer() {
    eval(r#"
        def test(num, buffer)
          [num].pack('C', buffer:)
        end
        test(65, "")
    "#);
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles(r#"
        buf = ""
        [test(65, buf), test(66, buf), test(67, buf), buf]
    "#), @r#"["ABC", "ABC", "ABC", "ABC"]"#);
}

#[test]
fn test_opt_newarray_send_pack_buffer_redefined() {
    eval(r#"
        class Array
          alias_method :old_pack, :pack
          def pack(fmt, buffer: nil)
            old_pack(fmt, buffer: buffer)
            "b"
          end
        end
        def test(num, buffer)
          [num].pack('C', buffer:)
        end
    "#);
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles_allowing_exits(r#"
        def test(num, buffer)
          [num].pack('C', buffer:)
        end
        buf = ""
        test(65, buf)
        buf = ""
        [test(65, buf), buf]
    "#), @r#"["b", "A"]"#);
}

#[test]
fn test_opt_newarray_send_hash() {
    eval("
        def test(x)
          [1, 2, x].hash
        end
        test(20)
    ");
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles("test(20).class"), @"Integer");
}

#[test]
fn test_opt_newarray_send_hash_redefined() {
    eval("
        Array.class_eval { def hash = 42 }
        def test(x)
          [1, 2, x].hash
        end
        test(20)
    ");
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles_allowing_exits("test(20)"), @"42");
}

#[test]
fn test_opt_newarray_send_max() {
    eval("
        def test(a,b) = [a,b].max
        test(10, 20)
    ");
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles("[test(10, 20), test(40, 30)]"), @"[20, 40]");
}

#[test]
fn test_opt_newarray_send_max_redefined() {
    eval("
        class Array
          alias_method :old_max, :max
          def max
            old_max * 2
          end
        end
        def test(a,b) = [a,b].max
    ");
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles_allowing_exits("
        def test(a,b) = [a,b].max
        test(15, 30)
        [test(15, 30), test(45, 35)]
    "), @"[60, 90]");
}

#[test]
fn test_opt_newarray_send_min() {
    eval("
        def test(a,b) = [a,b].min
        test(10, 20)
    ");
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles("[test(10, 20), test(40, 30)]"), @"[10, 30]");
}

#[test]
fn test_opt_newarray_send_min_redefined() {
    eval("
        class Array
          alias_method :old_min, :min
          def min
            old_min * 2
          end
        end
        def test(a,b) = [a,b].min
    ");
    assert_contains_opcode("test", YARVINSN_opt_newarray_send);
    assert_snapshot!(assert_compiles_allowing_exits("
        def test(a,b) = [a,b].min
        test(15, 30)
        [test(15, 30), test(45, 35)]
    "), @"[30, 70]");
}

#[test]
fn test_new_hash_empty() {
    eval("
        def test = {}
        test
    ");
    assert_contains_opcode("test", YARVINSN_newhash);
    assert_snapshot!(assert_compiles("test"), @"{}");
}

// Exercises the empty-hash GC fast path under GC pressure. Guards against
// baking object flags as a GC-managed VALUE: T_HASH (8) has no immediate-mask
// bits set, so misclassifying it as a heap object records a bogus GC offset
// and crashes during marking.
#[test]
fn test_new_hash_empty_gc_stress() {
    eval("
        def make = {}
    ");
    assert_contains_opcode("make", YARVINSN_newhash);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          make
          h = make
          h[:a] = 1
          [h.class, h.size, h.default, h]
        ensure
          GC.stress = false
        end
    "#), @"[Hash, 1, nil, {a: 1}]");
}

// Static-symbol keys hash and compare without running Ruby, so NewHash takes the
// leaf bulk-insert fast path into an inline-allocated ar_table. Runs under GC
// stress to guard the leaf-call preparation.
#[test]
fn test_new_hash_static_sym_keys_gc_stress() {
    eval("
        def make(a, b) = {x: a, y: b}
    ");
    assert_contains_opcode("make", YARVINSN_newhash);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          make(1, 2)
          h = make(:foo, [3])
          [h.class, h.size, h[:x], h[:y], h[:z], h.default, h]
        ensure
          GC.stress = false
        end
    "#), @"[Hash, 2, :foo, [3], nil, nil, {x: :foo, y: [3]}]");
}

// Eight pairs fills an inline embedded ar_table (the fast path); nine crosses
// RHASH_AR_TABLE_MAX_SIZE, so it's built as a pre-sized st_table instead. Both stay
// on the static-symbol leaf path, so this guards the ar_table and st_table routes.
#[test]
fn test_new_hash_static_sym_ar_table_boundary() {
    eval("
        def eight(v) = {a:v,b:v,c:v,d:v,e:v,f:v,g:v,h:v}
        def nine(v)  = {a:v,b:v,c:v,d:v,e:v,f:v,g:v,h:v,i:v}
    ");
    assert_contains_opcode("eight", YARVINSN_newhash);
    assert_contains_opcode("nine", YARVINSN_newhash);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          [eight(7).size, eight(7)[:h], nine([9]).size, nine([9])[:i]]
        ensure
          GC.stress = false
        end
    "#), @"[8, 7, 9, [9]]");
}

// Dynamic symbol keys hash and compare without running Ruby, so NewHash takes the
// leaf bulk-insert fast path into an inline-allocated ar_table. Runs under GC
// stress to guard the leaf-call preparation.
#[test]
fn test_new_hash_dynamic_sym_keys_gc_stress() {
    eval(r#"
        def make(k, v) = { :"x_#{k}" => v, :"y_#{k}" => v }
    "#);
    assert_contains_opcode("make", YARVINSN_newhash);
    assert_contains_opcode("make", YARVINSN_intern);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          make("warm", 0)
          h = make("k", [3])
          [h.class, h.size, h[:"x_k"], h[:"y_k"]]
        ensure
          GC.stress = false
        end
    "#), @"[Hash, 2, [3], [3]]");
}

// The NewHash inline-alloc fast path must bake the slot-size shape_id into the
// object flags. Without it, a cross-ractor move sizes the destination object
// from a zero shape_id, so the moved hash is allocated too small and its keys
// are corrupted.
#[test]
fn test_new_hash_sym_keys_ractor_move() {
    eval("
        def create_hash
          { an_object: Array.new, hi: true, bonjour: true }
        end
    ");
    assert_contains_opcode("create_hash", YARVINSN_newhash);
    assert_snapshot!(inspect("
        r = Ractor.new do
          h = receive
          30.times { |i| h[i] = true }
          h.keys.delete_if { |k| Integer === k }
        end

        create_hash
        create_hash

        h = create_hash
        r.send(h, move: true)
        r.value
    "), @"[:an_object, :hi, :bonjour]");
}

#[test]
fn test_object_alloc_gc_stress() {
    eval("
        class Foo
          def initialize
            @a = 1
            @b = 2
          end
          def sum = @a + @b
        end
        def make = Foo.new
    ");
    assert_contains_opcode("make", YARVINSN_opt_new);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          make
          foo = make
          foo.instance_variable_set(:@c, 3)
          [foo.class, foo.sum, foo.instance_variables]
        ensure
          GC.stress = false
        end
    "#), @"[Foo, 3, [:@a, :@b, :@c]]");
}

#[test]
fn test_string_copy_gc_stress() {
    eval(r#"
        # frozen_string_literal: false
        def make = "hello world"
    "#);
    assert_contains_opcode("make", YARVINSN_dupstring);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          make
          s = make
          orig = s.dup
          s << "!"
          [s.class, s, s.frozen?, s.encoding.name, s.length, orig]
        ensure
          GC.stress = false
        end
    "#), @r#"[String, "hello world!", false, "UTF-8", 12, "hello world"]"#);
}

#[test]
fn test_string_copy_large_gc_stress() {
    eval(r#"
        # frozen_string_literal: false
        def make = "the quick brown fox jumps over the lazy dog, the quick brown fox jumps over"
    "#);
    assert_contains_opcode("make", YARVINSN_dupstring);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          make
          s = make
          s << "!"
          [s.class, s.frozen?, s.length, s.end_with?("!")]
        ensure
          GC.stress = false
        end
    "#), @"[String, false, 76, true]");
}

#[test]
fn test_string_copy_memcpy_gc_stress() {
    eval(r#"
        # frozen_string_literal: false
        def make = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
    "#);
    assert_contains_opcode("make", YARVINSN_dupstring);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          make
          s = make
          s << "!"
          [s.class, s.frozen?, s.length, s.end_with?("!")]
        ensure
          GC.stress = false
        end
    "#), @"[String, false, 157, true]");
}

#[test]
fn test_string_copy_chilled_gc_stress() {
    eval(r#"
        def make = "hello world"
    "#);
    assert_contains_opcode("make", YARVINSN_dupchilledstring);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          make
          s = make
          orig = s.dup
          s << "!"
          [s.class, s, s.frozen?, s.encoding.name, s.length, orig]
        ensure
          GC.stress = false
        end
    "#), @r#"[String, "hello world!", false, "UTF-8", 12, "hello world"]"#);
}

#[test]
fn test_string_append_same_encoding() {
    eval(r#"
        def test(s, x) = s << x
    "#);
    assert_contains_opcode("test", YARVINSN_opt_ltlt);
    assert_snapshot!(assert_compiles(r#"
        s = +"abc"
        test(s, "déf")
        test(s, "ghé")
        [s, s.encoding.name, s.valid_encoding?]
    "#), @r#"["abcdéfghé", "UTF-8", true]"#);
}

#[test]
fn test_string_append_encoding_mismatch() {
    eval(r#"
        def test(s, x) = s << x
    "#);
    assert_contains_opcode("test", YARVINSN_opt_ltlt);
    // The first append takes the mismatched-encoding path and switches the
    // empty BINARY receiver to UTF-8; later appends take the fast path.
    assert_snapshot!(assert_compiles(r#"
        s = String.new(encoding: Encoding::BINARY)
        test(s, "é")
        test(s, "é")
        [s, s.encoding.name, s.valid_encoding?]
    "#), @r#"["éé", "UTF-8", true]"#);
}

#[test]
fn test_string_append_incompatible_encoding() {
    eval(r#"
        def test(s, x) = s << x
    "#);
    assert_contains_opcode("test", YARVINSN_opt_ltlt);
    assert_snapshot!(assert_compiles(r#"
        s = "\xFF".b
        begin
          test(s, "é")
          :no_error
        rescue Encoding::CompatibilityError
          :compatibility_error
        end
    "#), @":compatibility_error");
}

#[test]
fn test_string_append_broken_coderange() {
    eval(r#"
        def test(s, x) = s << x
    "#);
    assert_contains_opcode("test", YARVINSN_opt_ltlt);
    // Same encoding, but the appended bytes break the receiver's coderange.
    assert_snapshot!(assert_compiles(r#"
        s = +"abc"
        test(s, "\xFF".dup.force_encoding(Encoding::UTF_8))
        [s.bytesize, s.valid_encoding?]
    "#), @"[4, false]");
}

#[test]
fn test_string_append_growth() {
    eval(r#"
        def test(s, x) = s << x
    "#);
    assert_contains_opcode("test", YARVINSN_opt_ltlt);
    // Cross the embedded->heap boundary and several capacity doublings so
    // both the in-place fast path and the resizing fallback are exercised.
    assert_snapshot!(assert_compiles(r#"
        s = String.new(encoding: Encoding::UTF_8)
        200.times { test(s, "0123456789") }
        [s.bytesize, s == "0123456789" * 200]
    "#), @"[2000, true]");
}

#[test]
fn test_string_append_codepoint_binary() {
    eval(r#"
        def test(s, x) = s << x
    "#);
    assert_contains_opcode("test", YARVINSN_opt_ltlt);
    // Grow a binary buffer one byte at a time, crossing the embedded->heap boundary
    // and several capacity doublings, with both ASCII and non-ASCII bytes.
    assert_snapshot!(assert_compiles(r#"
        s = String.new(encoding: Encoding::BINARY)
        1000.times { |i| test(s, i % 255) }
        [s.bytesize, s == (0...1000).map { |i| (i % 255).chr(Encoding::BINARY) }.join, s.encoding.name]
    "#), @r#"[1000, true, "ASCII-8BIT"]"#);
}

#[test]
fn test_string_append_codepoint_coderange() {
    eval(r#"
        def test(s, x) = s << x
    "#);
    assert_contains_opcode("test", YARVINSN_opt_ltlt);
    // ASCII bytes keep the buffer 7BIT; a non-ASCII byte makes it VALID for good.
    assert_snapshot!(assert_compiles(r#"
        s = String.new(encoding: Encoding::BINARY)
        10.times { test(s, 0x41) }
        before = s.ascii_only?
        test(s, 0xC3)
        middle = [s.ascii_only?, s.valid_encoding?]
        test(s, 0x41)
        [before, middle, s.ascii_only?, s.bytesize]
    "#), @"[true, [false, true], false, 12]");
}

#[test]
fn test_string_append_codepoint_slow_paths() {
    eval(r#"
        def test(s, x) = s << x
    "#);
    assert_contains_opcode("test", YARVINSN_opt_ltlt);
    // Frozen receivers, out-of-range codepoints, shared buffers, and non-binary
    // encodings all have to fall back to rb_str_concat().
    assert_snapshot!(assert_compiles(r#"
        results = []
        1000.times { test(String.new(encoding: Encoding::BINARY), 0x41) }
        results << (begin; test("abc".b.freeze, 0x41); rescue FrozenError; :frozen; end)
        results << (begin; test(String.new(encoding: Encoding::BINARY), -1); rescue RangeError; :range; end)
        results << (begin; test(String.new(encoding: Encoding::BINARY), 0x100); rescue RangeError; :range; end)
        base = "y".b * 200
        shared = base[0, 100]
        # A plain loop, not `10.times`: a second block at `Integer#times`'s `yield i` would
        # miss the direct block dispatch it specialized on the first one and exit once.
        i = 0
        while i < 10
          test(shared, 0x42)
          i += 1
        end
        results << [base == "y".b * 200, shared.bytesize]
        utf8 = +"abc"
        results << test(utf8, 0x3042)
        results
    "#), @r#"[:frozen, :range, :range, [true, 110], "abcあ"]"#);
}

#[test]
fn test_string_append_codepoint_gc_stress() {
    eval(r#"
        def test(s, x) = s << x
    "#);
    assert_contains_opcode("test", YARVINSN_opt_ltlt);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          s = "a".b
          10.times { test(s, 0x62) }
          [s.bytesize, s == "a" + "b" * 10]
        ensure
          GC.stress = false
        end
    "#), @"[11, true]");
}

#[test]
fn test_string_append_gc_stress() {
    eval(r#"
        def test(s, x) = s << x
    "#);
    assert_contains_opcode("test", YARVINSN_opt_ltlt);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          s = +"a"
          10.times { test(s, "bc") }
          [s.bytesize, s == "a" + "bc" * 10]
        ensure
          GC.stress = false
        end
    "#), @"[21, true]");
}

#[test]
fn test_string_ascii_only_p_unknown_coderange() {
    eval(r#"
        def test(s) = s.ascii_only?
        test("a#{1}")
        test("a#{1}")
    "#);
    // A string built at runtime has an UNKNOWN coderange until something scans it, so this scans
    // on a cold path rather than exiting.
    assert_snapshot!(assert_compiles(r#"[test("abc#{1}"), test("é#{1}")]"#), @"[true, false]");
}

#[test]
fn test_string_valid_encoding_p_unknown_coderange() {
    eval(r#"
        def test(s) = s.valid_encoding?
        test("a#{1}")
        test("a#{1}")
    "#);
    assert_snapshot!(assert_compiles(r#"
        [test("abc#{1}"), test("é#{1}"), test("\xFF#{1}".b.force_encoding(Encoding::UTF_8))]
    "#), @"[true, true, false]");
}

#[test]
fn test_new_hash_nonempty() {
    eval(r#"
        def test
          key = "key"
          value = "value"
          num = 42
          result = 100
          {key => value, num => result}
        end
        test
    "#);
    assert_contains_opcode("test", YARVINSN_newhash);
    assert_snapshot!(assert_compiles(r#"test"#), @r#"{"key" => "value", 42 => 100}"#);
}

#[test]
fn test_new_hash_single_key_value() {
    eval(r#"
        def test = {"key" => "value"}
        test
    "#);
    assert_contains_opcode("test", YARVINSN_newhash);
    assert_snapshot!(assert_compiles(r#"test"#), @r#"{"key" => "value"}"#);
}

#[test]
fn test_new_hash_with_computation() {
    eval(r#"
        def test(a, b)
          {"sum" => a + b, "product" => a * b}
        end
        test(2, 3)
    "#);
    assert_contains_opcode("test", YARVINSN_newhash);
    assert_snapshot!(assert_compiles(r#"test(2, 3)"#), @r#"{"sum" => 5, "product" => 6}"#);
}

#[test]
fn test_new_hash_with_user_defined_hash_method() {
    assert_snapshot!(inspect(r#"
        class CustomKey
          attr_reader :val
          def initialize(val)
            @val = val
          end
          def hash
            @val.hash
          end
          def eql?(other)
            other.is_a?(CustomKey) && @val == other.val
          end
        end
        def test
          key = CustomKey.new("key")
          hash = {key => "value"}
          hash[key] == "value"
        end
        test
        test
    "#), @"true");
}

#[test]
fn test_new_hash_with_user_hash_method_exception() {
    assert_snapshot!(inspect(r#"
        class BadKey
          def hash
            raise "Hash method failed!"
          end
        end
        def test
          key = BadKey.new
          {key => "value"}
        end
        begin
          test
        rescue => e
          e.class
        end
        begin
          test
        rescue => e
          e.class
        end
    "#), @"RuntimeError");
}

#[test]
fn test_new_hash_with_user_eql_method_exception() {
    assert_snapshot!(inspect(r#"
        class BadKey
          def hash
            42
          end
          def eql?(other)
            raise "Eql method failed!"
          end
        end
        def test
          key1 = BadKey.new
          key2 = BadKey.new
          {key1 => "value1", key2 => "value2"}
        end
        begin
          test
        rescue => e
          e.class
        end
        begin
          test
        rescue => e
          e.class
        end
    "#), @"RuntimeError");
}

#[test]
fn test_opt_hash_freeze() {
    eval("
        def test = {}.freeze
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_hash_freeze);
    assert_snapshot!(assert_compiles("
        result = [test]
        class Hash
          def freeze = 5
        end
        result << test
    "), @"[{}, 5]");
}

#[test]
fn test_opt_hash_freeze_rewritten() {
    eval("
        class Hash
          def freeze = 5
        end
        def test = {}.freeze
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_hash_freeze);
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"5");
}

#[test]
fn test_opt_aset_hash() {
    eval("
        def test(h, k, v)
          h[k] = v
        end
        test({}, :key, 42)
    ");
    assert_contains_opcode("test", YARVINSN_opt_aset);
    assert_snapshot!(assert_compiles("h = {}; test(h, :key, 42); h[:key]"), @"42");
}

#[test]
fn test_opt_aset_hash_returns_value() {
    assert_snapshot!(inspect("
        def test(h, k, v)
          h[k] = v
        end
        test({}, :key, 100)
        test({}, :key, 100)
    "), @"100");
}

#[test]
fn test_opt_aset_hash_string_key() {
    assert_snapshot!(inspect(r#"
        def test(h, k, v)
          h[k] = v
        end
        h = {}
        test(h, "foo", "bar")
        test(h, "foo", "bar")
        h["foo"]
    "#), @r#""bar""#);
}

#[test]
fn test_opt_aset_hash_subclass() {
    assert_snapshot!(inspect("
        class MyHash < Hash; end
        def test(h, k, v)
          h[k] = v
        end
        h = MyHash.new
        test(h, :key, 42)
        test(h, :key, 42)
        h[:key]
    "), @"42");
}

#[test]
fn test_opt_aset_hash_too_few_args() {
    assert_snapshot!(inspect(r#"
        def test(h)
          h.[]= 123
        rescue ArgumentError
          "ArgumentError"
        end
        test({})
        test({})
    "#), @r#""ArgumentError""#);
}

#[test]
fn test_opt_aset_hash_too_many_args() {
    assert_snapshot!(inspect(r#"
        def test(h)
          h[:a, :b] = :c
        rescue ArgumentError
          "ArgumentError"
        end
        test({})
        test({})
    "#), @r#""ArgumentError""#);
}

#[test]
fn test_opt_ary_freeze() {
    eval("
        def test = [].freeze
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_ary_freeze);
    assert_snapshot!(assert_compiles("
        result = [test]
        class Array
          def freeze = 5
        end
        result << test
    "), @"[[], 5]");
}

#[test]
fn test_opt_ary_freeze_rewritten() {
    eval("
        class Array
          def freeze = 5
        end
        def test = [].freeze
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_ary_freeze);
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"5");
}

#[test]
fn test_opt_str_freeze() {
    eval("
        def test = ''.freeze
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_str_freeze);
    assert_snapshot!(assert_compiles(r#"
        result = [test]
        class String
          def freeze = 5
        end
        result << test
    "#), @r#"["", 5]"#);
}

#[test]
fn test_opt_str_freeze_rewritten() {
    eval("
        class String
          def freeze = 5
        end
        def test = ''.freeze
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_str_freeze);
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"5");
}

#[test]
fn test_opt_str_uminus() {
    eval("
        def test = -''
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_str_uminus);
    assert_snapshot!(assert_compiles(r#"
        result = [test]
        class String
          def -@ = 5
        end
        result << test
    "#), @r#"["", 5]"#);
}

#[test]
fn test_opt_str_uminus_rewritten() {
    eval("
        class String
          def -@ = 5
        end
        def test = -''
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_str_uminus);
    assert_snapshot!(assert_compiles_allowing_exits("test"), @"5");
}

#[test]
fn test_new_array_empty() {
    eval("
        def test = []
        test
    ");
    assert_contains_opcode("test", YARVINSN_newarray);
    assert_snapshot!(assert_compiles("test"), @"[]");
}

#[test]
fn test_new_array_nonempty() {
    assert_snapshot!(inspect("
        def a = 5
        def test = [a]
        test
        test
    "), @"[5]");
}

#[test]
fn test_new_array_order() {
    assert_snapshot!(inspect("
        def a = 3
        def b = 2
        def c = 1
        def test = [a, b, c]
        test
        test
    "), @"[3, 2, 1]");
}

#[test]
fn test_new_array_embedded_gc_stress() {
    eval(r#"
        def make(a) = [a, a, a]
    "#);
    assert_contains_opcode("make", YARVINSN_newarray);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          s = "x"
          make(s)
          a = make(s)
          a << :extra
          [a.frozen?, a.class, a]
        ensure
          GC.stress = false
        end
    "#), @r#"[false, Array, ["x", "x", "x", :extra]]"#);
}

#[test]
fn test_new_array_embedded_memcpy_gc_stress() {
    eval(r#"
        def make(a) = [a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a] # size: 17
    "#);
    assert_contains_opcode("make", YARVINSN_newarray);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          s = "y"
          make(s)
          m = make(s)
          [m.frozen?, m.length, m.class]
        ensure
          GC.stress = false
        end
    "#), @"[false, 17, Array]");
}

#[test]
fn test_array_dup() {
    assert_snapshot!(inspect("
        def test = [1,2,3]
        test
        test
    "), @"[1, 2, 3]");
}

#[test]
fn test_array_dup_embedded_gc_stress() {
    eval(r#"
        def make = [1, 100000000000000000000, :sym]
    "#);
    assert_contains_opcode("make", YARVINSN_duparray);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          make
          a = make
          a << :extra
          [a.frozen?, a.class, a]
        ensure
          GC.stress = false
        end
    "#), @"[false, Array, [1, 100000000000000000000, :sym, :extra]]");
}

#[test]
fn test_array_dup_non_embedded_gc_stress() {
    eval("
        def make = [10, 20, 30, 40, 50]
    ");
    assert_contains_opcode("make", YARVINSN_duparray);
    assert_snapshot!(assert_compiles(r#"
        begin
          GC.stress = true
          make
          m = make
          [m.frozen?, m]
        ensure
          GC.stress = false
        end
    "#), @"[false, [10, 20, 30, 40, 50]]");
}

#[test]
fn test_array_push_embedded_and_heap_growth() {
    // Pushes cross the embedded->heap boundary and grow the heap buffer,
    // exercising both the fast path and the rb_ary_push fallback
    assert_snapshot!(inspect("
        def test(a, v) = a << v
        arr = []
        100.times { |i| test(arr, i) }
        [arr.size, arr.first, arr.last, arr.sum]
    "), @"[100, 0, 99, 4950]");
}

#[test]
fn test_array_push_returns_array() {
    assert_snapshot!(inspect("
        def test(a, v) = a << v
        arr = [1] * 30
        test(arr, 2).equal?(arr)
    "), @"true");
}

#[test]
fn test_array_push_shared_array() {
    // Pushing to an array whose buffer is shared (or shared root) must not
    // corrupt the other array
    assert_snapshot!(inspect("
        def test(a, v) = a << v
        test([], 0)
        root = (0..20).to_a
        child = root[0..-1]
        test(child, :x)
        test(root, :y)
        [child.last, root.last, child.size, root.size]
    "), @"[:x, :y, 22, 22]");
}

#[test]
fn test_array_push_frozen() {
    assert_snapshot!(inspect("
        def test(a, v) = a << v
        test([], 0)
        arr = ([1] * 30).freeze
        begin
          test(arr, 2)
        rescue FrozenError
          :frozen
        end
    "), @":frozen");
}

#[test]
fn test_array_push_write_barrier_gc_stress() {
    // An old array referencing young pushed objects must survive GC
    assert_snapshot!(inspect(r#"
        def test(a, v) = a << v
        arr = [nil] * 30
        arr.clear
        3.times { GC.start } # promote arr to old gen
        GC.stress = true
        10.times { |i| test(arr, "str#{i}") }
        GC.stress = false
        GC.start
        [arr.size, arr.first, arr.last]
    "#), @r#"[10, "str0", "str9"]"#);
}

#[test]
fn test_array_fixnum_aref() {
    eval("
        def test(x) = [1,2,3][x]
        test(2)
    ");
    assert_contains_opcode("test", YARVINSN_opt_aref);
    assert_snapshot!(assert_compiles("test(2)"), @"3");
}

#[test]
fn test_array_fixnum_aref_negative_index() {
    eval("
        def test(x) = [1,2,3][x]
        test(-1)
    ");
    assert_contains_opcode("test", YARVINSN_opt_aref);
    assert_snapshot!(assert_compiles("test(-1)"), @"3");
}

#[test]
fn test_array_fixnum_aref_out_of_bounds_positive() {
    eval("
        def test(x) = [1,2,3][x]
        test(10)
    ");
    assert_contains_opcode("test", YARVINSN_opt_aref);
    assert_snapshot!(assert_compiles_allowing_exits("test(10)"), @"nil");
}

#[test]
fn test_array_fixnum_aref_out_of_bounds_negative() {
    eval("
        def test(x) = [1,2,3][x]
        test(-10)
    ");
    assert_contains_opcode("test", YARVINSN_opt_aref);
    assert_snapshot!(assert_compiles_allowing_exits("test(-10)"), @"nil");
}

#[test]
fn test_array_fixnum_aref_array_subclass() {
    eval("
        class MyArray < Array; end
        def test(arr, idx) = arr[idx]
        test(MyArray[1,2,3], 2)
    ");
    assert_contains_opcode("test", YARVINSN_opt_aref);
    assert_snapshot!(assert_compiles("test(MyArray[1,2,3], 2)"), @"3");
}

#[test]
fn test_array_aref_non_fixnum_index() {
    assert_snapshot!(inspect(r#"
        def test(arr, idx) = arr[idx]
        test([1,2,3], 1)
        test([1,2,3], 1)
        begin
          test([1,2,3], "1")
        rescue => e
          e.class
        end
    "#), @"TypeError");
}

#[test]
fn test_array_fixnum_aset() {
    eval("
        def test(arr, idx)
          arr[idx] = 7
        end
        test([1,2,3], 2)
    ");
    assert_contains_opcode("test", YARVINSN_opt_aset);
    assert_snapshot!(assert_compiles("arr = [1,2,3]; test(arr, 2); arr"), @"[1, 2, 7]");
}

#[test]
fn test_array_fixnum_aset_returns_value() {
    eval("
        def test(arr, idx)
          arr[idx] = 7
        end
        test([1,2,3], 2)
    ");
    assert_contains_opcode("test", YARVINSN_opt_aset);
    assert_snapshot!(assert_compiles("test([1,2,3], 2)"), @"7");
}

#[test]
fn test_array_fixnum_aset_out_of_bounds() {
    assert_snapshot!(inspect("
        def test(arr)
          arr[5] = 7
        end
        arr = [1,2,3]
        test(arr)
        arr = [1,2,3]
        test(arr)
        arr
    "), @"[1, 2, 3, nil, nil, 7]");
}

#[test]
fn test_array_fixnum_aset_grows_array_without_exiting() {
    eval("
        def test(arr, n)
          i = 0
          while i < n
            arr[i] = i
            i += 1
          end
          arr
        end
        test([], 3)
    ");
    assert_contains_opcode("test", YARVINSN_opt_aset);
    assert_snapshot!(assert_compiles("test([], 5)"), @"[0, 1, 2, 3, 4]");
}

/// The push half of a Ragel-generated parser's state machine.
#[test]
fn test_array_fixnum_aset_ragel_push_without_exiting() {
    eval("
        def test(n)
          stack = []
          top = 0
          cs = 7
          i = 0
          while i < n
            stack[top] = cs
            top += 1
            i += 1
          end
          stack
        end
        test(3)
    ");
    assert_snapshot!(assert_compiles("test(4)"), @"[7, 7, 7, 7]");
}

#[test]
fn test_array_fixnum_aset_negative_out_of_bounds() {
    assert_snapshot!(inspect("
        def test(arr, idx)
          arr[idx] = 7
        end
        test([1,2,3], -1)
        test([1,2,3], -1)
        begin
          test([1,2,3], -5)
        rescue IndexError => e
          e.message
        end
    "), @r#""index -5 too small for array; minimum: -3""#);
}

#[test]
fn test_array_fixnum_aset_negative_index() {
    assert_snapshot!(inspect("
        def test(arr)
          arr[-1] = 7
        end
        arr = [1,2,3]
        test(arr)
        arr = [1,2,3]
        test(arr)
        arr
    "), @"[1, 2, 7]");
}

#[test]
fn test_array_fixnum_aset_shared() {
    assert_snapshot!(inspect("
        def test(arr, idx, val)
          arr[idx] = val
        end
        arr = (0..50).to_a
        test(arr, 0, -1)
        test(arr, 1, -2)
        shared = arr[10, 20]
        test(shared, 0, 999)
        [arr[10], shared[0], arr[0], arr[1]]
    "), @"[10, 999, -1, -2]");
}

#[test]
fn test_array_fixnum_aset_frozen() {
    assert_snapshot!(inspect("
        def test(arr, idx, val)
          arr[idx] = val
        end
        arr = [1,2,3]
        test(arr, 1, 9)
        test(arr, 1, 9)
        arr.freeze
        begin
          test(arr, 1, 9)
        rescue => e
          e.class
        end
    "), @"FrozenError");
}

#[test]
fn test_array_fixnum_aset_array_subclass() {
    eval("
        class MyArray < Array; end
        def test(arr, idx)
          arr[idx] = 7
        end
        test(MyArray.new, 0)
    ");
    assert_contains_opcode("test", YARVINSN_opt_aset);
    assert_snapshot!(assert_compiles("arr = MyArray.new; test(arr, 0); arr[0]"), @"7");
}

#[test]
fn test_array_aset_non_fixnum_index() {
    assert_snapshot!(inspect(r#"
        def test(arr, idx)
          arr[idx] = 7
        end
        test([1,2,3], 0)
        test([1,2,3], 0)
        begin
          test([1,2,3], "0")
        rescue => e
          e.class
        end
    "#), @"TypeError");
}

#[test]
fn test_empty_array_pop() {
    assert_snapshot!(inspect("
        def test(arr) = arr.pop
        test([])
        test([])
    "), @"nil");
}

#[test]
fn test_array_pop_no_arg() {
    assert_snapshot!(inspect("
        def test(arr) = arr.pop
        test([32, 33, 42])
        test([32, 33, 42])
    "), @"42");
}

#[test]
fn test_array_pop_arg() {
    assert_snapshot!(inspect("
        def test(arr) = arr.pop(2)
        test([32, 33, 42])
        test([32, 33, 42])
    "), @"[33, 42]");
}

#[test]
fn test_new_range_inclusive() {
    assert_snapshot!(inspect("
        def test(a, b) = a..b
        test(1, 5)
        test(1, 5)
    "), @"1..5");
}

#[test]
fn test_new_range_exclusive() {
    assert_snapshot!(inspect("
        def test(a, b) = a...b
        test(1, 5)
        test(1, 5)
    "), @"1...5");
}

#[test]
fn test_new_range_with_literal() {
    assert_snapshot!(inspect("
        def test(n) = n..10
        test(3)
        test(3)
    "), @"3..10");
}

#[test]
fn test_new_range_fixnum_both_literals_inclusive() {
    eval("
        def test()
          a = 2
          (1..a)
        end
    ");
    assert_contains_opcode("test", YARVINSN_newrange);
    assert_snapshot!(assert_compiles("test; test"), @"1..2");
}

#[test]
fn test_new_range_fixnum_both_literals_exclusive() {
    eval("
        def test()
          a = 2
          (1...a)
        end
    ");
    assert_contains_opcode("test", YARVINSN_newrange);
    assert_snapshot!(assert_compiles("test; test"), @"1...2");
}

#[test]
fn test_new_range_fixnum_low_literal_inclusive() {
    eval("
        def test(a) = (1..a)
    ");
    assert_contains_opcode("test", YARVINSN_newrange);
    assert_snapshot!(assert_compiles("test(2); test(3)"), @"1..3");
}

#[test]
fn test_new_range_fixnum_low_literal_exclusive() {
    eval("
        def test(a) = (1...a)
    ");
    assert_contains_opcode("test", YARVINSN_newrange);
    assert_snapshot!(assert_compiles("test(2); test(3)"), @"1...3");
}

#[test]
fn test_new_range_fixnum_high_literal_inclusive() {
    eval("
        def test(a) = (a..10)
    ");
    assert_contains_opcode("test", YARVINSN_newrange);
    assert_snapshot!(assert_compiles("test(2); test(3)"), @"3..10");
}

#[test]
fn test_new_range_fixnum_high_literal_exclusive() {
    eval("
        def test(a) = (a...10)
    ");
    assert_contains_opcode("test", YARVINSN_newrange);
    assert_snapshot!(assert_compiles("test(2); test(3)"), @"3...10");
}

#[test]
fn test_if() {
    assert_snapshot!(inspect("
        def test(n)
          if n < 5
            0
          end
        end
        test(3)
        [test(3), test(7)]
    "), @"[0, nil]");
}

#[test]
fn test_if_else() {
    assert_snapshot!(inspect("
        def test(n)
          if n < 5
            0
          else
            1
          end
        end
        test(3)
        [test(3), test(7)]
    "), @"[0, 1]");
}

#[test]
fn test_if_fixnum_compare_kinds() {
    // Each fixnum compare fused into its CondBranch must still branch correctly
    // in both directions.
    assert_snapshot!(inspect("
        def lt(a, b)  = a <  b ? 1 : 0
        def le(a, b)  = a <= b ? 1 : 0
        def gt(a, b)  = a >  b ? 1 : 0
        def ge(a, b)  = a >= b ? 1 : 0
        def eq(a, b)  = a == b ? 1 : 0
        def neq(a, b) = a != b ? 1 : 0
        r = []
        2.times do
          r = [lt(1, 2), lt(2, 1), le(1, 1), le(2, 1), gt(2, 1), gt(1, 2),
               ge(1, 1), ge(1, 2), eq(1, 1), eq(1, 2), neq(1, 2), neq(1, 1)]
        end
        r
    "), @"[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]");
}

#[test]
fn test_if_compare_result_also_used() {
    // The comparison result is both branched on and returned, so it cannot be
    // folded away into the branch and must still be materialized.
    assert_snapshot!(inspect("
        def test(a, b)
          x = a < b
          if x
            x
          else
            :nope
          end
        end
        test(1, 2)
        [test(1, 2), test(2, 1)]
    "), @"[true, :nope]");
}

#[test]
fn test_if_truthiness_of_non_boolean() {
    // Branching on an arbitrary VALUE: nil and false are falsy, everything else truthy.
    assert_snapshot!(inspect("
        def test(x) = x ? 1 : 0
        test(1)
        [test(1), test(true), test(:sym), test(nil), test(false), test(0)]
    "), @"[1, 1, 1, 0, 0, 1]");
}

#[test]
fn test_if_else_params() {
    assert_snapshot!(inspect("
        def test(n, a, b)
          if n < 5
            a
          else
            b
          end
        end
        test(3, 1, 2)
        [test(3, 1, 2), test(7, 10, 20)]
    "), @"[1, 20]");
}

#[test]
fn test_if_else_nested() {
    assert_snapshot!(inspect("
        def test(a, b, c, d, e)
          if 2 < a
            if a < 4
              b
            else
              c
            end
          else
            if a < 0
              d
            else
              e
            end
          end
        end
        test(-1, 1, 2, 3, 4)
        [
          test(-1,  1,  2,  3,  4),
          test( 0,  5,  6,  7,  8),
          test( 3,  9, 10, 11, 12),
          test( 5, 13, 14, 15, 16),
        ]
    "), @"[3, 8, 9, 14]");
}

#[test]
fn test_if_else_chained() {
    assert_snapshot!(inspect("
        def test(a)
          (if 2 < a then 1 else 2 end) + (if a < 4 then 10 else 20 end)
        end
        test(0)
        [test(0), test(3), test(5)]
    "), @"[12, 11, 21]");
}

#[test]
fn test_if_elsif_else() {
    assert_snapshot!(inspect("
        def test(n)
          if n < 5
            0
          elsif 8 < n
            1
          else
            2
          end
        end
        test(3)
        [test(3), test(7), test(9)]
    "), @"[0, 2, 1]");
}

#[test]
fn test_ternary_operator() {
    assert_snapshot!(inspect("
        def test(n, a, b)
          n < 5 ? a : b
        end
        test(3, 1, 2)
        [test(3, 1, 2), test(7, 10, 20)]
    "), @"[1, 20]");
}

#[test]
fn test_ternary_operator_nested() {
    assert_snapshot!(inspect("
        def test(n, a, b)
          (n < 5 ? a : b) + 1
        end
        test(3, 1, 2)
        [test(3, 1, 2), test(7, 10, 20)]
    "), @"[2, 21]");
}

#[test]
fn test_while_loop() {
    assert_snapshot!(inspect("
        def test(n)
          i = 0
          while i < n
            i = i + 1
          end
          i
        end
        test(10)
        test(10)
    "), @"10");
}

#[test]
fn test_while_loop_chain() {
    assert_snapshot!(inspect("
        def test(n)
          i = 0
          while i < n
            i = i + 1
          end
          while i < n * 10
            i = i * 3
          end
          i
        end
        test(5)
        [test(5), test(10)]
    "), @"[135, 270]");
}

#[test]
fn test_while_loop_nested() {
    assert_snapshot!(inspect("
        def test(n, m)
          i = 0
          while i < n
            j = 0
            while j < m
              j += 2
            end
            i += j
          end
          i
        end
        test(0, 0)
        [test(0, 0), test(1, 3), test(10, 5)]
    "), @"[0, 4, 12]");
}

#[test]
fn test_while_loop_if_else() {
    assert_snapshot!(inspect("
        def test(n)
          i = 0
          while i < n
            if n >= 10
              return -1
            else
              i = i + 1
            end
          end
          i
        end
        test(9)
        [test(9), test(10)]
    "), @"[9, -1]");
}

#[test]
fn test_if_while_loop() {
    assert_snapshot!(inspect("
        def test(n)
          i = 0
          if n < 10
            while i < n
              i += 1
            end
          else
            while i < n
              i += 3
            end
          end
          i
        end
        test(9)
        [test(9), test(10)]
    "), @"[9, 12]");
}

#[test]
fn test_live_reg_past_ccall() {
    assert_snapshot!(inspect("
        def callee = 1
        def test = callee + callee
        test
        test
    "), @"2");
}

#[test]
fn test_method_call() {
    assert_snapshot!(inspect("
        def callee(a, b)
          a - b
        end
        def test
          callee(4, 2) + 10
        end
        test
        test
    "), @"12");
}

#[test]
fn test_polymorphic_iseq_dispatch_same_site() {
    assert_snapshot!(inspect("
        class A; def foo = 1; end
        class B; def foo = 2; end
        def test(obj) = obj.foo
        test(A.new); test(A.new)   # warm up and specialize the call site for A
        [test(A.new), test(B.new)]
    "), @"[1, 2]");
}

#[test]
fn test_polymorphic_send_with_literal_block_dispatches_directly() {
    // A polymorphic call site that passes a literal block dispatches on the receiver type in
    // the same way a block-less send does, and every arm must run the block correctly.
    set_call_threshold(4);
    eval("
        class C; def each; yield 1; yield 2; end; end
        class D; def each; yield 3; end; end
        class Unseen; def each; yield 4; end; end
        def test(o) = o.each { |x| x * 10 }
        test C.new; test D.new; test C.new; test D.new
    ");
    assert_snapshot!(assert_compiles_allowing_exits("[test(C.new), test(D.new), test(Unseen.new)]"), @"[20, 30, 40]");
}

#[test]
fn test_megamorphic_send_chain_dispatches_and_falls_back() {
    // A call site that sees ten receiver classes is megamorphic: the profiled classes get
    // guarded in-line and everything else takes the dynamic send. Both the chained classes
    // and a class the profile never saw must return the right value.
    set_call_threshold(21);
    eval("
        class C0; def foo = 0; end
        class C1; def foo = 1; end
        class C2; def foo = 2; end
        class C3; def foo = 3; end
        class C4; def foo = 4; end
        class C5; def foo = 5; end
        class C6; def foo = 6; end
        class C7; def foo = 7; end
        class C8; def foo = 8; end
        class C9; def foo = 9; end
        class Unseen; def foo = 42; end
        def test(o) = o.foo
        OBJS = [C0.new, C1.new, C2.new, C3.new, C4.new, C5.new, C6.new, C7.new, C8.new, C9.new]
        3.times { OBJS.each { |o| test o } }
    ");
    assert_snapshot!(assert_compiles("OBJS.map { |o| test o } + [test(Unseen.new)]"), @"[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42]");
}

/// Thirty subclasses of one class, none of which defines the method: the site is megamorphic
/// in the receiver class but every receiver resolves the one inherited method.
const ANCESTOR_GUARD_SETUP: &str = "
    class Base; def foo = 1; end
    class Other; def foo = 99; end
    SUBS = 30.times.map { Class.new(Base) }
    OBJS = SUBS.map(&:new)
    def test(o) = o.foo
    3.times { OBJS.each { |o| test o } }
";

#[test]
fn test_ancestor_guard_dispatches_inherited_method() {
    // A subclass the profile never saw inherits the same method, so it takes the guarded path
    // too; an unrelated class takes the dynamic fallthrough.
    set_call_threshold(21);
    eval(ANCESTOR_GUARD_SETUP);
    assert_snapshot!(assert_compiles("
        [OBJS.map { |o| test o }.uniq, test(Class.new(Base).new), test(Other.new)]
    "), @"[[1], 1, 99]");
}

#[test]
fn test_ancestor_guard_invalidated_by_subclass_override() {
    // Defining the method in a subclass after the guard was compiled has to invalidate it.
    set_call_threshold(21);
    eval(ANCESTOR_GUARD_SETUP);
    assert_snapshot!(assert_compiles_allowing_exits("
        before = OBJS.map { |o| test o }.uniq
        SUBS[0].class_eval { def foo = 100 }
        [before, OBJS.map { |o| test o }.uniq.sort]
    "), @"[[1], [1, 100]]");
}

#[test]
fn test_ancestor_guard_invalidated_by_prepend() {
    // So does prepending a module that defines it below the guarded class.
    set_call_threshold(21);
    eval(ANCESTOR_GUARD_SETUP);
    assert_snapshot!(assert_compiles_allowing_exits("
        before = OBJS.map { |o| test o }.uniq
        SUBS[1].prepend(Module.new { def foo = 200 })
        [before, OBJS.map { |o| test o }.uniq.sort]
    "), @"[[1], [1, 200]]");
}

#[test]
fn test_ancestor_guard_rejects_singleton_receiver() {
    // A singleton method on one instance is invisible to the subclass walk, so the guard
    // itself has to send singleton-class receivers down the dynamic fallthrough.
    set_call_threshold(21);
    eval(ANCESTOR_GUARD_SETUP);
    assert_snapshot!(assert_compiles_allowing_exits("
        o = OBJS[2]
        def o.foo = 300
        OBJS.map { |x| test x }.uniq.sort
    "), @"[1, 300]");
}

#[test]
fn test_ancestor_guard_module_defined_method() {
    // The shared method can come from a module included into the guarded class rather than
    // from the class itself.
    set_call_threshold(21);
    eval("
        module M; def foo = 7; end
        class ModBase; include M; end
        MOD_SUBS = 30.times.map { Class.new(ModBase) }
        MOD_OBJS = MOD_SUBS.map(&:new)
        def test_mod(o) = o.foo
        3.times { MOD_OBJS.each { |o| test_mod o } }
    ");
    assert_snapshot!(assert_compiles("
        [MOD_OBJS.map { |o| test_mod o }.uniq, test_mod(Class.new(ModBase).new)]
    "), @"[[7], 7]");
}

#[test]
fn test_recursive_fact() {
    assert_snapshot!(inspect("
        def fact(n)
          if n == 0
            return 1
          end
          return n * fact(n-1)
        end
        fact(0)
        [fact(0), fact(3), fact(6)]
    "), @"[1, 6, 720]");
}

#[test]
fn test_recursive_fib() {
    assert_snapshot!(inspect("
        def fib(n)
          if n < 2
            return n
          end
          return fib(n-1) + fib(n-2)
        end
        fib(0)
        [fib(0), fib(3), fib(4)]
    "), @"[0, 2, 3]");
}

#[test]
fn test_spilled_basic_block_args() {
    assert_snapshot!(inspect("
        def test(n1, n2)
          n3 = 3
          n4 = 4
          n5 = 5
          n6 = 6
          n7 = 7
          n8 = 8
          n9 = 9
          n10 = 10
          if n1 < n2
            n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10
          end
        end
        test(1, 2)
        test(1, 2)
    "), @"55");
}

// The tests below rotate values between block parameters that the register
// allocator spills. resolve_ssa() used to break such copy cycles through
// SCRATCH_REG, which {arch}_scratch_split also uses to stage memory-to-memory
// Movs, so the parked value was clobbered and the rotation degenerated into a
// broadcast of one element. See Assembler::parcopy_spare().

#[test]
fn test_spilled_block_param_swap() {
    assert_snapshot!(inspect("
        def test(a, b, n)
          i = 0
          while i < n
            a, b = b, a
            i += 1
          end
          [a, b]
        end
        test(10, 20, 4)
        test(10, 20, 4)
        test(10, 20, 5)
    "), @"[20, 10]");
}

#[test]
fn test_spilled_block_param_swap_many_locals() {
    assert_snapshot!(inspect("
        def test(a, b, n)
          c = 3
          d = 4
          e = 5
          f = 6
          g = 7
          h = 8
          i = 0
          while i < n
            a, b = b, a
            i += 1
          end
          [a, b, c, d, e, f, g, h]
        end
        test(10, 20, 4)
        test(10, 20, 4)
        test(10, 20, 4)
    "), @"[10, 20, 3, 4, 5, 6, 7, 8]");
}

#[test]
fn test_spilled_block_param_rotate3() {
    assert_snapshot!(inspect("
        def test(a, b, c, n)
          i = 0
          while i < n
            a, b, c = b, c, a
            i += 1
          end
          [a, b, c]
        end
        test(1, 2, 3, 4)
        test(1, 2, 3, 4)
        test(1, 2, 3, 4)
    "), @"[2, 3, 1]");
}

#[test]
fn test_spilled_block_param_rotate3_reverse() {
    assert_snapshot!(inspect("
        def test(a, b, c, n)
          i = 0
          while i < n
            a, b, c = c, a, b
            i += 1
          end
          [a, b, c]
        end
        test(1, 2, 3, 4)
        test(1, 2, 3, 4)
        test(1, 2, 3, 4)
    "), @"[3, 1, 2]");
}

#[test]
fn test_spilled_block_param_swap_with_fixed_third() {
    assert_snapshot!(inspect("
        def test(a, b, c, n)
          i = 0
          while i < n
            a, b, c = b, a, c
            i += 1
          end
          [a, b, c]
        end
        test(1, 2, 3, 5)
        test(1, 2, 3, 5)
        test(1, 2, 3, 5)
    "), @"[2, 1, 3]");
}

#[test]
fn test_spilled_block_param_rotate4() {
    assert_snapshot!(inspect("
        def test(a, b, c, d, n)
          i = 0
          while i < n
            a, b, c, d = b, c, d, a
            i += 1
          end
          [a, b, c, d]
        end
        test(1, 2, 3, 4, 5)
        test(1, 2, 3, 4, 5)
        test(1, 2, 3, 4, 5)
    "), @"[2, 3, 4, 1]");
}

#[test]
fn test_spilled_block_param_rotate5() {
    assert_snapshot!(inspect("
        def test(a, b, c, d, e, n)
          i = 0
          while i < n
            a, b, c, d, e = e, a, b, c, d
            i += 1
          end
          [a, b, c, d, e]
        end
        test(1, 2, 3, 4, 5, 7)
        test(1, 2, 3, 4, 5, 7)
        test(1, 2, 3, 4, 5, 7)
    "), @"[4, 5, 1, 2, 3]");
}

#[test]
fn test_spilled_block_param_rotate_then_swap() {
    assert_snapshot!(inspect("
        def test(a, b, c, d, n)
          i = 0
          while i < n
            a, b, c = b, c, a
            c, d = d, c
            i += 1
          end
          [a, b, c, d]
        end
        test(1, 2, 3, 4, 3)
        test(1, 2, 3, 4, 3)
        test(1, 2, 3, 4, 3)
    "), @"[4, 1, 2, 3]");
}

#[test]
fn test_spilled_block_param_swap_with_early_exit() {
    assert_snapshot!(inspect("
        def test(a, b, n)
          i = 0
          while i < n
            a, b = b, a
            return [a, b, :early] if i == 2
            i += 1
          end
          [a, b]
        end
        test(10, 20, 6)
        test(10, 20, 6)
        test(10, 20, 6)
    "), @"[20, 10, :early]");
}

#[test]
fn test_spilled_block_param_conditional_swap() {
    assert_snapshot!(inspect("
        def test(a, b, n)
          i = 0
          while i < n
            a, b = b, a if i.even?
            i += 1
          end
          [a, b]
        end
        test(10, 20, 5)
        test(10, 20, 5)
        test(10, 20, 5)
    "), @"[20, 10]");
}

#[test]
fn test_spilled_block_param_swap_with_call() {
    assert_snapshot!(inspect("
        def test(a, b, n)
          i = 0
          while i < n
            a, b = b, a
            a = a.itself
            i += 1
          end
          [a, b]
        end
        test(10, 20, 4)
        test(10, 20, 4)
        test(10, 20, 4)
    "), @"[10, 20]");
}

#[test]
fn test_putself() {
    assert_snapshot!(inspect("
        class Integer
          def minus(a)
            self - a
          end
        end
        5.minus(2)
        5.minus(2)
    "), @"3");
}

#[test]
fn test_getinstancevariable_nil() {
    assert_snapshot!(inspect("
        def test() = @foo
        test()
        test()
    "), @"nil");
}

#[test]
fn test_getinstancevariable() {
    assert_snapshot!(inspect("
        @foo = 3
        def test() = @foo
        test()
        test()
    "), @"3");
}

#[test]
fn test_getinstancevariable_miss() {
    assert_snapshot!(inspect("
        class C
          def foo
            @foo
          end
          def foo_then_bar
            @foo = 1
            @bar = 2
          end
          def bar_then_foo
            @bar = 3
            @foo = 4
          end
        end
        o1 = C.new
        o1.foo_then_bar
        result = []
        result << o1.foo
        result << o1.foo
        o2 = C.new
        o2.bar_then_foo
        result << o2.foo
        result
    "), @"[1, 1, 4]");
}

#[test]
fn test_setinstancevariable() {
    assert_snapshot!(inspect("
        def test() = @foo = 1
        test()
        test()
        @foo
    "), @"1");
}

#[test]
fn test_polymorphic_setinstancevariable_with_shape_transitions() {
    set_call_threshold(3);
    assert_snapshot!(inspect(r#"
        class C
          def set(value) = @a = value
        end

        normal = C.new
        with_b = C.new
        with_b.instance_variable_set(:@b, true)
        normal.set(:profile_normal)
        with_b.set(:profile_with_b)

        normal = C.new
        with_b = C.new
        with_b.instance_variable_set(:@b, true)
        results = [normal.set(:normal), with_b.set(:with_b)]
        results << normal.instance_variable_get(:@a)
        results << with_b.instance_variable_get(:@a)
    "#), @"[:normal, :with_b, :normal, :with_b]");
}

#[test]
fn test_getclassvariable() {
    assert_snapshot!(inspect("
        class Foo
          def self.test = @@x
        end
        Foo.class_variable_set(:@@x, 42)
        Foo.test()
        Foo.test()
    "), @"42");
}

#[test]
fn test_getclassvariable_raises() {
    assert_snapshot!(inspect(r#"
        class Foo
          def self.test = @@x
        end
        begin
          Foo.test
          Foo.test
        rescue NameError => e
          e.message
        end
    "#), @r#""uninitialized class variable @@x in Foo""#);
}

#[test]
fn test_setclassvariable() {
    assert_snapshot!(inspect("
        class Foo
          def self.test = @@x = 42
        end
        Foo.test()
        Foo.test()
        Foo.class_variable_get(:@@x)
    "), @"42");
}

#[test]
fn test_setclassvariable_raises() {
    assert_snapshot!(inspect(r#"
        class Foo
          def self.test = @@x = 42
          freeze
        end
        begin
          Foo.test
          Foo.test
        rescue FrozenError => e
          e.message
        end
    "#), @r#""can't modify frozen Class: Foo""#);
}

#[test]
fn test_attr_reader() {
    eval("
        class C
          attr_reader :foo
          def initialize
            @foo = 4
          end
        end
        def test(c) = c.foo
        test(C.new)
    ");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
    assert_snapshot!(assert_compiles("c = C.new; [test(c), test(c)]"), @"[4, 4]");
}

#[test]
fn test_attr_accessor_getivar() {
    eval("
        class C
          attr_accessor :foo
          def initialize
            @foo = 4
          end
        end
        def test(c) = c.foo
        test(C.new)
    ");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
    assert_snapshot!(assert_compiles("c = C.new; [test(c), test(c)]"), @"[4, 4]");
}

#[test]
fn test_getivar_t_data_then_string() {
    // This is a regression test for a type confusion miscomp where
    // we end up reading the fields object using an offset off of a
    // string, assuming that it has a the same layout as a T_DATA object.
    // At the time of writing the fields object of strings are stored
    // in a global table, out-of-line of each string.
    // The string and the thread end up sharing one shape ID.
    set_call_threshold(2);
    eval(r#"
      module GetThousand
        def test = @var1000
      end
      class Thread
        include GetThousand
      end
      class String
        include GetThousand
      end
      OBJ = Thread.new { }
      OBJ.join
      STR = +''
      (0..1000).each do |i|
        ivar_name = :"@var#{i}"
        OBJ.instance_variable_set(ivar_name, i)
        STR.instance_variable_set(ivar_name, i)
      end
      OBJ.test; OBJ.test # profile and compile for Thread (T_DATA)
    "#);
    assert_snapshot!(assert_compiles_allowing_exits("[STR.test, STR.test]"), @"[1000, 1000]");
}

#[test]
fn test_getivar_t_object_then_string() {
    // This test construct an object and a string that have the same set of ivars.
    // They wouldn't share the same shape ID, though, and we rely on this fact in
    // our guards.
    set_call_threshold(2);
    eval(r#"
      module GetThousand
        def test = @var1000
      end
      class MyObject
        include GetThousand
      end
      class String
        include GetThousand
      end
      OBJ = MyObject.new
      STR = +''
      (0..1000).each do |i|
        ivar_name = :"@var#{i}"
        OBJ.instance_variable_set(ivar_name, i)
        STR.instance_variable_set(ivar_name, i)
      end
      OBJ.test; OBJ.test # profile and compile for MyObject
    "#);
    assert_snapshot!(assert_compiles_allowing_exits("[STR.test, STR.test]"), @"[1000, 1000]");
}

#[test]
fn test_getivar_t_class_then_string() {
    // This is a regression test for a type confusion miscomp where
    // we end up reading the fields object using an offset off of a
    // string, assuming that it has a the same layout as a T_CLASS object.
    // At the time of writing the fields object of strings are stored
    // in a global table, out-of-line of each string.
    // The string and the class end up sharing one shape ID.
    set_call_threshold(2);
    eval(r#"
      module GetThousand
        def test = @var1000
      end
      class MyClass
        extend GetThousand
      end
      class String
        include GetThousand
      end
      STR = +''
      (0..1000).each do |i|
        ivar_name = :"@var#{i}"
        MyClass.instance_variable_set(ivar_name, i)
        STR.instance_variable_set(ivar_name, i)
      end
      p MyClass.test; p MyClass.test # profile and compile for MyClass
      p STR.test
    "#);
    assert_snapshot!(assert_compiles_allowing_exits("[STR.test, STR.test]"), @"[1000, 1000]");
}


#[test]
fn test_attr_accessor_setivar() {
    eval("
        class C
          attr_accessor :foo
          def initialize
            @foo = 4
          end
        end
        def test(c)
          c.foo = 5
          c.foo
        end
        test(C.new)
    ");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
    assert_snapshot!(assert_compiles("c = C.new; [test(c), test(c)]"), @"[5, 5]");
}

#[test]
fn test_attr_writer() {
    eval("
        class C
          attr_writer :foo
          def initialize
            @foo = 4
          end
          def get_foo = @foo
        end
        def test(c)
          c.foo = 5
          c.get_foo
        end
        test(C.new)
    ");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
    assert_snapshot!(assert_compiles("c = C.new; [test(c), test(c)]"), @"[5, 5]");
}

#[test]
fn test_getconstant() {
    eval("
        class Foo
          CONST = 1
        end
        def test(klass)
          klass::CONST
        end
        test(Foo)
    ");
    assert_contains_opcode("test", YARVINSN_getconstant);
    assert_snapshot!(assert_compiles("test(Foo)"), @"1");
}

#[test]
fn test_expandarray_no_splat() {
    eval("
        def test(o)
          a, b = o
          [a, b]
        end
        test [3, 4]
    ");
    assert_contains_opcode("test", YARVINSN_expandarray);
    assert_snapshot!(assert_compiles("test [3, 4]"), @"[3, 4]");
}

#[test]
fn test_expandarray_nil() {
    eval("
        def test
          a, b, c = nil
          [a, b, c]
        end
        test
    ");
    assert_contains_opcode("test", YARVINSN_expandarray);
    // The Ragel-generated parsers in the mail and parser gems start with this, so it has to be
    // compiled without a side exit.
    assert_snapshot!(assert_compiles("test"), @"[nil, nil, nil]");
}

#[test]
fn test_expandarray_scalar() {
    eval("
        def test(o)
          a, b = o
          [a, b]
        end
        test 5
    ");
    assert_contains_opcode("test", YARVINSN_expandarray);
    assert_snapshot!(assert_compiles("test 5"), @"[5, nil]");
}

#[test]
fn test_expandarray_short_array() {
    eval("
        def test(o)
          a, b, c = o
          [a, b, c]
        end
        test [1]
        test [2]
    ");
    assert_contains_opcode("test", YARVINSN_expandarray);
    // The array shape guards that the array is long enough, so this exits and recompiles.
    assert_snapshot!(assert_compiles_allowing_exits("test [1]"), @"[1, nil, nil]");
}

#[test]
fn test_expandarray_to_ary() {
    eval("
        class Pair
          def to_ary = [1, 2]
        end
        def test(o)
          a, b, c = o
          [a, b, c]
        end
        test Pair.new
    ");
    assert_contains_opcode("test", YARVINSN_expandarray);
    assert_snapshot!(assert_compiles_allowing_exits("test Pair.new"), @"[1, 2, nil]");
}

#[test]
fn test_expandarray_to_ary_defined_after_compile() {
    // The nil site is compiled for the scalar shape, which still calls rb_check_array_type() at
    // run time, so defining NilClass#to_ary afterwards must be honored rather than folded away.
    assert_snapshot!(inspect("
        def test
          a, b, c = nil
          [a, b, c]
        end
        test
        test
        test
        class NilClass
          def to_ary = [1, 2, 3]
        end
        test
    "), @"[1, 2, 3]");
}

#[test]
fn test_expandarray_converges_from_array_to_nil() {
    // A site profiled as Array that starts seeing nil must recompile rather than exit forever.
    eval("
        def test(o)
          a, b = o
          [a, b]
        end
        20.times { test([1, 2]) }
    ");
    let exits_before = crate::stats::total_exit_count();
    assert_snapshot!(inspect("200.times { test(nil) }; test(nil)"), @"[nil, nil]");
    let exits = crate::stats::total_exit_count() - exits_before;
    assert!(exits < 100, "expected the site to converge, but it exited {exits} times");
}

#[test]
fn test_expandarray_splat() {
    eval("
        def test(o)
          a, *b = o
          [a, b]
        end
        test [3, 4]
    ");
    assert_contains_opcode("test", YARVINSN_expandarray);
    assert_snapshot!(assert_compiles_allowing_exits("test [3, 4]"), @"[3, [4]]");
}

#[test]
fn test_expandarray_splat_post() {
    eval("
        def test(o)
          a, *b, c = o
          [a, b, c]
        end
        test [3, 4, 5]
    ");
    assert_contains_opcode("test", YARVINSN_expandarray);
    assert_snapshot!(assert_compiles_allowing_exits("test [3, 4, 5]"), @"[3, [4], 5]");
}

#[test]
fn test_constant_invalidation() {
    eval("
        class C; end
        def test = C
        test
        test
        C = 123
    ");
    assert_contains_opcode("test", YARVINSN_opt_getconstant_path);
    assert_snapshot!(assert_compiles("test"), @"123");
}

#[test]
fn test_constant_path_invalidation() {
    eval("
        module A
          module B; end
        end
        module Foo
          C = 'Foo::C'
        end
        A::B = Foo
        def test = A::B::C
    ");
    assert_contains_opcode("test", YARVINSN_opt_getconstant_path);
    assert_snapshot!(assert_compiles(r#"
        module A
          module B; end
        end
        module Foo
          C = "Foo::C"
        end
        module Bar
          C = "Bar::C"
        end
        A::B = Foo
        def test = A::B::C
        result = []
        result << test
        result << test
        A::B = Bar
        result << test
        result
    "#), @r#"["Foo::C", "Foo::C", "Bar::C"]"#);
}

#[test]
fn test_dupn() {
    eval("
        def test(array) = (array[1, 2] ||= :rhs)
        test([1, 1])
    ");
    assert_contains_opcode("test", YARVINSN_dupn);
    assert_snapshot!(assert_compiles_allowing_exits("
        one = [1, 1]
        start_empty = []
        [test(one), one, test(start_empty), start_empty]
    "), @"[[1], [1, 1], :rhs, [nil, :rhs]]");
}

#[test]
fn test_bop_invalidation() {
    assert_snapshot!(inspect(r#"
        def test
          eval("class Integer; def +(_) = 100; end")
          1 + 2
        end
        test
        test
    "#), @"100");
}

#[test]
fn test_defined_with_defined_values() {
    eval("
        class Foo; end
        def bar; end
        $ruby = 1
        def test = [defined?(Foo), defined?(bar), defined?($ruby)]
        test
    ");
    assert_contains_opcode("test", YARVINSN_defined);
    assert_snapshot!(assert_compiles("test"), @r#"["constant", "method", "global-variable"]"#);
}

#[test]
fn test_defined_with_undefined_values() {
    eval("
        def test = [defined?(FooUndef), defined?(bar_undef), defined?($ruby_undef)]
        test
    ");
    assert_contains_opcode("test", YARVINSN_defined);
    assert_snapshot!(assert_compiles("test"), @"[nil, nil, nil]");
}

#[test]
fn test_defined_with_method_call() {
    eval(r#"
        def test = [defined?("x".reverse(1)), defined?("x".reverse(1).reverse)]
        test
    "#);
    assert_contains_opcode("test", YARVINSN_defined);
    assert_snapshot!(assert_compiles(r#"test"#), @r#"["method", nil]"#);
}

#[test]
fn test_defined_method_raise() {
    assert_snapshot!(inspect(r#"
        class C
          def assert_equal expected, actual
            if expected != actual
              raise "NO"
            end
          end
          def test_defined_method
            assert_equal(nil, defined?("x".reverse(1).reverse))
          end
        end
        c = C.new
        result = []
        result << c.test_defined_method
        result << c.test_defined_method
        result << c.test_defined_method
        result
    "#), @"[nil, nil, nil]");
}

#[test]
fn test_defined_yield() {
    eval("
        def test = defined?(yield)
    ");
    assert_contains_opcode("test", YARVINSN_defined);
    assert_snapshot!(assert_compiles("[test, test, test{}]"), @r#"[nil, nil, "yield"]"#);
}

#[test]
fn test_defined_yield_from_block() {
    assert_snapshot!(inspect("
        def test
          yield_self { yield_self { defined?(yield) } }
        end
        [test, test, test{}]
    "), @r#"[nil, nil, "yield"]"#);
}

#[test]
fn test_block_given_p() {
    assert_snapshot!(inspect("
        def test = block_given?
        [test, test, test{}]
    "), @"[false, false, true]");
}

#[test]
fn test_block_given_p_from_block() {
    assert_snapshot!(inspect("
        def test
          yield_self { yield_self { block_given? } }
        end
        [test, test, test{}]
    "), @"[false, false, true]");
}

#[test]
fn test_invokeblock_without_block_after_jit_call() {
    assert_snapshot!(inspect(r#"
        def test(*arr, &b)
          arr.class
          yield
        end
        test { }
        begin
          test
        rescue => e
          e.message
        end
    "#), @r#""no block given (yield)""#);
}

#[test]
fn test_putspecialobject_vm_core_and_cbase() {
    eval("
        def test
          alias bar test
          10
        end
        test
    ");
    assert_contains_opcode("test", YARVINSN_putspecialobject);
    assert_snapshot!(assert_compiles("bar"), @"10");
}

#[test]
fn test_putspecialobject_const_base() {
    assert_snapshot!(inspect("
        Foo = 1
        def test = Foo
        test
        test
    "), @"1");
}

#[test]
fn test_branchnil() {
    eval("
        def test(x)
          x&.succ
        end
        test(0)
    ");
    assert_contains_opcode("test", YARVINSN_branchnil);
    assert_snapshot!(assert_compiles("[test(1), test(nil)]"), @"[2, nil]");
}

#[test]
fn test_nil_nil() {
    eval("
        def test = nil.nil?
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles("test"), @"true");
}

#[test]
fn test_non_nil_nil() {
    eval("
        def test = 1.nil?
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles("test"), @"false");
}

#[test]
fn test_getspecial_last_match() {
    eval(r#"
        def test(str)
          str =~ /hello/
          $&
        end
        test("hello world")
    "#);
    assert_contains_opcode("test", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles(r#"test("hello world")"#), @r#""hello""#);
}

#[test]
fn test_getspecial_match_pre() {
    eval(r#"
        def test(str)
          str =~ /world/
          $`
        end
        test("hello world")
    "#);
    assert_contains_opcode("test", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles(r#"test("hello world")"#), @r#""hello ""#);
}

#[test]
fn test_getspecial_match_post() {
    eval(r#"
        def test(str)
          str =~ /hello/
          $'
        end
        test("hello world")
    "#);
    assert_contains_opcode("test", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles(r#"test("hello world")"#), @r#"" world""#);
}

#[test]
fn test_getspecial_match_last_group() {
    eval(r#"
        def test(str)
          str =~ /(hello) (world)/
          $+
        end
        test("hello world")
    "#);
    assert_contains_opcode("test", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles(r#"test("hello world")"#), @r#""world""#);
}

#[test]
fn test_getspecial_numbered_match_1() {
    eval(r#"
        def test(str)
          str =~ /(hello) (world)/
          $1
        end
        test("hello world")
    "#);
    assert_contains_opcode("test", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles(r#"test("hello world")"#), @r#""hello""#);
}

#[test]
fn test_getspecial_numbered_match_2() {
    eval(r#"
        def test(str)
          str =~ /(hello) (world)/
          $2
        end
        test("hello world")
    "#);
    assert_contains_opcode("test", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles(r#"test("hello world")"#), @r#""world""#);
}

#[test]
fn test_getspecial_numbered_match_nonexistent() {
    eval(r#"
        def test(str)
          str =~ /(hello)/
          $2
        end
        test("hello world")
    "#);
    assert_contains_opcode("test", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles(r#"test("hello world")"#), @"nil");
}

#[test]
fn test_getspecial_no_match() {
    eval(r#"
        def test(str)
          str =~ /xyz/
          $&
        end
        test("hello world")
    "#);
    assert_contains_opcode("test", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles(r#"test("hello world")"#), @"nil");
}

#[test]
fn test_getspecial_complex_pattern() {
    eval(r#"
        def test(str)
          str =~ /(\d+)/
          $1
        end
        test("abc123def")
    "#);
    assert_contains_opcode("test", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles(r#"test("abc123def")"#), @r#""123""#);
}

#[test]
fn test_getspecial_multiple_groups() {
    eval(r#"
        def test(str)
          str =~ /(\d+)-(\d+)/
          $2
        end
        test("123-456")
    "#);
    assert_contains_opcode("test", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles(r#"test("123-456")"#), @r#""456""#);
}

// In a JIT-to-JIT call, the callee's cfp->jit_return is published at entry.
// Putting $& as the first C call in the callee exercises CFP_ZJIT_FRAME before
// gen_save_pc_for_gc has a chance to update the entry JITFrame.
#[test]
fn test_getspecial_symbol_in_jit_to_jit_callee() {
    eval(r#"
        def callee = $&
        def caller_method = callee

        # Warm up callee so it JITs
        callee
        callee

        caller_method
        caller_method
    "#);
    assert_contains_opcode("callee", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles("caller_method"), @"nil");
}

// Same JIT-to-JIT setup, exercising gen_getspecial_number ($N).
#[test]
fn test_getspecial_number_in_jit_to_jit_callee() {
    eval(r#"
        def callee = $1
        def caller_method = callee

        callee
        callee

        caller_method
        caller_method
    "#);
    assert_contains_opcode("callee", YARVINSN_getspecial);
    assert_snapshot!(assert_compiles("caller_method"), @"nil");
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "x86_64", target_arch = "aarch64"),
))]
mod signal_profiler {
    use super::*;
    use std::ptr::null_mut;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    use libc::{self, c_int};

    const PROFILE_FRAMES_LIMIT: usize = 128;

    static SAMPLES: AtomicUsize = AtomicUsize::new(0);
    static IN_HANDLER: AtomicBool = AtomicBool::new(false);

    extern "C" fn sample_profile_frames(signum: c_int) {
        if signum != libc::SIGPROF || IN_HANDLER.swap(true, Ordering::Relaxed) {
            return;
        }

        let mut frames = [VALUE(0); PROFILE_FRAMES_LIMIT];
        let mut lines = [0; PROFILE_FRAMES_LIMIT];
        let collected_size = unsafe {
            rb_profile_frames(
                0,
                PROFILE_FRAMES_LIMIT as c_int,
                frames.as_mut_ptr(),
                lines.as_mut_ptr(),
            )
        };
        if collected_size > 0 {
            SAMPLES.fetch_add(1, Ordering::Relaxed);
        }

        IN_HANDLER.store(false, Ordering::Relaxed);
    }

    struct TargetThread(libc::pthread_t);

    // pthread_t is valid to pass to pthread_kill from another thread.
    unsafe impl Send for TargetThread {}

    pub struct Profiler {
        old_sigprof: libc::sigaction,
        stop_sampler: Arc<AtomicBool>,
        sampler: Option<JoinHandle<()>>,
    }

    impl Profiler {
        pub fn start(interval_usec: u64) -> Self {
            assert!(interval_usec > 0);
            SAMPLES.store(0, Ordering::Relaxed);
            IN_HANDLER.store(false, Ordering::Relaxed);

            let mut handler: libc::sigaction = unsafe { std::mem::zeroed() };
            assert_eq!(unsafe { libc::sigemptyset(&mut handler.sa_mask) }, 0, "sigemptyset failed");
            handler.sa_sigaction = sample_profile_frames as *const () as libc::sighandler_t;
            handler.sa_flags = libc::SA_RESTART;

            let mut old_sigprof: libc::sigaction = unsafe { std::mem::zeroed() };
            assert_eq!(
                unsafe { libc::sigaction(libc::SIGPROF, &handler, &mut old_sigprof) },
                0,
                "sigaction failed",
            );

            let target_thread = TargetThread(unsafe { libc::pthread_self() });
            let stop_sampler = Arc::new(AtomicBool::new(false));
            let sampler_stop = Arc::clone(&stop_sampler);
            let interval = Duration::from_micros(interval_usec);
            let sampler = thread::spawn(move || {
                while !sampler_stop.load(Ordering::Relaxed) {
                    unsafe {
                        libc::pthread_kill(target_thread.0, libc::SIGPROF);
                    }
                    thread::sleep(interval);
                }
            });

            Self {
                old_sigprof,
                stop_sampler,
                sampler: Some(sampler),
            }
        }

        pub fn samples(&self) -> usize {
            SAMPLES.load(Ordering::Relaxed)
        }
    }

    impl Drop for Profiler {
        fn drop(&mut self) {
            self.stop_sampler.store(true, Ordering::Relaxed);
            if let Some(sampler) = self.sampler.take() {
                let _ = sampler.join();
            }
            unsafe {
                libc::sigaction(libc::SIGPROF, &self.old_sigprof, null_mut());
            }
        }
    }
}

// Simulate sampling profilers such as stackprof/vernier: a SIGPROF handler
// interrupts a JIT frame and calls rb_profile_frames().
#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "x86_64", target_arch = "aarch64"),
))]
#[test]
fn test_profile_frames_from_signal_handler() {
    eval(r#"
        def profiled_leaf_loop(n)
          i = 0
          while i < n
            i += 1
          end
          i
        end

        # Compile the method before arming the timer so samples land in JIT code.
        profiled_leaf_loop(1)
        profiled_leaf_loop(1)
    "#);

    let profiler = signal_profiler::Profiler::start(100);
    assert_snapshot!(assert_compiles("profiled_leaf_loop(20_000_000)"), @"20000000");
    assert!(profiler.samples() > 0, "rb_profile_frames was not called from SIGPROF handler");
}

// A direct JIT-to-JIT call switches the CFP register before entering the callee.
// Signal profilers must not observe the callee through ec->cfp until the callee's
// cfp->jit_return points at a valid JITFrame.
#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "x86_64", target_arch = "aarch64"),
))]
#[test]
fn test_profile_frames_during_direct_jit_to_jit_entry() {
    with_inlining_threshold(0, || {
        eval(r#"
            def profiled_direct_callee(value)
              value + 1
            end

            def profiled_direct_loop(n)
              i = 0
              value = 0
              while i < n
                value = profiled_direct_callee(value)
                i += 1
              end
              value
            end

            # Compile both methods and patch the caller's SendDirect site before
            # arming the sampler.
            profiled_direct_callee(0)
            profiled_direct_callee(0)
            profiled_direct_loop(1)
            profiled_direct_loop(1)
            profiled_direct_loop(1)
        "#);

        let profiler = signal_profiler::Profiler::start(10);
        assert_snapshot!(assert_compiles("profiled_direct_loop(1_000_000)"), @"1000000");
        assert!(profiler.samples() > 0, "rb_profile_frames was not called from SIGPROF handler");
    });
}

#[test]
fn test_profile_under_nested_jit_call() {
    assert_snapshot!(inspect("
        def profile
          1 + 2
        end
        def jit_call(flag)
          if flag
            profile
          end
        end
        def entry(flag)
          jit_call(flag)
        end
        [entry(false), entry(false), entry(true)]
    "), @"[nil, nil, 3]");
}

#[test]
fn test_bop_redefined() {
    assert_snapshot!(inspect("
        def test
          1 + 2
        end
        test
        [test, Integer.class_eval { def +(_) = 100 }, test]
    "), @"[3, :+, 100]");
}

#[test]
fn test_bop_redefined_with_adjacent_patch_points() {
    assert_snapshot!(inspect("
        def test
          1 + 2 + 3 + 4 + 5
        end
        test
        [test, Integer.class_eval { def +(_) = 100 }, test]
    "), @"[15, :+, 100]");
}

#[test]
fn test_method_redefined_with_top_self() {
    assert_snapshot!(inspect(r#"
        def foo
          "original"
        end
        def test = foo
        test; test
        result1 = test
        def foo
          "redefined"
        end
        result2 = test
        [result1, result2]
    "#), @r#"["original", "redefined"]"#);
}

#[test]
fn test_method_redefined_with_module() {
    assert_snapshot!(inspect(r#"
        module Foo
          def self.foo = "original"
        end
        def test = Foo.foo
        test
        result1 = test
        def Foo.foo = "redefined"
        result2 = test
        [result1, result2]
    "#), @r#"["original", "redefined"]"#);
}

#[test]
fn test_module_name_with_guard_passes() {
    assert_snapshot!(inspect(r#"
        def test(mod)
          mod.name
        end
        test(String)
        test(Integer)
    "#), @r#""Integer""#);
}

#[test]
fn test_module_name_with_guard_side_exit() {
    assert_snapshot!(inspect(r#"
        class MyClass
          def name = "Bar"
        end
        def test(mod)
          mod.name
        end
        results = []
        results << test(String)
        results << test(Integer)
        results << test(MyClass.new)
        results
    "#), @r#"["String", "Integer", "Bar"]"#);
}

#[test]
fn test_objtostring_calls_to_s_on_non_strings() {
    assert_snapshot!(inspect(r##"
        results = []
        class Foo
          def to_s
            "foo"
          end
        end
        def test(str)
          "#{str}"
        end
        results << test(Foo.new)
        results << test(Foo.new)
        results
    "##), @r#"["foo", "foo"]"#);
}

#[test]
fn test_objtostring_rewrite_does_not_call_to_s_on_strings() {
    assert_snapshot!(inspect(r##"
        results = []
        class String
          def to_s
            "bad"
          end
        end
        def test(foo)
          "#{foo}"
        end
        results << test("foo")
        results << test("foo")
        results
    "##), @r#"["foo", "foo"]"#);
}

#[test]
fn test_objtostring_rewrite_does_not_call_to_s_on_string_subclasses() {
    assert_snapshot!(inspect(r##"
        results = []
        class StringSubclass < String
          def to_s
            "bad"
          end
        end
        foo = StringSubclass.new("foo")
        def test(str)
          "#{str}"
        end
        results << test(foo)
        results << test(foo)
        results
    "##), @r#"["foo", "foo"]"#);
}

#[test]
fn test_objtostring_profiled_string_fastpath() {
    assert_snapshot!(inspect(r##"
        def test(str)
          "#{str}"
        end
        test('foo'); test('foo')
    "##), @r#""foo""#);
}

#[test]
fn test_objtostring_profiled_string_subclass_fastpath() {
    assert_snapshot!(inspect(r##"
        class MyString < String; end
        def test(str)
          "#{str}"
        end
        foo = MyString.new("foo")
        test(foo); test(foo)
    "##), @r#""foo""#);
}

#[test]
fn test_objtostring_profiled_string_fastpath_exits_on_nonstring() {
    assert_snapshot!(inspect(r##"
        def test(str)
          "#{str}"
        end
        test('foo')
        test(1)
    "##), @r#""1""#);
}

#[test]
fn test_objtostring_profiled_nonstring_calls_to_s() {
    assert_snapshot!(inspect(r##"
        def test(str)
          "#{str}"
        end
        test([1,2,3]);
        test([1,2,3]);
    "##), @r#""[1, 2, 3]""#);
}

#[test]
fn test_objtostring_profiled_nonstring_guard_exits_when_string() {
    assert_snapshot!(inspect(r##"
        def test(str)
          "#{str}"
        end
        test([1,2,3]);
        test('foo');
    "##), @r#""foo""#);
}

#[test]
fn test_string_bytesize_with_guard() {
    assert_snapshot!(inspect("
        def test(str)
          str.bytesize
        end
        test('hello')
        test('world')
    "), @"5");
}

#[test]
fn test_string_bytesize_multibyte() {
    assert_snapshot!(inspect(r#"
        def test(s)
          s.bytesize
        end
        test("💎")
        test("💎")
    "#), @"4");
}

#[test]
fn test_nil_value_nil_opt_with_guard() {
    eval("
        def test(val) = val.nil?
        test(nil)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles("test(nil)"), @"true");
}

#[test]
fn test_nil_value_nil_opt_with_guard_side_exit() {
    eval("
        def test(val) = val.nil?
        test(nil)
        test(nil)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles_allowing_exits("test(1)"), @"false");
}

#[test]
fn test_true_nil_opt_with_guard() {
    eval("
        def test(val) = val.nil?
        test(true)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles("test(true)"), @"false");
}

#[test]
fn test_true_nil_opt_with_guard_side_exit() {
    eval("
        def test(val) = val.nil?
        test(true)
        test(true)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles_allowing_exits("test(nil)"), @"true");
}

#[test]
fn test_false_nil_opt_with_guard() {
    eval("
        def test(val) = val.nil?
        test(false)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles("test(false)"), @"false");
}

#[test]
fn test_false_nil_opt_with_guard_side_exit() {
    eval("
        def test(val) = val.nil?
        test(false)
        test(false)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles_allowing_exits("test(nil)"), @"true");
}

#[test]
fn test_integer_nil_opt_with_guard() {
    eval("
        def test(val) = val.nil?
        test(1)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles("test(2)"), @"false");
}

#[test]
fn test_integer_nil_opt_with_guard_side_exit() {
    eval("
        def test(val) = val.nil?
        test(1)
        test(2)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles_allowing_exits("test(nil)"), @"true");
}

#[test]
fn test_float_nil_opt_with_guard() {
    eval("
        def test(val) = val.nil?
        test(1.0)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles("test(2.0)"), @"false");
}

#[test]
fn test_float_nil_opt_with_guard_side_exit() {
    eval("
        def test(val) = val.nil?
        test(1.0)
        test(2.0)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles_allowing_exits("test(nil)"), @"true");
}

#[test]
fn test_symbol_nil_opt_with_guard() {
    eval("
        def test(val) = val.nil?
        test(:foo)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles("test(:bar)"), @"false");
}

#[test]
fn test_symbol_nil_opt_with_guard_side_exit() {
    eval("
        def test(val) = val.nil?
        test(:foo)
        test(:bar)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles_allowing_exits("test(nil)"), @"true");
}

#[test]
fn test_class_nil_opt_with_guard() {
    eval("
        def test(val) = val.nil?
        test(String)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles_allowing_exits("test(Integer)"), @"false");
}

#[test]
fn test_class_nil_opt_with_guard_side_exit() {
    eval("
        def test(val) = val.nil?
        test(String)
        test(Integer)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles_allowing_exits("test(nil)"), @"true");
}

#[test]
fn test_module_nil_opt_with_guard() {
    eval("
        def test(val) = val.nil?
        test(Enumerable)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles_allowing_exits("test(Kernel)"), @"false");
}

#[test]
fn test_module_nil_opt_with_guard_side_exit() {
    eval("
        def test(val) = val.nil?
        test(Enumerable)
        test(Kernel)
    ");
    assert_contains_opcode("test", YARVINSN_opt_nil_p);
    assert_snapshot!(assert_compiles_allowing_exits("test(nil)"), @"true");
}

#[test]
fn test_basic_object_guard_works_with_immediate() {
    assert_snapshot!(inspect("
        class Foo; end
        def test(val) = val.class
        test(Foo.new)
        test(Foo.new)
        test(nil)
    "), @"NilClass");
}

#[test]
fn test_basic_object_guard_works_with_false() {
    assert_snapshot!(inspect("
        class Foo; end
        def test(val) = val.class
        test(Foo.new)
        test(Foo.new)
        test(false)
    "), @"FalseClass");
}

#[test]
fn test_string_concat() {
    eval(r##"
        def test = "#{1}#{2}#{3}"
        test
    "##);
    assert_contains_opcode("test", YARVINSN_concatstrings);
    assert_snapshot!(assert_compiles(r##"test"##), @r#""123""#);
}

#[test]
fn test_string_concat_empty() {
    eval(r##"
        def test = "#{}"
        test
    "##);
    assert_contains_opcode("test", YARVINSN_concatstrings);
    assert_snapshot!(assert_compiles(r##"test"##), @r#""""#);
}

#[test]
fn test_regexp_interpolation() {
    eval(r##"
        def test = /#{1}#{2}#{3}/
        test
    "##);
    assert_contains_opcode("test", YARVINSN_toregexp);
    assert_snapshot!(assert_compiles(r##"test"##), @"/123/");
}

#[test]
fn test_once_regexp_interpolated_only_once() {
    eval(r##"
        $once_count = 0
        def test(str) = /#{($once_count += 1; "a".upcase)}b/o =~ str
        test("Ab")
        test("Ab")
    "##);
    assert_contains_opcode("test", YARVINSN_once);
    assert_snapshot!(
        assert_compiles(r##"[test("Ab"), test("xxAb"), test("zz"), $once_count]"##),
        @"[0, 2, nil, 1]"
    );
}

#[test]
fn test_once_caches_the_same_regexp() {
    eval(r##"
        def test = /#{"a"}b/o
        test
        test
    "##);
    assert_contains_opcode("test", YARVINSN_once);
    assert_snapshot!(assert_compiles(r##"[test.equal?(test), test.source]"##), @r#"[true, "ab"]"#);
}

#[test]
fn test_once_reruns_body_after_raise() {
    // vm_once_dispatch() calls vm_once_clear() when the body raises, so the next
    // execution runs the body again.
    eval(r##"
        $once_calls = 0
        def test
          /#{($once_calls += 1; raise "boom" if $once_calls <= 2; "a")}b/o
        rescue => e
          e.message
        end
    "##);
    assert_contains_opcode("test", YARVINSN_once);
    assert_snapshot!(
        assert_compiles_allowing_exits(r##"[test, test, test.source, test.source, $once_calls]"##),
        @r#"["boom", "boom", "ab", "ab", 3]"#
    );
}

#[test]
fn test_new_range_non_leaf() {
    assert_snapshot!(inspect("
        def jit_entry(v) = make_range_then_exit(v)
        def make_range_then_exit(v)
          range = (v..1)
          super rescue range
        end
        jit_entry(0)
        jit_entry(0)
        jit_entry(0/1r)
    "), @"(0/1)..1");
}

#[test]
fn test_raise_in_second_argument() {
    assert_snapshot!(inspect("
        def write(hash, key)
          hash[key] = raise rescue true
          hash
        end
        write({}, :warmup)
        write({}, :ok)
    "), @"{ok: true}");
}

#[test]
fn test_struct_set() {
    assert_snapshot!(inspect("
        C = Struct.new(:foo).new(1)
        def test
          C.foo = Object.new
          42
        end
        r = [test, test]
        C.freeze
        r << begin
          test
        rescue FrozenError
          :frozen_error
        end
    "), @"[42, 42, :frozen_error]");
}

#[test]
fn test_opt_case_dispatch() {
    eval("
        def test(x)
          case x
          when :foo
            true
          else
            false
          end
        end
        test(:warmup)
    ");
    assert_contains_opcode("test", YARVINSN_opt_case_dispatch);
    assert_snapshot!(assert_compiles("[test(:foo), test(1)]"), @"[true, false]");
}

#[test]
fn test_checkmatch_case() {
    eval(r#"
        def test(o)
          case o
          in Integer
            1
          else
            2
          end
        end
    "#);
    assert_contains_opcode("test", YARVINSN_checkmatch);
    assert_snapshot!(assert_compiles(r#"[test(1), test(2), test("3")]"#), @"[1, 1, 2]");
}

#[test]
fn test_checkmatch_case_splat_array() {
    eval(r#"
        def test(o)
          case o
          when *[1, 2]
            1
          else
            2
          end
        end
    "#);
    assert_contains_opcode("test", YARVINSN_checkmatch);
    assert_snapshot!(assert_compiles("[test(1), test(2), test(3)]"), @"[1, 1, 2]");
}

#[test]
fn test_checkmatch_when_splat_array() {
    eval(r#"
        def test
          case
          when *[1, 2]
            1
          else
            2
          end
        end
    "#);
    assert_contains_opcode("test", YARVINSN_checkmatch);
    assert_snapshot!(assert_compiles("[test, test]"), @"[1, 1]");
}

#[test]
fn test_checkmatch_rescue() {
    // Rescue behavior is tested functionally here. It still side-exits because
    // JIT exception handling is not supported yet.
    eval(r#"
        def test
          begin
            raise TypeError
          rescue TypeError
            1
          end
        end
    "#);
    assert_snapshot!(assert_compiles("[test, test]"), @"[1, 1]");
}

#[test]
fn test_checkmatch_rescue_splat_array() {
    eval(r#"
        def test
          begin
            raise TypeError
          rescue *[TypeError, ArgumentError]
            1
          end
        end
    "#);
    assert_snapshot!(assert_compiles("[test, test]"), @"[1, 1]");
}

#[test]
fn test_stack_overflow() {
    assert_snapshot!(inspect("
        def recurse(n)
          return if n == 0
          recurse(n-1)
          nil
        end
        recurse(2)
        recurse(2)
        begin
          recurse(20_000)
        rescue SystemStackError
        end
    "), @"nil");
}

#[test]
fn test_invokeblock() {
    eval("
        def test
          yield
        end
        def entry
          test { 42 }
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @"42");
}

#[test]
fn test_invokeblock_with_args() {
    eval("
        def test(x, y)
          yield x, y
        end
        def entry
          test(1, 2) { |a, b| a + b }
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @"3");
}

#[test]
fn test_invokeblock_no_block_given() {
    eval("
        def test
          yield rescue :error
        end
        test { }
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    // Compiled expecting an ISEQ block; calling with none misses the handler guard and
    // deopts, so the interpreter raises LocalJumpError (rescued to :error).
    assert_snapshot!(assert_compiles_allowing_exits("test"), @":error");
}

#[test]
fn test_invokeblock_multiple_yields() {
    eval("
        def test
          yield 1
          yield 2
          yield 3
        end
        def entry
          results = []
          test { |x| results << x }
          results
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @"[1, 2, 3]");
}

/// `vm_callee_setup_block_arg()` truncates the extra arguments a `yield` passes a block that
/// takes fewer parameters, so the direct dispatch drops them too instead of falling back.
#[test]
fn test_invokeblock_truncates_extra_args() {
    eval("
        def test(a, b, c)
          yield a, b, c
        end
        def entry
          test(1, 2, 3) { |x| x }
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @"1");
}

/// A `yield` of one argument to a block that takes none: `10.times { ... }` in disguise, and
/// by far the most common arity mismatch.
#[test]
fn test_invokeblock_truncates_lone_arg_to_paramless_block() {
    eval("
        def test(a)
          yield a
        end
        def entry
          out = 0
          test(1) { out += 1 }
          out
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @"1");
}

/// The other half of the block arity rule: missing parameters are `nil`, not an ArgumentError.
/// `yield` with no arguments never auto-splats, so this is a static fill.
#[test]
fn test_invokeblock_nil_fills_missing_args() {
    eval("
        def test
          yield
        end
        def entry
          test { |a, b| [a, b] }
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @"[nil, nil]");
}

/// A lone `yield`ed value still auto-splats into a multi-parameter block; the nil-fill must
/// not shadow that rule.
#[test]
fn test_invokeblock_lone_arg_still_autosplats() {
    eval("
        def test(a)
          yield a
        end
        def entry
          test([1, 2]) { |x, y| [x, y] }
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @"[1, 2]");
}

/// A shared `yield` site whose blocks need different reshapes: the polymorphic chain has to
/// adapt per arm, not once for the site.
#[test]
fn test_invokeblock_polymorphic_chain_mixed_arities() {
    set_call_threshold(3);
    eval("
        def test(a, b)
          yield a, b
        end
        def entry
          [test(1, 2) { |x| x },
           test(1, 2) { |x, y, z| [x, y, z] },
           test(1, 2) { :none }]
        end
        entry
        entry
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @"[1, [1, 2, nil], :none]");
}

/// `Proc#call` goes through `rb_optimized_call`, not `invokeblock`, and applies the same
/// arity rules. Both spellings of the same block have to agree.
#[test]
fn test_invokeblock_truncate_matches_proc_call() {
    eval("
        def test(a, b)
          yield a, b
        end
        def entry(pr)
          [test(1, 2) { |x| x }, pr.call(1, 2)]
        end
        entry(proc { |x| x })
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry(proc { |x| x })"), @"[1, 1]");
}

/// `break` out of a block the dispatch reshaped still has to unwind to the method that owns
/// the block, not to the frame that yielded.
#[test]
fn test_invokeblock_truncated_block_with_break() {
    eval("
        def test(a)
          yield a
          :not_reached
        end
        def entry
          test(1) { break :broke }
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("entry"), @":broke");
}

/// `next` inside a truncated block returns from the block, and the yielding method keeps
/// running with that value.
#[test]
fn test_invokeblock_truncated_block_with_next() {
    eval("
        def test(a)
          yield(a) + 1
        end
        def entry
          test(1) { next 10 }
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @"11");
}

/// A non-local `return` out of a reshaped block unwinds past the yielding method.
#[test]
fn test_invokeblock_truncated_block_with_return() {
    eval("
        def test(a)
          yield a
          :not_reached
        end
        def entry
          test(1) { return :returned }
          :also_not_reached
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("entry"), @":returned");
}

/// A block that `break`s is dispatched directly, not through `rb_vm_invokeblock()`. The
/// throw unwinds out of the JIT-pushed block frame, which reports its ISEQ through the
/// JITFrame rather than `cfp->_iseq`.
#[test]
fn test_invokeblock_direct_dispatch_with_break() {
    eval("
        def test
          yield 1
          yield 2
          :not_reached
        end
        def entry
          test { |x| break x * 10 if x == 2 }
        end
        entry; entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("entry"), @"20");
}

/// `break` out of a directly dispatched block nested two `yield`s deep unwinds only out of
/// the `yield` that owns that block, leaving the outer one to run to completion.
#[test]
fn test_invokeblock_direct_dispatch_with_nested_break() {
    eval("
        def test
          yield 1
          :after
        end
        def entry
          test { |a| test { |b| break [:inner, a, b] } }
        end
        entry; entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("entry"), @":after");
}

/// `break` inside a lambda is a `return` from the lambda, not an unwind to the block owner.
#[test]
fn test_invokeblock_direct_dispatch_break_in_lambda() {
    eval("
        def test
          yield 1
        end
        def entry
          l = lambda { break :from_lambda }
          [test { |x| x }, l.call]
        end
        entry; entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("entry"), @"[1, :from_lambda]");
}

/// A `break` whose block outlived the method that created it is still an orphan.
#[test]
fn test_invokeblock_direct_dispatch_orphan_break() {
    eval("
        def make(&b) = b
        def test
          yield 1
        end
        def entry
          test { |x| x }
          pr = make { break :nope }
          begin
            pr.call
          rescue LocalJumpError
            :orphan
          end
        end
        entry; entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("entry"), @":orphan");
}

/// An `ensure` inside a block that `break`s still runs while the throw unwinds out of the
/// JIT-pushed block frame.
#[test]
fn test_invokeblock_direct_dispatch_break_runs_ensure() {
    eval("
        def test
          yield 1
          :not_reached
        end
        def entry
          ran = false
          out = test { |x| begin; break :broke; ensure; ran = true; end }
          [out, ran]
        end
        entry; entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("entry"), @"[:broke, true]");
}

/// A `yield` to a Symbol block sends the Symbol's method to the first yielded argument.
#[test]
fn test_invokeblock_symbol_handler() {
    eval("
        def test(a, b)
          [yield(a), yield(b)]
        end
        def entry
          test(1, 2, &:to_s)
        end
        entry; entry; entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @r#"["1", "2"]"#);
}

/// The extra `yield`ed arguments become the send's arguments, with the first one the receiver.
#[test]
fn test_invokeblock_symbol_handler_with_args() {
    eval("
        def test(a, b)
          yield a, b
        end
        def entry
          test(3, 4, &:+)
        end
        entry; entry; entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @"7");
}

/// A Symbol block only reaches public methods, exactly like `vm_call_symbol()`: a private or
/// protected method is a NoMethodError, not a call.
#[test]
fn test_invokeblock_symbol_handler_visibility() {
    eval("
        class SymVis
          def pub = :pub
          private def priv = :priv
          protected def prot = :prot
        end
        def test(a)
          yield a
        end
        def entry
          out = [test(SymVis.new, &:pub)]
          [:priv, :prot].each do |name|
            begin
              test(SymVis.new, &name)
            rescue NoMethodError => e
              out << e.message.split(' ').first
            end
          end
          out
        end
        entry; entry; entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("entry"), @r#"[:pub, "private", "protected"]"#);
}

/// A Symbol naming no method routes to `method_missing`, as the interpreter's Symbol block does.
#[test]
fn test_invokeblock_symbol_handler_method_missing() {
    eval("
        class SymMM
          def method_missing(name, *args) = [:mm, name]
          def respond_to_missing?(name, priv = false) = true
        end
        def test(a)
          yield a
        end
        def entry
          test(SymMM.new, &:nope)
        end
        entry; entry; entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("entry"), @"[:mm, :nope]");
}

/// A site that yields to more than one Symbol dispatches on each, and anything the chain does
/// not name still goes through the generic `invokeblock`.
#[test]
fn test_invokeblock_symbol_handler_polymorphic() {
    eval("
        def test(a)
          yield a
        end
        def entry
          [test(1, &:to_s), test(1, &:succ), test(1, &:-@)]
        end
        entry; entry; entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles("entry"), @r#"["1", 2, -1]"#);
}

/// Redefining the method a Symbol block names invalidates the compiled dispatch rather than
/// keeping the old target. Each call redefines it again, so the value the compiled code
/// returns can only be right if it re-resolved.
#[test]
fn test_invokeblock_symbol_handler_redefined() {
    eval("
        class SymRedef
          def val = 0
        end
        def test(a)
          yield a
        end
        def entry
          before = test(SymRedef.new, &:val)
          nxt = before + 1
          SymRedef.class_eval { define_method(:val) { nxt } }
          [before, test(SymRedef.new, &:val)]
        end
        entry
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("entry"), @"[1, 2]");
}

/// The iterator is inlined into the caller and its `yield` reshapes the arguments for the
/// block, whose body is then inlined at the yield too. The frame that push lays out has to
/// follow the reshaped arguments, not the interpreter's stack, or the frame the block raises
/// through is one slot too tall. `bootstraptest/test_syntax.rb` catches this as a
/// "Stack consistency error".
#[test]
fn test_inlined_block_at_reshaped_yield_unwinds_correctly() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def bar = raise
            def test
              1.times {
                begin
                  return bar
                rescue
                  :ok
                end
              }
            end
            test
            test
        "), @"1");
    });
}

/// Replacing the block at a site the dispatch specialized for a reshape has to keep giving
/// the new block the arity rules it asks for.
#[test]
fn test_invokeblock_reshape_respecializes_on_new_block() {
    set_call_threshold(2);
    eval("
        def test(a, b)
          yield a, b
        end
        def one = test(1, 2) { |x| x }
        def two = test(1, 2) { |x, y, z| [x, y, z] }
        one
        one
    ");
    assert_contains_opcode("test", YARVINSN_invokeblock);
    assert_snapshot!(assert_compiles_allowing_exits("[one, two]"), @"[1, [1, 2, nil]]");
}

#[test]
fn test_invokeblock_ifunc_map() {
    eval("
        class MyList
          include Enumerable
          def each
            yield 1
            yield 2
            yield 3
          end
        end
        def test = MyList.new.map { |x| x * 2 }
        test
    ");
    assert_snapshot!(assert_compiles("test"), @"[2, 4, 6]");
}

#[test]
fn test_invokeblock_ifunc_kwarg() {
    eval("
        def foo
          yield 1, a: 2
        end
        def test = enum_for(:foo).to_a
        test
    ");
    assert_snapshot!(assert_compiles("test"), @"[[1, {a: 2}]]");
}

#[test]
fn test_ccall_variadic_with_multiple_args() {
    eval("
        def test
          a = []
          a.push(1, 2, 3)
          a
        end
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
    assert_snapshot!(assert_compiles("test"), @"[1, 2, 3]");
}

#[test]
fn test_ccall_variadic_with_no_args() {
    eval("
        def test
          a = [1]
          a.push
        end
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
    assert_snapshot!(assert_compiles("test"), @"[1]");
}

#[test]
fn test_ccall_variadic_with_no_args_causing_argument_error() {
    eval("
        def test
          format
        rescue ArgumentError
          :error
        end
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_send_without_block);
    assert_snapshot!(assert_compiles("test"), @":error");
}

#[test]
fn test_allocating_in_hir_c_method_is() {
    eval("
        def a(f) = test(f)
        def test(f) = (f.new if f)
        def second = third
        def third = nil
        a(nil)
        a(nil)
        class Foo
        def self.new = :k
        end
        second
    ");
    assert_contains_opcode("test", YARVINSN_opt_new);
    assert_snapshot!(assert_compiles_allowing_exits("a(Foo)"), @":k");
}

#[test]
fn test_singleton_class_invalidation_annotated_ccall() {
    assert_snapshot!(inspect("
        def define_singleton(obj, define)
          if define
            [nil].reverse_each do
              class << obj
                def ==(_)
                  true
                end
              end
            end
          end
          false
        end
        def test(define)
          obj = BasicObject.new
          obj == define_singleton(obj, define)
        end
        result = []
        result << test(false)
        result << test(true)
        result
    "), @"[false, true]");
}

#[test]
fn test_singleton_class_invalidation_optimized_variadic_ccall() {
    assert_snapshot!(inspect("
        def define_singleton(arr, define)
          if define
            [nil].reverse_each do
              class << arr
                def push(x)
                  super(x * 1000)
                end
              end
            end
          end
          1
        end
        def test(define)
          arr = []
          val = define_singleton(arr, define)
          arr.push(val)
          arr[0]
        end
        result = []
        result << test(false)
        result << test(true)
        result
    "), @"[1, 1000]");
}

#[test]
fn test_is_a_string_special_case() {
    assert_snapshot!(inspect(r#"
        def test(x)
          x.is_a?(String)
        end
        test("foo")
        [test("bar"), test(1), test(false), test(:foo), test([]), test({})]
    "#), @"[true, false, false, false, false, false]");
}

#[test]
fn test_is_a_array_special_case() {
    assert_snapshot!(inspect("
        def test(x)
          x.is_a?(Array)
        end
        test([])
        [test([1,2,3]), test([]), test(1), test(false), test(:foo), test('foo'), test({})]
    "), @"[true, true, false, false, false, false, false]");
}

#[test]
fn test_is_a_hash_special_case() {
    assert_snapshot!(inspect("
        def test(x)
          x.is_a?(Hash)
        end
        test({})
        [test({:a => 'b'}), test({}), test(1), test(false), test(:foo), test([]), test('foo')]
    "), @"[true, true, false, false, false, false, false]");
}

#[test]
fn test_is_a_hash_subclass() {
    assert_snapshot!(inspect("
        class MyHash < Hash
        end
        def test(x)
          x.is_a?(Hash)
        end
        test({})
        test(MyHash.new)
    "), @"true");
}

#[test]
fn test_is_a_normal_case() {
    assert_snapshot!(inspect(r#"
        class MyClass
        end
        def test(x)
          x.is_a?(MyClass)
        end
        test("a")
        [test(MyClass.new), test("a")]
    "#), @"[true, false]");
}

#[test]
fn test_fixnum_div_zero() {
    eval("
        def test(n)
          n / 0
        rescue ZeroDivisionError => e
          e.message
        end
        test(0)
    ");
    assert_contains_opcode("test", YARVINSN_opt_div);
    assert_snapshot!(assert_compiles_allowing_exits(r#"test(0)"#), @r#""divided by 0""#);
}

#[test]
fn test_invokesuper_with_local_written_by_blockiseq() {
    assert_snapshot!(inspect(r#"
        class A
          def foo = "A"
        end
        class B < A
          def foo
            x = nil
            [nil].each do |_|
              x = super
            end
            x
          end
        end
        def test = B.new.foo
        test
        test
    "#), @r#""A""#);
}

#[test]
fn test_max_iseq_versions() {
    // A version killed by PatchPoint invalidation is replaced rather than counted against
    // the respecialization budget, so the total budget is the sum of the two limits.
    let max_versions = max_iseq_versions() + MAX_INVALIDATION_RECOMPILES as usize;
    eval(&format!("
        TEST = -1
        def test = TEST

        # compile and invalidate MAX+1 times
        i = 0
        while i < {max_versions} + 1
          test; test # compile a version

          Object.send(:remove_const, :TEST)
          TEST = i

          i += 1
        end
    "));

    // It should not exceed MAX_ISEQ_VERSIONS + MAX_INVALIDATION_RECOMPILES
    let iseq = get_method_iseq("self", "test");
    let payload = get_or_create_iseq_payload(iseq);
    assert_eq!(payload.versions.len(), max_versions);

    // The last call should not discard the JIT code
    assert!(matches!(unsafe { payload.versions.last().unwrap().as_ref() }.status, IseqStatus::Compiled(_)));
}

#[test]
fn test_ivar_respecialization_beyond_max_versions() {
    let max_versions = max_iseq_versions();
    // Burn every version on constant invalidation so `read` compiles its last version with
    // `no_side_exits`, freezing a shape dispatch built from a profile that has only seen A.
    eval(&format!("
        TEST = -1
        class Base
          def read = @v + TEST
        end
        class A < Base; def initialize = @v = 1; end
        class B < Base; def initialize = (@w = 0; @v = 2); end

        a = A.new
        i = 0
        while i < {max_versions} + 1
          a.read; a.read
          Object.send(:remove_const, :TEST)
          TEST = i
          i += 1
        end

        # Compile the frozen version while the profile has still only seen A.
        i = 0
        while i < 200
          a.read
          i += 1
        end
    "));
    let iseq = get_instance_method_iseq("Base", "read");
    // Each constant redefinition also earns the ISEQ a replacement version, so the count
    // here is max_versions plus however many of those it took to reach the frozen version.
    let frozen_versions = get_or_create_iseq_payload(iseq).versions.len();
    assert!(frozen_versions >= max_versions, "expected at least {max_versions} versions, got {frozen_versions}");

    // Now feed it a shape the frozen dispatch has no arm for, at a different ivar index. The
    // fallback path samples it and earns the ISEQ an extra version, so it exceeds the plain
    // version limit but stays within the respecialization budget.
    eval("
        b = B.new
        i = 0
        while i < 2000
          b.read
          i += 1
        end
    ");
    let payload = get_or_create_iseq_payload(iseq);
    assert!(payload.ivar_respecializations >= 1, "expected an ivar respecialization");
    assert!(payload.ivar_respecializations <= crate::payload::MAX_IVAR_RESPECIALIZATIONS);
    assert!(payload.versions.len() > frozen_versions);
    assert!(payload.versions.len() <= payload.version_limit());
    assert_snapshot!(assert_compiles_allowing_exits("[A.new.read, B.new.read]"), @"[5, 6]");
}

#[test]
fn test_version_growth_is_bounded_under_invalidation_and_shape_churn() {
    // Three sources of extra versions compose at one ISEQ: an invalidation grant per version a
    // broken PatchPoint kills (constant redefinition here), an ivar respecialization grant per
    // window whose fallback traffic a recompile would specialize (five receiver shapes here),
    // and the plain --zjit-max-versions budget. Each is separately capped, so however long the
    // churn runs the ISEQ must stay inside version_limit() and inside the sum of the caps.
    eval("
        class V0; def initialize = @a = 1; def read = @a; end
        class V1; def initialize = (@b = 0; @a = 2); def read = @a; end
        class V2; def initialize = (@b = 0; @c = 0; @a = 3); def read = @a; end
        class V3; def initialize = (@b = 0; @c = 0; @d = 0; @a = 4); def read = @a; end
        class V4; def initialize = (@b = 0; @c = 0; @d = 0; @e = 0; @a = 5); def read = @a; end
        VOBJS = [V0.new, V1.new, V2.new, V3.new, V4.new]
        VTEST = 0
        def churn(o) = o.read + VTEST
        i = 0
        while i < 4000
          VOBJS.each { |o| churn(o) }
          if i % 10 == 0
            Object.send(:remove_const, :VTEST)
            Object.const_set(:VTEST, i)
          end
          i += 1
        end
    ");
    let payload = get_or_create_iseq_payload(get_method_iseq("self", "churn"));
    let cap = max_iseq_versions()
        + crate::payload::MAX_IVAR_RESPECIALIZATIONS as usize
        + crate::payload::MAX_INVALIDATION_RECOMPILES as usize;
    assert!(payload.versions.len() <= payload.version_limit(),
        "{} versions over the limit of {}", payload.versions.len(), payload.version_limit());
    assert!(payload.version_limit() <= cap, "version_limit {} over the cap of {cap}", payload.version_limit());
    assert_snapshot!(inspect("churn(VOBJS[3])"), @"3994");
}

#[test]
fn test_splatkw_polymorphic_uses_generic_conversion() {
    // A `**kw` site that sees both nil and a Hash has no single shape to guard, so it
    // compiles to the generic conversion instead of a side exit that would end the block.
    assert_snapshot!(inspect("
        def kw(**kw) = kw
        def test(h) = [kw(**h), :after]
        test({a: 1}); test(nil)
        [test(nil), test({b: 2})]
    "), @"[[{}, :after], [{b: 2}, :after]]");
}

#[test]
fn test_splatkw_polymorphic_calls_to_hash() {
    assert_snapshot!(inspect("
        class ToHash; def to_hash = {c: 3}; end
        def kw(**kw) = kw
        def test(h) = kw(**h)
        test({a: 1}); test(nil)
        [test(ToHash.new), (begin; test(1); rescue TypeError; :type_error; end)]
    "), @"[{c: 3}, :type_error]");
}

#[test]
fn test_optional_arguments_side_exit() {
    assert_snapshot!(inspect("
        def test(a = (def foo = nil)) = a
        test
        [test, (undef :foo), test(1)]
    "), @"[:foo, nil, 1]");
}

#[test]
fn test_call_a_forwardable_method() {
    assert_snapshot!(inspect("
        def test_root = forwardable
        def forwardable(...) = Array.[](...)
        test_root
        test_root
    "), @"[]");
}

#[test]
fn test_send_on_heap_object_in_spilled_arg() {
    assert_snapshot!(inspect("
        def entry(a1, a2, a3, a4, a5, a6, a7, a8, a9)
          a9.itself.class
        end
        entry(1, 2, 3, 4, 5, 6, 7, 8, {})
        entry(1, 2, 3, 4, 5, 6, 7, 8, {})
    "), @"Hash");
}

#[test]
fn test_send_caller_splat_arguments() {
    eval("
        def test(a, b) = [a, b]
        def entry(args) = test(*args)
        entry([1, 2])
    ");
    assert_snapshot!(assert_compiles("entry([1, 2])"), @"[1, 2]");
}

#[test]
fn test_send_empty_caller_splat_arguments() {
    eval("
        def test(a = 1) = a
        def entry(args) = test(*args)
        entry([])
    ");
    assert_snapshot!(assert_compiles("entry([])"), @"1");
}

#[test]
fn test_send_caller_splat_arguments_with_positional_prefix() {
    eval("
        def test(a, b, c) = [a, b, c]
        def entry(args) = test(1, *args)
        entry([2, 3])
    ");
    assert_snapshot!(assert_compiles("entry([2, 3])"), @"[1, 2, 3]");
}

#[test]
fn test_send_many_caller_splat_arguments_to_rest_parameter() {
    eval("
        def test(*args) = args.length
        def entry(args) = test(*args)
        entry([1, 2, 3, 4, 5, 6, 7])
    ");
    assert_snapshot!(assert_compiles("entry([1, 2, 3, 4, 5, 6, 7])"), @"7");
}

#[test]
fn test_send_caller_splat_arguments_to_complex_parameters() {
    eval("
        def test(a, b = 2, *rest, z, k: 40) = [a, b, rest, z, k]
        def entry(args) = test(1, *args)
        entry([3, 4, 5])
    ");
    assert_snapshot!(assert_compiles("entry([3, 4, 5])"), @"[1, 3, [4], 5, 40]");
}

#[test]
fn test_send_caller_splat_arguments_with_required_keyword() {
    eval("
        def test(*args, k:) = [args, k]
        def entry(args) = test(*args, k: 40)
        entry([1, 2])
    ");
    assert_snapshot!(assert_compiles("entry([1, 2])"), @"[[1, 2], 40]");
}

#[test]
fn test_send_caller_splat_arguments_with_block_literal() {
    eval("
        def test(*args) = yield args.length
        def entry(args) = test(*args) { |n| n + 4 }
        entry([1, 2, 3])
    ");
    assert_snapshot!(assert_compiles("entry([1, 2, 3])"), @"7");
}

#[test]
fn test_send_caller_splat_length_mismatch_side_exits() {
    eval("
        def test(*args) = args
        def entry(args) = test(*args)
        entry([1, 2])
    ");
    assert_snapshot!(assert_compiles_allowing_exits("entry([1, 2, 3])"), @"[1, 2, 3]");
}

#[test]
fn test_send_caller_splat_with_ruby2_keywords_hash_side_exits() {
    eval("
        def capture(*args) = args
        ruby2_keywords(:capture)
        def test(arg = :default, k: nil) = [arg, k]
        def entry(args) = test(*args)
        entry(capture(k: 1))
    ");
    assert_snapshot!(assert_compiles_allowing_exits("entry(capture(k: 1))"), @"[:default, 1]");
}

#[test]
fn test_send_caller_splat_result_used_by_hash_aset() {
    eval("
        def test(value) = value
        def entry(args)
          hash = {}
          hash[:value] = test(*args)
        end
        entry([1])
    ");
    assert_snapshot!(assert_compiles("entry([2])"), @"2");
}

#[test]
fn test_send_kwarg() {
    assert_snapshot!(inspect("
        def test(a:, b:) = [a, b]
        def entry = test(b: 2, a: 1)
        entry
        entry
    "), @"[1, 2]");
}

#[test]
fn test_spilled_method_args() {
    assert_snapshot!(inspect("
        def foo(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10)
          n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10
        end
        def test
          foo(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        end
        test
        test
    "), @"55");
}

#[test]
fn test_spilled_method_args_first_and_last() {
    assert_snapshot!(inspect("
        def a(n1,n2,n3,n4,n5,n6,n7,n8,n9) = n1+n9
        a(2,0,0,0,0,0,0,0,-1)
        a(2,0,0,0,0,0,0,0,-1)
    "), @"1");
}

#[test]
fn test_spilled_method_args_last() {
    assert_snapshot!(inspect("
        def a(n1,n2,n3,n4,n5,n6,n7,n8) = n8
        a(1,1,1,1,1,1,1,0)
        a(1,1,1,1,1,1,1,0)
    "), @"0");
}

#[test]
fn test_spilled_method_args_self() {
    assert_snapshot!(inspect("
        def a(n1,n2,n3,n4,n5,n6,n7,n8) = self
        a(1,0,0,0,0,0,0,0).to_s
        a(1,0,0,0,0,0,0,0).to_s
    "), @r#""main""#);
}

#[test]
fn test_spilled_param_new_array() {
    assert_snapshot!(inspect("
        def a(n1,n2,n3,n4,n5,n6,n7,n8) = [n8]
        a(0,0,0,0,0,0,0, :ok)
        a(0,0,0,0,0,0,0, :ok)
    "), @"[:ok]");
}

#[test]
fn test_forty_param_method() {
    assert_snapshot!(inspect("
        def foo(_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,n40) = n40
        foo(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1)
        foo(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1)
    "), @"1");
}

#[test]
fn test_toplevel_local_after_eval() {
    assert_snapshot!(inspect("
        a = 1
        b = 2
        eval('b = 3')
        c = 4
        [a, b, c]
    "), @"[1, 3, 4]");
}

#[test]
fn test_send_exit_with_uninitialized_locals() {
    assert_snapshot!(inspect("
        def entry(init)
          function_stub_exit(init)
        end

        def function_stub_exit(init)
          uninitialized_local = 1 if init
          uninitialized_local
        end

        entry(true)
        entry(false)
    "), @"nil");
}

#[test]
fn test_invokebuiltin_dir_glob() {
    assert_snapshot!(inspect(r#"
        def test = Dir.glob(".")
        test
        test
    "#), @r#"["."]"#);
}

#[test]
fn test_profiled_fact() {
    assert_snapshot!(inspect("
        def fact(n)
          if n == 0
            return 1
          end
          return n * fact(n-1)
        end
        fact(1)
        [fact(0), fact(3), fact(6)]
    "), @"[1, 6, 720]");
}

#[test]
fn test_profiled_fib() {
    assert_snapshot!(inspect("
        def fib(n)
          if n < 2
            return n
          end
          return fib(n-1) + fib(n-2)
        end
        fib(3)
        [fib(0), fib(3), fib(4)]
    "), @"[0, 2, 3]");
}

#[test]
fn test_single_ractor_mode_invalidation() {
    assert_snapshot!(inspect(r#"
        C = Object.new

        def test
          C
        rescue Ractor::IsolationError
          "errored but not crashed"
        end

        test
        test

        Ractor.new {
          test
        }.value
    "#), @r#""errored but not crashed""#);
}

#[test]
fn test_ivar_attr_reader_optimization_with_multi_ractor_mode() {
    assert_snapshot!(inspect("
        class Foo
          class << self
            attr_accessor :bar

            def get_bar
              bar
            rescue Ractor::IsolationError
              42
            end
          end
        end

        Foo.bar = []

        def test
          Foo.get_bar
        end

        test
        test

        Ractor.new { test }.value
    "), @"42");
}

#[test]
fn test_ivar_get_with_multi_ractor_mode() {
    assert_snapshot!(inspect("
        class Foo
          def self.set_bar
            @bar = []
          end

          def self.bar
            @bar
          rescue Ractor::IsolationError
            42
          end
        end

        Foo.set_bar

        def test
          Foo.bar
        end

        test
        test

        Ractor.new { test }.value
    "), @"42");
}

#[test]
fn test_ivar_get_with_already_multi_ractor_mode() {
    assert_snapshot!(inspect("
        class Foo
          def self.set_bar
            @bar = []
          end

          def self.bar
            @bar
          rescue Ractor::IsolationError
            42
          end
        end

        Foo.set_bar
        r = Ractor.new {
          Ractor.receive
          Foo.bar
        }

        Foo.bar
        Foo.bar

        r << :go
        r.value
    "), @"42");
}

#[test]
fn test_ivar_set_with_multi_ractor_mode() {
    assert_snapshot!(inspect("
        class Foo
          def self.bar
            _foo = 1
            _bar = 2
            begin
              @bar = _foo + _bar
            rescue Ractor::IsolationError
              42
            end
          end
        end

        def test
          Foo.bar
        end

        test
        test

        Ractor.new { test }.value
    "), @"42");
}

#[test]
fn test_global_tracepoint() {
    assert_snapshot!(inspect("
        def foo = 1

        foo
        foo

        called = false

        tp = TracePoint.new(:return) { |event|
          if event.method_id == :foo
            called = true
          end
        }
        tp.enable do
          foo
        end
        called
    "), @"true");
}

#[test]
fn test_local_tracepoint() {
    assert_snapshot!(inspect("
        def foo = 1

        foo
        foo

        called = false

        tp = TracePoint.new(:return) { |_| called = true }
        tp.enable(target: method(:foo)) do
          foo
        end
        called
    "), @"true");
}

// Regression test: TracePoint return value for methods with rescue that use `return`.
// ZJIT's send fallback uses rb_vm_opt_send_without_block which calls VM_EXEC,
// setting FLAG_FINISH on the callee frame. This changes how throw TAG_RETURN is
// handled, causing the return value to be nil instead of the actual value.
#[test]
fn test_tracepoint_return_value_with_rescue() {
    assert_snapshot!(inspect("
        def f_raise
          raise
        rescue
          return :f_raise_return
        end

        ary = []
        TracePoint.new(:return, :b_return){|tp|
          ary << [tp.event, tp.method_id, tp.return_value]
        }.enable{
          send :f_raise
        }
        ary.pop # last b_return event is not required
        ary
    "), @"[[:return, :f_raise, :f_raise_return]]");
}

// Regression test: polymorphic getivar must not return nil for too-complex shapes.
// Too-complex shapes use hash tables for ivar storage, and rb_shape_get_iv_index()
// doesn't work for them. The polymorphic path must fall through to GetIvar instead.
#[test]
fn test_polymorphic_getivar_complex_shape() {
    // Need threshold >= 3 so both shapes get profiled before compilation
    set_call_threshold(3);
    assert_snapshot!(inspect(r#"
        class C
          def initialize(foo)
            @foo = foo
          end
          def foo = @foo
        end

        # Create a normal object and a too-complex object of the same class
        normal = C.new(:normal)
        complex = C.new(:complex)
        1001.times { |i| complex.instance_variable_set(:"@v#{i}", i) }
        1001.times { |i| complex.remove_instance_variable(:"@v#{i}") }

        # Profile with both shapes before compilation triggers at call 3
        normal.foo  # call 1: profile normal shape
        complex.foo # call 2: profile too-complex shape

        # The too-complex object should still return :complex, not nil
        [normal.foo, complex.foo]
    "#), @"[:normal, :complex]");
}

/// When a method with keyword defaults contains a block that creates a lambda,
/// the lambda causes EP escape, which globally patches NoEPEscape PatchPoints.
/// On subsequent calls the PatchPoint side exit (which uses without_locals())
/// must not leave stale keyword default values in the frame. We solve this by
/// invalidating the ISEQ version on EP escape so the interpreter takes over.
#[test]
fn test_ep_escape_preserves_keyword_default() {
    set_call_threshold(1);
    assert_snapshot!(inspect(r#"
        def target(dumped, additional_methods: [])
          dumped.class
          additional_methods.each { |m| ->{ m } }
          additional_methods
        end

        def forwarder(x, **kwargs)
          target(x, **kwargs)
        end

        5.times { forwarder("z") }
        forwarder("y", additional_methods: [:to_s])
        target("x")
    "#), @"[]");
}

#[test]
fn test_send_block_to_accepts_no_block() {
    // Methods with &nil should raise ArgumentError when called with a block
    assert_snapshot!(inspect("
        def m(a, &nil); a end

        def test
          m(1) {}
        rescue ArgumentError => e
          e.message
        end

        test
        test
    "), @r#""no block accepted""#);
}

#[test]
fn test_send_block_to_method_not_using_block() {
    // Passing a block to a method that doesn't use it should still work correctly.
    // ZJIT falls back to the interpreter for this case so that unused block
    // warnings are properly emitted.
    assert_snapshot!(inspect("
        def m_no_block = 42

        def test
          m_no_block {}
        end

        test
        test
    "), @"42");
}

#[test]
fn test_send_block_unused_warning_emitted_from_jit() {
    // When ZJIT compiles a send with a block as a dynamic dispatch fallback
    // (gen_send -> rb_vm_send), warn_unused_block uses cfp->pc for the dedup
    // key. We save cfp->pc before calling rb_vm_send so the key is stable
    // and won't spuriously collide with prior entries in the dedup table.
    assert_snapshot!(inspect(r#"
        $warnings = []
        module Warning
          def warn(message, category: nil)
            $warnings << message
          end
        end

        def m_unused_block_warn_test = 42

        def test
          $VERBOSE = true
          m_unused_block_warn_test {}
          $warnings.any? { |w| w.include?("may be ignored") }
        end

        test
        test
    "#), @"true");
}

#[test]
fn test_inlined_method_returns_correct_value() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def add_one(x) = x + 1
            def test(n) = add_one(n)

            test(2)
            test(2)
        "), @"3");
    });
}

#[test]
fn test_inlined_method_with_rest_parameter() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def add_rest(*rest) = rest[0] + rest[1]
            def test = add_rest(1, 2)

            test
            test
        "), @"3");
    });
}

#[test]
fn test_inlined_method_deoptimizes_on_redefinition() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(x) = x + 1
            def test(n) = callee(n)

            test(1)
            test(1)

            def callee(x) = x * 100

            test(1)
        "), @"100");
    });
}

#[test]
fn test_inlined_method_survives_compact_between_calls() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(x) = x + 1
            def test(n) = callee(n)

            test(1)
            test(1)

            GC.compact

            test(1)
        "), @"2");
    });
}

#[test]
fn test_inlined_method_survives_compact_during_call() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def trigger_compact = GC.compact
            def callee(x)
              trigger_compact
              x + 1
            end
            def test(n) = callee(n)

            test(1)
            test(1)
        "), @"2");
    });
}

#[test]
fn test_inlined_method_with_required_keyword() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(x, y:) = x + y
            def test(n) = callee(n, y: 10)

            test(2)
            test(2)
        "), @"12");
    });
}

#[test]
fn test_inlined_method_with_optional_keyword_supplied() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(x, y: 100) = x + y
            def test(n) = callee(n, y: 5)

            test(2)
            test(2)
        "), @"7");
    });
}

#[test]
fn test_inlined_method_with_optional_keyword_omitted() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(x, y: 100) = x + y
            def test(n) = callee(n)

            test(2)
            test(2)
        "), @"102");
    });
}

#[test]
fn test_inlined_method_with_reordered_keywords() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(a:, b:) = a - b
            def test = callee(b: 1, a: 10)

            test
            test
        "), @"9");
    });
}

#[test]
fn test_inlined_method_with_keyword_default_using_prior_param() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(x, y: x * 100) = x + y
            def test(n) = callee(n)

            test(2)
            test(2)
        "), @"202");
    });
}

#[test]
fn test_inlined_method_with_invokeblock() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(x)
              yield x
            end
            def test(n)
              callee(n) { |x| x + 2 }
            end

            test(10)
            test(10)
            test(10)
        "), @"12");
    });
}

#[test]
fn test_inlined_method_with_invokeblock_raise_materializes_stack() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def callee = [1, 2, yield]
            def test
              callee { raise }
            rescue
              :rescued
            end

            test
            test
        "), @":rescued");
    });
}

#[test]
fn test_inlined_method_with_block_param() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(x, &block)
              block.call(x)
            end
            def test(n)
              callee(n) { |x| x + 2 }
            end

            test(10)
            test(10)
            test(10)
        "), @"12");
    });
}

#[test]
fn test_inlined_method_that_forwards_block_arg_raise_materializes_stack() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def inner = yield
            def callee(&block) = [1, 2, inner(&block)]
            def test
              callee { raise }
            rescue
              :rescued
            end

            test
            test
        "), @":rescued");
    });
}

#[test]
fn test_inlined_method_that_forwards_block_arg() {
    // The callee captures a literal block in `&block` and forwards it on to `inner`. Both
    // frames inline: the frame `inner` runs in takes the forwarded handler as its specval,
    // so its `yield` reaches the block the outermost caller wrote.
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def inner(x)
              yield x
            end
            def callee(x, &block)
              inner(x, &block)
            end
            def test(n)
              callee(n) { |x| x + 2 }
            end

            test(10)
            test(10)
            test(10)
        "), @"12");
    });
}

#[test]
fn test_inlined_block_arg_callee_yields_to_proc() {
    // A `&blk` argument holding a plain Proc is the callee's block handler verbatim. The
    // inlined frame writes it into its specval, so `yield` inside the inlined body finds it.
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(x) = yield(x)
            def test(x, p) = callee(x, &p)

            doubler = ->(v) { v * 2 }
            200.times { |i| test(i, doubler) }
            test(21, doubler)
        "), @"42");
    });
}

#[test]
fn test_inlined_block_arg_callee_block_given() {
    // A forwarded block-param proxy may be `VM_BLOCK_HANDLER_NONE`, so neither
    // `block_given?` nor `defined?(yield)` may be folded in the inlined callee: both have to
    // read the specval the frame push installed.
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def callee(x)
              [block_given?, defined?(yield) ? :y : :n, block_given? ? yield(x) : :none]
            end
            def test(x, &blk) = callee(x, &blk)

            out = []
            200.times { |i| out = [test(i) { |v| v + 1 }, test(i)] }
            out
        "), @"[[true, :y, 200], [false, :n, :none]]");
    });
}

#[test]
fn test_inlined_block_arg_callee_proc_call_on_block_param() {
    // `getblockparam` inside the inlined callee materializes a Proc out of the frame's
    // specval and stores it back into the frame, which the `MODIFIED_BLOCK_PARAM` flag then
    // makes every later read take from the local instead.
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            def callee(x, &b) = [b.call(x), b.call(x + 1)]
            def test(x, &blk) = callee(x, &blk)

            out = nil
            200.times { |i| out = test(i) { |v| v * 3 } }
            out
        "), @"[597, 600]");
    });
}

#[test]
fn test_inlined_block_arg_callee_non_local_return() {
    // A `return` from the forwarded block unwinds past the inlined callee frame and out of
    // the method the block was written in.
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def callee = yield
            def forward(&b) = callee(&b)
            def test
              forward { return :returned }
              :fell_through
            end

            200.times { test }
            test
        "), @":returned");
    });
}

#[test]
fn test_inlined_block_arg_callee_break() {
    // A `break` out of the forwarded block terminates the call that the block was written
    // at, skipping the rest of both inlined frames.
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def callee = [yield, :not_reached]
            def forward(&b) = callee(&b)
            def test = [forward { break :broke }, :after]

            200.times { test }
            test
        "), @"[:broke, :after]");
    });
}

#[test]
fn test_inlined_block_arg_callee_argument_error() {
    // An arity error raised inside the inlined callee has to materialize the inlined frames
    // so the backtrace and the rescue in the caller both see the real frame chain.
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def strict(a) = yield(a)
            def test(&b)
              strict(1, 2, &b)
            rescue ArgumentError => e
              e.class
            end

            200.times { test { |x| x } }
            test { |x| x }
        "), @"ArgumentError");
    });
}

#[test]
fn test_inlined_block_arg_callee_setblockparam() {
    // Assigning the block parameter inside the inlined callee sets
    // `VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM` on the inlined frame, so `block_given?` still
    // reports the original handler while `b` reads the replacement.
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def callee(&b)
              b = proc { :replaced }
              [block_given?, b.call]
            end
            def test(&blk) = callee(&blk)

            out = []
            200.times { out = [test { :orig }, test] }
            out
        "), @"[[true, :replaced], [false, :replaced]]");
    });
}

#[test]
fn test_inlined_stack_map_materializes_before_rescue() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits(r#"
            def callee(left, receiver, other)
              left + (receiver << other rescue "b")
            end
            def test = callee("a", "x".freeze, "y")

            test
            test
        "#), @r#""ab""#);
    });
}

#[test]
fn test_inlined_method_with_rescue_caught_in_callee() {
    // The callee's begin/rescue catches an exception raised inside the same
    // callee. The runtime exception walker must find the rescue clause via the
    // inlined callee's CFP.
    with_inlining(|| {
        assert_snapshot!(assert_inlines(r#"
            def callee(x)
              begin
                raise "boom" if x.negative?
                0
              rescue
                42
              end
            end
            def test(n) = callee(n)

            test(-1)
            test(-1)
        "#), @"42");
    });
}

#[test]
fn test_inlined_method_with_rescue_caught_in_caller() {
    // The callee re-raises and the caller catches the exception after unwinding
    // the inlined callee frame.
    with_inlining(|| {
        assert_snapshot!(assert_inlines(r#"
            def callee(x)
              raise "boom" if x.negative?
              0
            end
            def test(n)
              begin
                callee(n)
              rescue
                99
              end
            end

            test(-1)
            test(-1)
        "#), @"99");
    });
}

#[test]
fn test_inlined_method_with_ensure_runs_on_propagation() {
    // The ensure ISEQ is compiled as an exception handler entry
    // (body->jit_exception), and it side-exits on the `throw` that re-raises the
    // exception after the ensure body runs.
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits(r##"
            $log = []
            def callee(x)
              begin
                raise "boom" if x.negative?
                $log << "no_raise"
              ensure
                $log << "ensured"
              end
            end
            def test(n)
              begin
                callee(n)
                "no_rescue"
              rescue
                "caught"
              end
            end

            test(-1)
            result = test(-1)
            "#{$log.first}: #{result}"
        "##), @r#""ensured: caught""#);
    });
}

#[test]
fn test_inlined_method_with_retry_resumes_begin_block() {
    // The begin/rescue/retry callee is larger than the default test inline budget,
    // so raise the threshold enough for it to be inlined.
    // The rescue ISEQ is compiled as an exception handler entry
    // (body->jit_exception), and it side-exits on the `throw` that `retry` emits.
    with_inlining_threshold(100, || {
        assert_snapshot!(assert_inlines_allowing_exits(r#"
            def callee(counter)
              begin
                counter[0] += 1
                raise "boom" if counter[0] < 2
                counter[0]
              rescue
                retry
              end
            end
            def test(c) = callee(c)

            test([0])
            test([0])
        "#), @"2");
    });
}

#[test]
fn test_inlined_method_with_super_call() {
    with_inlining(|| {
        assert_snapshot!(assert_inlines("
            class Parent
              def greet = 'hi'
            end

            class Child < Parent
              def greet = super + '!'
            end

            def test(c) = c.greet

            child = Child.new
            test(child)
            test(child)
        "), @r#""hi!""#);
    });
}

#[test]
fn test_inlined_method_with_block_break_across_inlined_boundary() {
    // A `break` from the literal block unwinds to the inlined callee's CFP,
    // where the callee's CATCH_TYPE_BREAK entry must match.
    with_inlining(|| {
        assert_snapshot!(assert_inlines_allowing_exits("
            def callee(arr)
              arr.each do |x|
                break 7 if x > 5
              end
            end
            def test(a) = callee(a)

            test([1, 6, 99])
            test([1, 6, 99])
        "), @"7");
    });
}

#[test]
fn test_load_immediates_into_registers_before_masking() {
    // See https://github.com/ruby/ruby/pull/16669 -- this is a reduced reproduction from a Ruby
    // spec.
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        def test
          klass = Class.new do
            def ===(o)
              true
            end
          end

          case 1
          when klass.new
            :called
          end == :called
        end

        test
        test
    "#), @"true");
}

#[test]
fn test_loop_terminates() {
    set_call_threshold(3);
    // Previous worklist-based type inference only worked for maximal SSA. This is a regression
    // test for hanging.
    assert_snapshot!(inspect(r#"
        class TheClass
          def set_value_loop
            i = 0
            while i < 10
              @levar ||= i
              i += 1
            end
          end
        end

        3.times do |i|
          TheClass.new.set_value_loop
        end
    "#), @"3");
}

// Regression test: getlocal with level=0 after setlocal_WC_0 was loading stale EP
// memory, causing Array#pack with buffer: keyword to receive the wrong buffer VALUE.
// See https://github.com/ruby/ruby/pull/16736
#[test]
fn test_getlocal_level_zero_after_setlocal_wc_0() {
    assert_snapshot!(inspect(r#"
        def test
          b = +"x"
          v = 2
          [v].pack("C*", buffer: b)
          b.size
        end
        test
    "#), @"2");
}

#[test]
fn test_uncached_getconstant_path() {
    set_call_threshold(1);
    eval("
        def test = RUBY_COPYRIGHT
        test
    ");
    assert_contains_opcode("test", YARVINSN_opt_getconstant_path);
    // RUBY_COPYRIGHT is version-dependent, so compare against its runtime value
    // rather than a fixed snapshot.
    assert_eq!(assert_compiles_allowing_exits("test"), inspect("RUBY_COPYRIGHT"));
}

#[test]
fn test_line_tracepoint_on_c_method() {
    set_call_threshold(1);
    eval("nil"); // boot the VM before assert_compiles_allowing_exits touches ZJITState
    assert_snapshot!(assert_compiles_allowing_exits(r#"
        events = []
        events.instance_variable_set(
          :@tp,
          TracePoint.new(:line) { |tp| events << [tp.event, tp.lineno] if tp.path == __FILE__ }
        )
        def events.to_str
          @tp.enable; ''
        end

        # Stay in generated code while enabling tracing
        def events.compiled(obj)
          String(obj)
          @tp.disable; __LINE__
        end

        line = events.compiled(events)
        events[0][-1] = (events[0][-1] == line)

        events.to_s # can't dump events as it's a singleton object AND it has a TracePoint instance variable, which also can't be dumped
    "#), @r#""[[:line, true]]""#);
}

#[test]
fn test_targeted_line_tracepoint_in_c_method_call() {
    set_call_threshold(1);
    eval("nil"); // boot the VM before assert_compiles_allowing_exits touches ZJITState
    assert_snapshot!(assert_compiles_allowing_exits(r#"
        events = []
        events.instance_variable_set(:@tp, TracePoint.new(:line) { |tp| events << tp.lineno })
        def events.to_str
          @tp.enable(target: method(:compiled))
          ''
        end

        # Stay in generated code while enabling tracing
        def events.compiled(obj)
          String(obj)
          __LINE__
        end

        line = events.compiled(events)
        events[0] = (events[0] == line)

        events.to_s # can't dump events as it's a singleton object AND it has a TracePoint instance variable, which also can't be dumped
    "#), @r#""[true]""#);
}

#[test]
fn test_regression_cfp_sp_set_correctly_before_leaf_gc_call() {
    set_call_threshold(14);
    eval("nil"); // boot the VM before assert_compiles_allowing_exits touches ZJITState
    assert_snapshot!(assert_compiles_allowing_exits(r#"
        def check(l, r)
          return 1 unless l
          1 + check(*l) + check(*r)
        end

        def tree(depth)
          # This duparray is our leaf-gc target.
          return [nil, nil] unless depth > 0

          # Modify the local and pass it to the following calls.
          depth -= 1
          [tree(depth), tree(depth)]
        end

        def test
          GC.stress = true
          2.times do
            t = tree(11)
            check(*t)
          end
          :ok
        end

        test
    "#), @":ok");
}

#[test]
fn test_regression_gc_stress_with_lazy_block_code() {
    eval("nil"); // boot the VM before assert_compiles_allowing_exits touches ZJITState
    assert_snapshot!(assert_compiles_allowing_exits(r#"
        def allocate_array
          [1, 2, 3]
        end

        begin
          GC.stress = true
          allocate_array
          allocate_array
          :ok
        ensure
          GC.stress = false
        end
    "#), @":ok");
}

// Hash recursion uses catch/throw internally. The target frame remains in JIT
// code after the caught throw, so longjmp must not materialize and detach it
// before a callee side exit uses its updated PC and stack map.
#[test]
fn test_keep_jit_frame_for_caught_jump() {
    rb_zjit_prepare_options();
    let old_call_threshold = unsafe { crate::options::rb_zjit_call_threshold };
    let old_inline_threshold = get_option!(inline_threshold);
    let old_max_versions = get_option!(max_versions);
    set_call_threshold(1);
    set_inline_threshold(0);
    set_max_versions(2);
    let result = inspect(r#"
        module KeepJITFrameAssertions
          def assert_receiver(*)
            raise unless is_a?(KeepJITFrameBase)
          end
        end

        class KeepJITFrameBase
          include KeepJITFrameAssertions

          def hash_class = Hash

          def test
            hash = hash_class[]
            recursive = [hash]
            hash[:x] = recursive
            object = Object.new
            lookup = { hash => object }

            [recursive, [hash]].each do |key|
              key = { x: key }
              assert_receiver(object, lookup[key], -> { key.inspect })
            end
          end
        end

        class KeepJITFrameHash < Hash
        end

        class KeepJITFrameSubclass < KeepJITFrameBase
          def hash_class = KeepJITFrameHash
        end

        KeepJITFrameBase.new.test
        KeepJITFrameSubclass.new.test
        :ok
    "#);
    set_max_versions(old_max_versions);
    set_inline_threshold(old_inline_threshold);
    set_call_threshold(old_call_threshold);
    assert_snapshot!(result, @":ok");
}

// A NoEPEscape patch point can be reached without the frame's locals ever being
// written to the stack: JIT-to-JIT calls don't write locals, and the code before
// the patch point may be leaf. When an EP escape fires while the version limit
// prevents invalidate_iseq_version() from running, every version containing a
// patched point must still stop receiving calls. Otherwise a fresh call would
// side-exit through the patched point's without_locals() frame state and the
// interpreter would read garbage locals.
#[test]
fn test_no_ep_escape_invalidation_at_max_versions() {
    rb_zjit_prepare_options();
    let old_call_threshold = unsafe { crate::options::rb_zjit_call_threshold };
    let old_max_versions = get_option!(max_versions);
    set_call_threshold(2);
    set_max_versions(1);
    let result = inspect(r#"
        def ep_escape_callee(a = "expected")
          binding if @ep_escape
          a
        end

        def ep_escape_caller = ep_escape_callee

        def ep_escape_dirty(x) = x
        def ep_escape_dirty_caller = ep_escape_dirty(:garbage)

        @ep_escape = nil
        ep_escape_callee; ep_escape_callee    # profile + compile callee
        ep_escape_caller; ep_escape_caller    # profile + compile caller
        @ep_escape = true
        ep_escape_caller                      # binding escapes callee's EP -> invalidation
        @ep_escape = nil
        ep_escape_dirty_caller                # dirty the stale local's stack slot
        ep_escape_caller
    "#);
    set_max_versions(old_max_versions);
    set_call_threshold(old_call_threshold);
    assert_snapshot!(result, @r#""expected""#);
}

#[test]
fn test_float_arithmetic() {
    set_call_threshold(1);
    eval("nil"); // boot the VM before assert_compiles_allowing_exits touches ZJITState
    assert_snapshot!(assert_compiles_allowing_exits("def test = 1.5 + 2.5; test"), @"4.0");
    assert_snapshot!(assert_compiles_allowing_exits("def test = 2.0 * 3.0; test"), @"6.0");
    assert_snapshot!(assert_compiles_allowing_exits("def test = 3.5 - 2.0; test"), @"1.5");
    assert_snapshot!(assert_compiles_allowing_exits("def test = 5.0 / 2.0; test"), @"2.5");
    assert_snapshot!(assert_compiles_allowing_exits("def test = 1.5 * 3; test"), @"4.5"); // Float * Fixnum
    assert_snapshot!(assert_compiles_allowing_exits("def test = (Float::NAN + 1.0).nan?; test"), @"true");
    assert_snapshot!(assert_compiles_allowing_exits("def test = Float::INFINITY * 2.0; test"), @"Infinity");
    assert_snapshot!(assert_compiles_allowing_exits("def test = 3.7.to_i; test"), @"3");
    assert_snapshot!(assert_compiles_allowing_exits("def test = (-2.9).to_i; test"), @"-2");
}

#[test]
fn test_send_backtrace() {
    eval("nil"); // boot the VM before assert_compiles_allowing_exits touches ZJITState
    assert_snapshot!(assert_compiles_allowing_exits(r#"
        def jit_frame2 = caller     # 1
        def jit_frame1 = jit_frame2 # 2
        def entry = jit_frame1      # 3
        entry # profile send        # 4
        entry                       # 5
    "#), @r#"["<compiled>:3:in 'Object#jit_frame1'", "<compiled>:4:in 'Object#entry'", "<compiled>:6:in '<compiled>'", "-e:in 'RubyVM::InstructionSequence#eval'"]"#);
}

// Regression test: when specialized_instruction is disabled (as power_assert does),
// eval'd code uses `send` instead of `opt_send_without_block`, producing SendNoProfiles.
// The `times` call with a literal block is the SendNoProfiles send whose exit profiling
// triggers recompilation of `run`. After recompilation, `make`'s eval("proc { }") crashes
// in vm_make_env_each because the caller frame's EP[-1] (specval) has a stale value.
#[test]
fn test_send_no_profiles_with_disabled_specialized_instruction() {
    set_call_threshold(1);
    assert_snapshot!(inspect(r#"
        RubyVM::InstructionSequence.compile_option = { specialized_instruction: false }
        eval <<~'INNERRUBY'
          def make = eval("proc { }")
          def run(n) = n.times { make }
        INNERRUBY
        run(6)
        :ok
    "#), @":ok");
}

/// Thirty unrelated classes, each defining its own `value`, called from one site: the site is
/// megamorphic in the receiver class *and* in the method it resolves to, so neither a class
/// guard chain nor an ancestor guard can cover it and every call takes the dynamic fallthrough
/// that the send class table sits in front of. See [`crate::send_cache`].
const SEND_CACHE_SETUP: &str = r#"
    KLASSES = 30.times.map { |i| k = Class.new; k.class_eval("def value; #{i}; end"); k }
    OBJS = KLASSES.map(&:new)
    def test(o) = o.value
    3.times { OBJS.each { |o| test o } }
"#;

/// Run `program` and assert the send class table served at least one call while it ran.
#[track_caller]
fn assert_send_cache_hits(program: &str) -> String {
    let before = crate::state::ZJITState::get_counters().send_cache_hit;
    let result = inspect(program);
    let after = crate::state::ZJITState::get_counters().send_cache_hit;
    assert!(after > before, "expected the send class table to serve a call, but the counter did not move");
    result
}

#[test]
fn test_send_cache_dispatches_every_class_at_a_megamorphic_site() {
    // The baseline: each receiver must reach its own class's method, both for the classes the
    // table has warmed and for one it has never seen.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(SEND_CACHE_SETUP);
    assert_snapshot!(assert_send_cache_hits(r#"
        k = Class.new; k.class_eval("def value; 99; end")
        [OBJS.map { |o| test o }.sum, test(k.new)]
    "#), @"[435, 99]");
}

/// A site with far more classes than a fresh table has slots, so that it evicts
/// its way past the growth threshold while the program runs.
const SEND_CACHE_THRASHING_SETUP: &str = r#"
    KLASSES = 300.times.map { |i| k = Class.new; k.class_eval("def value; #{i}; end"); k }
    OBJS = KLASSES.map(&:new)
    def test(o) = o.value
    3.times { OBJS.each { |o| test o } }
"#;

/// Run `program` and assert a send class table outgrew its initial size at some
/// point in this test. The counter is read absolutely rather than as a delta
/// because the warmup that makes a site megamorphic is itself enough thrashing
/// to grow the table, and each test here has the counter to itself.
#[track_caller]
fn assert_send_cache_grows(program: &str) -> String {
    let result = inspect(program);
    let grew = crate::state::ZJITState::get_counters().send_cache_grow_count;
    assert!(grew > 0, "expected the send class table to grow, but it never did");
    result
}

#[test]
fn test_send_cache_grows_and_keeps_dispatching_every_class() {
    // A table starts at 32 slots and is replaced by a bigger one when it thrashes, which
    // moves every slot and drops everything cached in the old table. Both the classes
    // cached before the swap and the ones cached after must still reach their own method:
    // the probe reads the length and the address out of the table's header, so it follows
    // the table rather than reading where the old one used to be.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(SEND_CACHE_THRASHING_SETUP);
    assert_snapshot!(assert_send_cache_grows(r#"
        20.times.map { OBJS.map { |o| test o }.sum }.uniq
    "#), @"[44850]");
}

#[test]
fn test_send_cache_invalidates_a_method_redefined_after_the_table_grew() {
    // Growing the table must not lose the invalidation property: entries in the new table
    // validate themselves against METHOD_ENTRY_INVALIDATED exactly as the old table's did.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(SEND_CACHE_THRASHING_SETUP);
    assert_snapshot!(assert_send_cache_grows(r#"
        before = 20.times.map { OBJS.map { |o| test o }.sum }.uniq
        KLASSES.each { |k| k.class_eval("def value; 1; end") }
        [before, OBJS.map { |o| test o }.sum]
    "#), @"[[44850], 300]");
}

#[test]
fn test_send_cache_dispatches_a_method_redefined_after_it_was_cached() {
    // The invalidation path. Redefining `value` runs rb_clear_method_cache, which sets
    // METHOD_ENTRY_INVALIDATED on the method entry the table cached, and the probe tests
    // exactly that flag -- so the next call re-resolves instead of calling the old body.
    set_call_threshold(61);
    eval(SEND_CACHE_SETUP);
    assert_snapshot!(inspect(r#"
        before = OBJS.map { |o| test o }.sum
        KLASSES.each_with_index { |k, i| k.class_eval("def value; #{i * 100}; end") }
        after = OBJS.map { |o| test o }.sum
        again = OBJS.map { |o| test o }.sum
        [before, after, again]
    "#), @"[435, 43500, 43500]");
}

#[test]
fn test_send_cache_dispatches_a_method_prepended_after_it_was_cached() {
    // prepend inserts an origin iclass above the class, which invalidates the cached method
    // entry the same way a redefinition does.
    set_call_threshold(61);
    eval(SEND_CACHE_SETUP);
    assert_snapshot!(inspect(r#"
        before = OBJS.map { |o| test o }.sum
        pre = Module.new { def value = super + 1000 }
        KLASSES.each { |k| k.prepend(pre) }
        [before, OBJS.map { |o| test o }.sum]
    "#), @"[435, 30435]");
}

#[test]
fn test_send_cache_dispatches_a_method_included_after_it_was_cached() {
    // Including a second module ahead of the one a class already got the method from changes
    // which method the *same* receiver class resolves the name to, with no change to the class
    // itself. The cached entry has to notice.
    set_call_threshold(61);
    eval(r#"
        INNER = Module.new { def value = 1 }
        BASES = 30.times.map { Class.new { include INNER } }
        OBJS = BASES.map(&:new)
        def test(o) = o.value
        3.times { OBJS.each { |o| test o } }
    "#);
    assert_snapshot!(inspect(r#"
        before = OBJS.map { |o| test o }.sum
        outer = Module.new { def value = 7 }
        BASES.each { |k| k.include(outer) }
        [before, OBJS.map { |o| test o }.sum, OBJS.map { |o| test o }.sum]
    "#), @"[30, 210, 210]");
}

#[test]
fn test_send_cache_dispatches_through_a_refinement_activated_after_warmup() {
    // Activating a refinement goes through rb_clear_all_refinement_method_cache, and a
    // refinement call site resolves to a refinement callcache, which the table declines to
    // store at all (it would pin the method entry).
    set_call_threshold(61);
    eval(SEND_CACHE_SETUP);
    assert_snapshot!(inspect(r#"
        before = OBJS.map { |o| test o }.sum
        refined = Module.new do
          KLASSES.each { |k| refine(k) { def value = 5 } }
        end
        after = OBJS.map { |o| test o }.sum
        using refined
        [before, after, OBJS.map { |o| test o }.sum]
    "#), @"[435, 435, 435]");
}

#[test]
fn test_send_cache_raises_after_the_cached_method_is_undefined() {
    // undef_method invalidates the entry, and the fresh lookup produces the empty callcache,
    // which the table refuses to store -- so every later call re-resolves and still raises.
    set_call_threshold(61);
    eval(SEND_CACHE_SETUP);
    assert_snapshot!(inspect(r#"
        before = OBJS.map { |o| test o }.sum
        KLASSES.each { |k| k.send(:undef_method, :value) }
        errors = OBJS.map { |o| begin; test o; rescue NoMethodError; :raised; end }
        [before, errors.uniq, OBJS.map { |o| begin; test o; rescue NoMethodError; :raised; end }.uniq]
    "#), @"[435, [:raised], [:raised]]");
}

#[test]
fn test_send_cache_respects_a_visibility_change_after_warmup() {
    // A visibility change is a method table mutation like any other, so the cached entry is
    // invalidated and the re-resolved call has to fail the permission check.
    set_call_threshold(61);
    eval(SEND_CACHE_SETUP);
    assert_snapshot!(inspect(r#"
        before = OBJS.map { |o| test o }.sum
        KLASSES.each { |k| k.send(:private, :value) }
        [before, OBJS.map { |o| begin; test o; rescue NoMethodError; :private; end }.uniq]
    "#), @"[435, [:private]]");
}

#[test]
fn test_send_cache_sees_a_singleton_method_defined_after_warmup() {
    // Defining a singleton method changes the receiver's class to its singleton class, which
    // is a key the table has never seen rather than a stale hit on the original class.
    set_call_threshold(61);
    eval(SEND_CACHE_SETUP);
    assert_snapshot!(inspect(r#"
        obj = OBJS.first
        before = test(obj)
        obj.define_singleton_method(:value) { 555 }
        [before, test(obj), test(OBJS[1])]
    "#), @"[0, 555, 1]");
}

#[test]
fn test_send_cache_dispatches_method_missing() {
    // A name no receiver defines resolves to the empty callcache every time. The table stores
    // nothing, so this exercises the uncacheable path on every call.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(r#"
        MM_KLASSES = 30.times.map { |i| Class.new { define_method(:method_missing) { |name, *| name } } }
        MM_OBJS = MM_KLASSES.map(&:new)
        def test_mm(o) = o.nope
        3.times { MM_OBJS.each { |o| test_mm o } }
    "#);
    let before = crate::state::ZJITState::get_counters().send_cache_uncacheable;
    assert_snapshot!(inspect("MM_OBJS.map { |o| test_mm o }.uniq"), @"[:nope]");
    let after = crate::state::ZJITState::get_counters().send_cache_uncacheable;
    assert!(after > before, "expected the empty callcache to be reported uncacheable");
}

#[test]
fn test_send_cache_passes_arguments_and_blocks() {
    // Two more call shapes: one with positional arguments, and one with a literal block, which
    // goes through rb_zjit_send_cached rather than its without-block sibling.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(r#"
        ARG_KLASSES = 30.times.map { |i| k = Class.new; k.class_eval(<<~RUBY); k }
            def add(a, b) = a + b + #{i}
            def each_twice = (yield #{i}; yield #{i})
        RUBY
        ARG_OBJS = ARG_KLASSES.map(&:new)
        def test_add(o) = o.add(1, 2)
        def test_block(o) = (n = 0; o.each_twice { |x| n += x }; n)
        3.times { ARG_OBJS.each { |o| test_add o; test_block o } }
    "#);
    assert_snapshot!(assert_send_cache_hits(r#"
        [ARG_OBJS.map { |o| test_add o }.sum, ARG_OBJS.map { |o| test_block o }.sum]
    "#), @"[525, 870]");
}

#[test]
fn test_send_cache_is_shared_by_two_sites_with_the_same_call_shape() {
    // The table is keyed on (name, argc, flags), not on the site, so a second site calling
    // `value` with no arguments reuses the first one's table and starts warm.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(SEND_CACHE_SETUP);
    eval("OBJS.map { |o| test o }");
    let tables_before = crate::state::ZJITState::get_send_caches().len();
    eval(r#"
        def test_other(o) = o.value
        3.times { OBJS.each { |o| test_other o } }
    "#);
    assert_snapshot!(inspect("OBJS.map { |o| test_other o }.sum"), @"435");
    assert_eq!(
        tables_before,
        crate::state::ZJITState::get_send_caches().len(),
        "a second site with the same call shape should reuse the first site's table",
    );
}

#[test]
fn test_send_cache_dispatches_correctly_under_gc_stress() {
    // The table holds callcache pointers, so it has to be marked as a GC root; without that
    // this collects them out from under the probe.
    set_call_threshold(61);
    eval(SEND_CACHE_SETUP);
    assert_snapshot!(inspect(r#"
        GC.stress = true
        sums = 2.times.map { OBJS.map { |o| test o }.sum }
        GC.stress = false
        sums
    "#), @"[435, 435]");
}

#[test]
fn test_send_cache_dispatches_correctly_across_compaction() {
    // Compaction moves the classes the slots are hashed from, so the table drops its entries
    // and refills; the answers must not change either way.
    set_call_threshold(61);
    eval(SEND_CACHE_SETUP);
    assert_snapshot!(inspect(r#"
        before = OBJS.map { |o| test o }.sum
        GC.compact
        middle = OBJS.map { |o| test o }.sum
        GC.compact
        [before, middle, OBJS.map { |o| test o }.sum]
    "#), @"[435, 435, 435]");
}

#[test]
fn test_send_cache_survives_class_churn_and_redefinition() {
    // The combination the individual tests cover one at a time: an undersized table so
    // essentially every call evicts, classes dying under it (so the GC has to invalidate their
    // callcaches rather than leave the table pointing at freed memory), methods redefined and
    // prepended while the site is hot, and a compaction in the middle.
    set_call_threshold(2);
    crate::options::rb_zjit_prepare_options();
    unsafe { crate::options::OPTIONS.as_mut() }.unwrap().send_cache_entries = 8;
    let result = inspect(r#"
        def test(o) = o.value
        errors = 0
        12.times do |round|
          ks = 20.times.map { |i| k = Class.new; k.class_eval("def value; #{i + round * 1000}; end"); k }
          objs = ks.map(&:new)
          objs.each_with_index { |o, i| errors += 1 unless test(o) == i + round * 1000 }
          ks.each_with_index { |k, i| k.class_eval("def value; #{-(i + 1)}; end") }
          objs.each_with_index { |o, i| errors += 1 unless test(o) == -(i + 1) }
          ks.each { |k| k.prepend(Module.new { def value = super * 2 }) }
          objs.each_with_index { |o, i| errors += 1 unless test(o) == -2 * (i + 1) }
          GC.compact if round == 6
          GC.start
        end
        errors
    "#);
    unsafe { crate::options::OPTIONS.as_mut() }.unwrap().send_cache_entries =
        crate::send_cache::DEFAULT_CACHE_ENTRIES;
    assert_snapshot!(result, @"0");
}

#[test]
fn test_send_cache_disabled_still_dispatches_correctly() {
    // --zjit-disable-send-cache has to leave the old path intact, since that is what an A/B
    // measurement compares against.
    set_call_threshold(61);
    crate::options::rb_zjit_prepare_options();
    unsafe { crate::options::OPTIONS.as_mut() }.unwrap().disable_send_cache = true;
    eval(SEND_CACHE_SETUP);
    let result = inspect("OBJS.map { |o| test o }.sum");
    unsafe { crate::options::OPTIONS.as_mut() }.unwrap().disable_send_cache = false;
    assert_snapshot!(result, @"435");
}

#[test]
fn test_send_cache_skips_keyword_argument_sites() {
    // Sites with an explicit keyword list are deliberately excluded from the table, so this
    // exercises the plain interpreter entry point at a megamorphic site.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(r#"
        KW_KLASSES = 30.times.map { |i| k = Class.new; k.class_eval("def kw(a:) = a + #{i}"); k }
        KW_OBJS = KW_KLASSES.map(&:new)
        def test_kw(o) = o.kw(a: 1)
        3.times { KW_OBJS.each { |o| test_kw o } }
    "#);
    let before = crate::state::ZJITState::get_counters().send_cache_hit;
    assert_snapshot!(inspect("KW_OBJS.map { |o| test_kw o }.sum"), @"465");
    assert_eq!(
        before,
        crate::state::ZJITState::get_counters().send_cache_hit,
        "a keyword-argument site should not be routed through the table",
    );
}

/// Like [`SEND_CACHE_SETUP`], but warmed hard enough that the *callees* are compiled too:
/// the inline direct dispatch path only takes over once `ISEQ_BODY(iseq)->jit_entry` exists,
/// so a site whose targets are still interpreted exercises the slow half of the probe.
/// `value` takes no arguments and has no locals, which is the callee shape
/// `zjit_send_cache_direct_cme()` accepts.
const MEGA_DIRECT_SETUP: &str = r#"
    KLASSES = 30.times.map { |i| k = Class.new; k.class_eval("def value; #{i}; end"); k }
    OBJS = KLASSES.map(&:new)
    def test(o) = o.value
    61.times { OBJS.each { |o| test o } }
"#;

/// Run `program` and assert JIT code dispatched at least one send without leaving JIT code.
#[track_caller]
fn assert_megamorphic_direct_hits(program: &str) -> String {
    let before = crate::state::ZJITState::get_counters().send_megamorphic_direct;
    let result = inspect(program);
    let after = crate::state::ZJITState::get_counters().send_megamorphic_direct;
    assert!(after > before, "expected a megamorphic send to be dispatched directly, but the counter did not move");
    result
}

#[test]
fn test_megamorphic_direct_dispatches_every_class() {
    // The baseline: every receiver reaches its own class's method through the inline path, and
    // a class the table has never seen still gets there through the fallback.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(assert_megamorphic_direct_hits(r#"
        k = Class.new; k.class_eval("def value; 99; end")
        [OBJS.map { |o| test o }.sum, test(k.new)]
    "#), @"[435, 99]");
}

#[test]
fn test_megamorphic_direct_passes_arguments() {
    // Arguments are handed over by leaving them where the caller's operand stack already put
    // them, so a wrong frame layout shows up as a wrong argument value rather than a crash.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(r#"
        ADD_KLASSES = 30.times.map { |i| k = Class.new; k.class_eval("def add(a, b, c) = a + b + c + #{i}"); k }
        ADD_OBJS = ADD_KLASSES.map(&:new)
        def test_add(o) = o.add(1, 20, 300)
        61.times { ADD_OBJS.each { |o| test_add o } }
    "#);
    assert_snapshot!(assert_megamorphic_direct_hits(
        "ADD_OBJS.map { |o| test_add o }"
    ), @"[321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350]");
}

#[test]
fn test_megamorphic_direct_after_redefinition() {
    // The invalidation path that matters most: the slot still holds the old callcache, whose
    // method entry rb_clear_method_cache() has flagged. The probe tests that flag before it
    // ever looks at the ISEQ, so the redefined body runs from the very next call.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(assert_megamorphic_direct_hits(r#"
        before = OBJS.map { |o| test o }.sum
        KLASSES.each_with_index { |k, i| k.class_eval("def value; #{i * 100}; end") }
        after = OBJS.map { |o| test o }.sum
        [before, after, OBJS.map { |o| test o }.sum]
    "#), @"[435, 43500, 43500]");
}

#[test]
fn test_megamorphic_direct_after_prepend() {
    // prepend changes which method the same receiver class resolves the name to. The direct
    // path caches no resolution of its own, so this is the callcache's invalidation again.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(assert_megamorphic_direct_hits(r#"
        before = OBJS.map { |o| test o }.sum
        pre = Module.new { def value = super + 1000 }
        KLASSES.each { |k| k.prepend(pre) }
        [before, OBJS.map { |o| test o }.sum]
    "#), @"[435, 30435]");
}

#[test]
fn test_megamorphic_direct_after_include() {
    // Including a module ahead of the one a class got the method from changes the resolution
    // without touching the receiver class, which is the case a class-keyed table could get
    // wrong if it did not consult the method entry.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(r#"
        INNER_D = Module.new { def value = 1 }
        BASES_D = 30.times.map { Class.new { include INNER_D } }
        OBJS_D = BASES_D.map(&:new)
        def test_d(o) = o.value
        61.times { OBJS_D.each { |o| test_d o } }
    "#);
    assert_snapshot!(assert_megamorphic_direct_hits(r#"
        before = OBJS_D.map { |o| test_d o }.sum
        outer = Module.new { def value = 7 }
        BASES_D.each { |k| k.include(outer) }
        [before, OBJS_D.map { |o| test_d o }.sum]
    "#), @"[30, 210]");
}

#[test]
fn test_megamorphic_direct_respects_a_visibility_change() {
    // Making the method private must start raising: the direct path skips vm_call_method(),
    // which is where the permission check lives, so it may only ever take over calls that
    // function would have let through.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(assert_megamorphic_direct_hits(r#"
        before = OBJS.map { |o| test o }.sum
        KLASSES.each { |k| k.send(:private, :value) }
        [before, OBJS.map { |o| begin; test o; rescue NoMethodError; :private; end }.uniq]
    "#), @"[435, [:private]]");
}

#[test]
fn test_megamorphic_direct_declines_protected_methods() {
    // A protected method's check reads the *caller's* self, which nothing in a class-keyed
    // table can stand in for, so those targets stay on the slow path and keep raising for a
    // caller that is not a kind_of the defining class.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(r#"
        PROT_KLASSES = 30.times.map { |i| k = Class.new; k.class_eval("def value; #{i}; end\nprotected :value"); k }
        PROT_OBJS = PROT_KLASSES.map(&:new)
        def test_prot(o) = (begin; o.value; rescue NoMethodError; :protected; end)
        61.times { PROT_OBJS.each { |o| test_prot o } }
    "#);
    let before = crate::state::ZJITState::get_counters().send_megamorphic_direct;
    assert_snapshot!(inspect("PROT_OBJS.map { |o| test_prot o }.uniq"), @"[:protected]");
    assert_eq!(
        before,
        crate::state::ZJITState::get_counters().send_megamorphic_direct,
        "a protected method must not be dispatched directly",
    );
}

#[test]
fn test_megamorphic_direct_after_undef() {
    // undef_method leaves the name resolving to the empty callcache, which the table refuses
    // to store, so every later call re-resolves and reaches method_missing.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(assert_megamorphic_direct_hits(r#"
        before = OBJS.map { |o| test o }.sum
        KLASSES.each { |k| k.send(:undef_method, :value) }
        [before, OBJS.map { |o| begin; test o; rescue NoMethodError; :raised; end }.uniq]
    "#), @"[435, [:raised]]");
}

#[test]
fn test_megamorphic_direct_with_a_refinement_activated_after_warmup() {
    // A refinement call site resolves to a refinement callcache, which the table declines to
    // store; the unrefined site keeps its direct dispatch and the refined one does not get it.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(assert_megamorphic_direct_hits(r#"
        before = OBJS.map { |o| test o }.sum
        refined = Module.new do
          KLASSES.each { |k| refine(k) { def value = 5 } }
        end
        after = OBJS.map { |o| test o }.sum
        m = Module.new
        m.module_eval("using refined; def self.call1(o) = o.value", __FILE__, __LINE__)
        [before, after, m.call1(OBJS[3])]
    "#), @"[435, 435, 5]");
}

#[test]
fn test_megamorphic_direct_sees_a_singleton_method() {
    // Defining a singleton method moves the receiver to a class the table has never seen,
    // rather than leaving a stale hit on the original one.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(assert_megamorphic_direct_hits(r#"
        obj = OBJS.first
        before = test(obj)
        obj.define_singleton_method(:value) { 555 }
        [before, test(obj), test(OBJS[1])]
    "#), @"[0, 555, 1]");
}

#[test]
fn test_megamorphic_direct_under_tracepoint() {
    // Enabling a TracePoint resets every compiled entry point, and the direct path re-reads
    // `body->jit_entry` on every call rather than caching it, so the hook has to fire for the
    // callees too -- 30 of them, once each.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(inspect(r#"
        seen = []
        tp = TracePoint.new(:call) { |t| seen << t.method_id }
        tp.enable
        sum = OBJS.map { |o| test o }.sum
        tp.disable
        [sum, seen.count(:value), OBJS.map { |o| test o }.sum]
    "#), @"[435, 30, 435]");
}

#[test]
fn test_megamorphic_direct_propagates_exceptions() {
    // The direct path pushes no tag of its own, so a raise in the callee unwinds to whatever
    // tag the JIT stack was entered under. Both the callee's own rescue and the caller's have
    // to still find their frame.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(assert_megamorphic_direct_hits(r#"
        KLASSES[5].class_eval("def value; raise 'inner'; rescue; :rescued_inside; end")
        KLASSES[7].class_eval("def value; raise 'outer'; end")
        outer = begin
          OBJS.map { |o| test o }
        rescue => e
          e.message
        end
        [test(OBJS[5]), outer, test(OBJS[9])]
    "#), @r#"[:rescued_inside, "outer", 9]"#);
}

#[test]
fn test_megamorphic_direct_declines_complex_callee_shapes() {
    // Optional parameters, a block parameter and extra locals all take the callee off the
    // direct path, because the frame it pushes is fixed at compile time: the arguments are the
    // whole local area and nothing is nil-filled.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(r#"
        OPT_KLASSES = 30.times.map { |i| k = Class.new; k.class_eval(<<~RUBY); k }
            def opt(a, b = 10) = a + b + #{i}
            def blk(a, &b) = a + #{i}
            def loc(a) = (x = a * 2; y = x + 1; y + #{i})
        RUBY
        OPT_OBJS = OPT_KLASSES.map(&:new)
        def test_opt(o) = o.opt(1)
        def test_blk(o) = o.blk(1)
        def test_loc(o) = o.loc(1)
        61.times { OPT_OBJS.each { |o| test_opt o; test_blk o; test_loc o } }
    "#);
    let before = crate::state::ZJITState::get_counters().send_megamorphic_direct;
    assert_snapshot!(inspect(r#"
        [OPT_OBJS.map { |o| test_opt o }.sum,
         OPT_OBJS.map { |o| test_blk o }.sum,
         OPT_OBJS.map { |o| test_loc o }.sum]
    "#), @"[765, 465, 525]");
    assert_eq!(
        before,
        crate::state::ZJITState::get_counters().send_megamorphic_direct,
        "callees with optionals, a block parameter or extra locals must not be dispatched directly",
    );
}

#[test]
fn test_megamorphic_direct_recurses_without_corrupting_the_stack() {
    // Every direct call pushes a frame from JIT code and pops it on return; a wrong SP or CFP
    // adjustment compounds with depth rather than showing up on the first call.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(r#"
        REC_KLASSES = 30.times.map { |i| k = Class.new; k.class_eval("def rec(n) = n <= 0 ? #{i} : rec(n - 1)"); k }
        REC_OBJS = REC_KLASSES.map(&:new)
        def test_rec(o, n) = o.rec(n)
        61.times { REC_OBJS.each { |o| test_rec(o, 3) } }
    "#);
    assert_snapshot!(assert_megamorphic_direct_hits(
        "[REC_OBJS.map { |o| test_rec(o, 200) }.sum, REC_OBJS.map { |o| test_rec(o, 1) }.sum]"
    ), @"[435, 435]");
}

#[test]
fn test_megamorphic_direct_raises_on_stack_overflow() {
    // The stack overflow check runs before the frame push, against a bound that covers any
    // callee the fill path accepts. Without it a deep enough chain writes past the VM stack.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(r#"
        DEEP_KLASSES = 30.times.map { |i| k = Class.new; k.class_eval("def deep(n) = deep(n + 1)"); k }
        DEEP_OBJS = DEEP_KLASSES.map(&:new)
        def test_deep(o) = o.deep(0)
    "#);
    assert_snapshot!(inspect(r#"
        DEEP_OBJS.map { |o| begin; test_deep(o); rescue SystemStackError; :stack_error; end }.uniq
    "#), @"[:stack_error]");
}

#[test]
fn test_megamorphic_direct_under_gc_stress() {
    // The slot's second word is only reached after its callcache validates, so a class or
    // method entry collected under the probe has to read as a miss, not as a dangling ISEQ.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(inspect(r#"
        GC.stress = true
        sums = 2.times.map { OBJS.map { |o| test o }.sum }
        GC.stress = false
        sums
    "#), @"[435, 435]");
}

#[test]
fn test_megamorphic_direct_across_compaction() {
    // Compaction moves the classes the slots are hashed from, so both words are dropped
    // together and refilled; a slot left half-cleared would pair one class's callcache with
    // another's method entry.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    eval(MEGA_DIRECT_SETUP);
    assert_snapshot!(inspect(r#"
        before = OBJS.map { |o| test o }.sum
        GC.compact
        middle = OBJS.map { |o| test o }.sum
        GC.compact
        [before, middle, OBJS.map { |o| test o }.sum]
    "#), @"[435, 435, 435]");
}

#[test]
fn test_megamorphic_direct_survives_class_churn_and_redefinition() {
    // Everything at once, with an undersized table so nearly every call evicts: classes dying
    // under the site, methods redefined and prepended while it is hot, and a compaction in the
    // middle. Any slot that ever pairs a stale method entry with a live callcache shows up as
    // a wrong answer here.
    crate::options::enable_zjit_stats();
    set_call_threshold(2);
    crate::options::rb_zjit_prepare_options();
    unsafe { crate::options::OPTIONS.as_mut() }.unwrap().send_cache_entries = 8;
    let result = inspect(r#"
        def test_churn(o) = o.value
        errors = 0
        12.times do |round|
          ks = 20.times.map { |i| k = Class.new; k.class_eval("def value; #{i + round * 1000}; end"); k }
          objs = ks.map(&:new)
          4.times { objs.each_with_index { |o, i| errors += 1 unless test_churn(o) == i + round * 1000 } }
          ks.each_with_index { |k, i| k.class_eval("def value; #{-(i + 1)}; end") }
          objs.each_with_index { |o, i| errors += 1 unless test_churn(o) == -(i + 1) }
          ks.each { |k| k.prepend(Module.new { def value = super * 2 }) }
          objs.each_with_index { |o, i| errors += 1 unless test_churn(o) == -2 * (i + 1) }
          GC.compact if round == 6
          GC.start
        end
        errors
    "#);
    unsafe { crate::options::OPTIONS.as_mut() }.unwrap().send_cache_entries =
        crate::send_cache::DEFAULT_CACHE_ENTRIES;
    assert_snapshot!(result, @"0");
}

#[test]
fn test_megamorphic_direct_disabled_still_dispatches_correctly() {
    // --zjit-disable-megamorphic-direct has to leave the C dispatch intact, since that is what
    // an A/B measurement compares against.
    crate::options::enable_zjit_stats();
    set_call_threshold(61);
    crate::options::rb_zjit_prepare_options();
    unsafe { crate::options::OPTIONS.as_mut() }.unwrap().disable_megamorphic_direct = true;
    eval(MEGA_DIRECT_SETUP);
    let before = crate::state::ZJITState::get_counters().send_megamorphic_direct;
    let result = inspect("OBJS.map { |o| test o }.sum");
    let after = crate::state::ZJITState::get_counters().send_megamorphic_direct;
    unsafe { crate::options::OPTIONS.as_mut() }.unwrap().disable_megamorphic_direct = false;
    assert_eq!(before, after, "the direct path should be off");
    assert_snapshot!(result, @"435");
}

/// A Ruby-level iterator, so its `yield` really is the `invokeblock` instruction rather than
/// `rb_yield()` -- only the former reaches `vm_invoke_symbol_block()`, whose semantics the
/// Symbol block arm reproduces. `SYM_MEGA_SETUP` then hands it 60 unrelated receiver classes,
/// which is more than any guard chain covers, so the arm's send has to be dispatched through
/// the class table. See crate::hir::Function::push_symbol_block_mega.
const SYM_MEGA_SETUP: &str = r#"
    def each_of(a)
      i = 0
      out = []
      while i < a.size
        out << (yield a[i])
        i += 1
      end
      out
    end
    SYM_KLASSES = 60.times.map { |i| k = Class.new; k.class_eval("def value; #{i}; end"); k }
    SYM_OBJS = SYM_KLASSES.map(&:new)
    def sym_run(a) = each_of(a, &:value)
    30.times { sym_run(SYM_OBJS) }
"#;

#[test]
fn test_symbol_block_megamorphic_dispatches_every_class() {
    // The baseline for the whole feature: a `yield` to `&:sym` over 60 receiver classes reaches
    // each class's own method, and the dispatch happens inside JIT code rather than through
    // rb_vm_invokeblock().
    crate::options::enable_zjit_stats();
    set_call_threshold(2);
    eval(SYM_MEGA_SETUP);
    let before = crate::state::ZJITState::get_counters().send_megamorphic_direct;
    let result = inspect(r#"
        k = Class.new; k.class_eval("def value; 99; end")
        [sym_run(SYM_OBJS).sum, sym_run(SYM_OBJS + [k.new]).last]
    "#);
    let after = crate::state::ZJITState::get_counters().send_megamorphic_direct;
    assert!(after > before,
        "expected the Symbol block's send to be dispatched out of the class table");
    assert_snapshot!(result, @"[1770, 99]");
}

#[test]
fn test_symbol_block_megamorphic_passes_arguments() {
    // The receiver is the first yielded value and the rest are the send's arguments, which is
    // the layout the operand stack already has -- a wrong frame layout shows up as a wrong
    // argument rather than a crash.
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        def each_of2(a, x, y)
          i = 0; out = []
          while i < a.size
            out << (yield a[i], x, y)
            i += 1
          end
          out
        end
        ks = 60.times.map { |i| k = Class.new; k.class_eval("def add(a, b) = a + b + #{i}"); k }
        objs = ks.map(&:new)
        out = nil
        30.times { out = each_of2(objs, 1, 20, &:add) }
        [out.first, out.last, out.size]
    "#), @"[21, 80, 60]");
}

#[test]
fn test_symbol_block_megamorphic_rejects_private_and_protected() {
    // `vm_call_symbol()` reaches public methods only. A private one is not callable because the
    // `yield`'s call flags never carry VM_CALL_FCALL, and a protected one is rejected outright
    // -- there is no caller-is-an-instance test the way an ordinary send has, so not even a
    // receiver of the caller's own class may reach it.
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        def each_of(a)
          i = 0; out = []
          while i < a.size; out << (yield a[i]); i += 1; end
          out
        end
        klass = Class.new do
          def pub = :pub
          private def priv = :priv
          protected def prot = :prot
          define_method(:from_inside) { |o, sym| each_of([o], &sym) }
        end
        def run(a, sym) = (each_of(a, &sym) rescue [$!.class, $!.message])
        30.times { run([klass.new], :pub) }
        30.times { run([klass.new], :priv) }
        30.times { klass.new.from_inside(klass.new, :prot) rescue nil }
        inside = (klass.new.from_inside(klass.new, :prot) rescue [$!.class, $!.message[0, 14]])
        [run([klass.new], :pub), run([klass.new], :priv).first, run([klass.new], :prot).first, inside]
    "#), @"[[:pub], NoMethodError, NoMethodError, [NoMethodError, \"protected meth\"]]");
}

#[test]
fn test_symbol_block_megamorphic_dispatches_method_missing() {
    // A name no receiver defines is not something the table may answer: it has no method entry
    // to validate, so the call goes to vm_call_symbol(), which builds the method_missing
    // arguments the way the interpreter does.
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        def each_of(a)
          i = 0; out = []
          while i < a.size; out << (yield a[i]); i += 1; end
          out
        end
        mm = Class.new { def method_missing(n, *a) = [:mm, n, a] }
        ks = 60.times.map { |i| k = Class.new; k.class_eval("def value; #{i}; end"); k }
        objs = ks.map(&:new) + [mm.new]
        out = nil
        30.times { out = each_of(objs, &:value) }
        nome = (each_of([Object.new], &:value) rescue $!.class)
        [out.size, out.last, nome]
    "#), @"[61, [:mm, :value, []], NoMethodError]");
}

#[test]
fn test_symbol_block_megamorphic_sees_a_method_redefined_mid_iteration() {
    // Redefinition invalidates the method entry the table cached, which both the inline probe
    // and the C helper reject before dispatching -- so the change lands on the very next
    // element rather than at the next compile.
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        REDEF = Class.new { def v = 1 }
        def each_redef(a)
          i = 0; out = []
          while i < a.size
            out << (yield a[i])
            REDEF.class_eval("def v = 2") if i == 1
            i += 1
          end
          out
        end
        objs = Array.new(4) { REDEF.new }
        30.times { REDEF.class_eval("def v = 1"); each_redef(objs) { |o| o.v } }
        REDEF.class_eval("def v = 1")
        each_redef(objs, &:v)
    "#), @"[1, 1, 2, 2]");
}

#[test]
fn test_symbol_block_megamorphic_raises_after_the_method_is_undefined() {
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        def each_of(a)
          i = 0; out = []
          while i < a.size; out << (yield a[i]); i += 1; end
          out
        end
        UNDEFD = Class.new { def gone = :here }
        o = UNDEFD.new
        30.times { each_of([o, o], &:gone) }
        before = each_of([o], &:gone)
        UNDEFD.class_eval { undef_method :gone }
        [before, (each_of([o], &:gone) rescue $!.class)]
    "#), @"[[:here], NoMethodError]");
}

#[test]
fn test_symbol_block_megamorphic_handles_immediate_receivers() {
    // The inline probe leaves immediates and `false` to the C helper, because CLASS_OF() on one
    // is a table lookup rather than a field load. They still have to reach the right method.
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        def each_of(a)
          i = 0; out = []
          while i < a.size; out << (yield a[i]); i += 1; end
          out
        end
        out = nil
        30.times { out = each_of([1, :sym, nil, true, false, 2.0, "s".freeze], &:inspect) }
        out
    "#), @r#"["1", ":sym", "nil", "true", "false", "2.0", "\"s\""]"#);
}

#[test]
fn test_symbol_block_megamorphic_ignores_refinements() {
    // A Symbol block never sees a refinement: vm_caller_setup_arg_block() turns `&:sym` written
    // where one is active into an ifunc lambda instead of a Symbol handler, and
    // vm_call_symbol() resolves against a nil cref. So the table, whose search does not consult
    // a cref either, may only answer for a method entry that is not a refined one.
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        module Ref
          refine String do
            def upcase = "REFINED"
          end
        end
        def each_of(a)
          i = 0; out = []
          while i < a.size; out << (yield a[i]); i += 1; end
          out
        end
        def plain(a) = each_of(a, &:upcase)
        eval("using Ref\ndef refined(a) = each_of(a, &:upcase)", TOPLEVEL_BINDING)
        30.times { plain(["a"]); refined(["a"]) }
        [plain(["a"]), refined(["a"]), plain(["a"])]
    "#), @r#"[["A"], ["REFINED"], ["A"]]"#);
}

#[test]
fn test_symbol_block_megamorphic_dispatches_correctly_under_gc_stress() {
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        def each_of(a)
          i = 0; out = []
          while i < a.size; out << (yield a[i]); i += 1; end
          out
        end
        ks = 60.times.map { |i| k = Class.new; k.class_eval("def value; #{i}; end"); k }
        objs = ks.map(&:new)
        30.times { each_of(objs, &:value) }
        GC.stress = true
        out = each_of(objs, &:value).sum + each_of([1, "a", :b], &:inspect).size
        GC.stress = false
        GC.compact
        [out, each_of(objs, &:value).sum]
    "#), @"[1773, 1770]");
}

#[test]
fn test_symbol_block_megamorphic_disabled_falls_back_to_invokeblock() {
    // --zjit-disable-symbol-block-mega has to leave the site working, since that is what an
    // A/B measurement compares against.
    crate::options::enable_zjit_stats();
    set_call_threshold(2);
    crate::options::rb_zjit_prepare_options();
    unsafe { crate::options::OPTIONS.as_mut() }.unwrap().disable_symbol_block_mega = true;
    eval(SYM_MEGA_SETUP);
    let before = crate::state::ZJITState::get_counters().symbol_block_mega_sites;
    let result = inspect("sym_run(SYM_OBJS).sum");
    let after = crate::state::ZJITState::get_counters().symbol_block_mega_sites;
    unsafe { crate::options::OPTIONS.as_mut() }.unwrap().disable_symbol_block_mega = false;
    assert_eq!(before, after, "the megamorphic Symbol block tier should be off");
    assert_snapshot!(result, @"1770");
}

#[test]
fn test_array_aref_out_of_bounds_reads_nil() {
    assert_snapshot!(inspect(r#"
      def test(ary, idx) = ary[idx]
      ary = [1, 2, 3]
      test(ary, 0)
      test(ary, 0)
      [test(ary, 0), test(ary, 2), test(ary, 3), test(ary, 100),
       test(ary, -1), test(ary, -3), test(ary, -4), test(ary, -100),
       test([], 0), test([], -1)]
    "#), @"[1, 3, nil, nil, 3, 1, nil, nil, nil, nil]");
}

#[test]
fn test_array_aref_walks_off_the_end_without_exiting() {
    assert_snapshot!(inspect(r#"
      def test(ary)
        out = []
        idx = 0
        while (elem = ary[idx])
          out << elem
          idx += 1
        end
        out
      end
      test([1, 2, 3])
      test([1, 2, 3])
      test([4, 5])
    "#), @"[4, 5]");
}

#[test]
fn test_splat_forwarded_to_rest_parameter() {
    assert_snapshot!(inspect(r#"
      def callee(name, *args) = [name, args]
      def test(name, args) = callee(name, *args)
      test(:a, [1])
      test(:a, [1])
      [test(:a, []), test(:b, [1]), test(:c, [1, 2, 3]),
       test(:d, [{x: 1}]), test(:e, [nil])]
    "#), @"[[:a, []], [:b, [1]], [:c, [1, 2, 3]], [:d, [{x: 1}]], [:e, [nil]]]");
}

#[test]
fn test_splat_forwarded_to_rest_parameter_after_extra_positionals() {
    assert_snapshot!(inspect(r#"
      def callee(name, *args) = [name, args]
      def test(name, obj, args) = callee(name, obj, *args)
      test(:a, :o, [1])
      test(:a, :o, [1])
      [test(:a, :o, []), test(:b, :o, [1]), test(:c, :o, [1, 2])]
    "#), @"[[:a, [:o]], [:b, [:o, 1]], [:c, [:o, 1, 2]]]");
}

#[test]
fn test_splat_forwarded_to_rest_parameter_is_a_copy() {
    assert_snapshot!(inspect(r#"
      def callee(*args)
        args << :added
        args
      end
      def test(args) = callee(*args)
      ary = [1, 2]
      test(ary)
      test(ary)
      [test(ary), ary]
    "#), @"[[1, 2, :added], [1, 2]]");
}

#[test]
fn test_splat_forwarded_through_two_frames_with_varying_length() {
    assert_snapshot!(inspect(r#"
      def inner(name, *args) = [name, args]
      def middle(name, *args) = inner(name, *args)
      def test(name, args) = middle(name, *args)
      3.times { test(:warm, [1]) }
      3.times { test(:warm, [1, 2]) }
      [test(:a, []), test(:b, [1]), test(:c, [1, 2, 3])]
    "#), @"[[:a, []], [:b, [1]], [:c, [1, 2, 3]]]");
}

#[test]
fn test_splat_with_ruby2_keywords_hash_is_not_forwarded() {
    assert_snapshot!(inspect(r#"
      def target(*args, **kw) = [args, kw]
      def callee(name, *args) = [name, args]
      ruby2_keywords def fwd(*args) = callee(:n, *args)
      def test(*args) = fwd(*args)
      test(1, k: 2)
      test(1, k: 2)
      test(1, k: 2)
    "#), @"[:n, [1, {k: 2}]]");
}


/// Run `program` and assert that no shape guard side-exited while it ran.
#[track_caller]
fn assert_no_shape_guard_exits(program: &str) -> String {
    let before = crate::state::ZJITState::get_counters().exit_guard_shape_failure;
    let result = inspect(program);
    let after = crate::state::ZJITState::get_counters().exit_guard_shape_failure;
    assert_eq!(before, after, "expected no shape guard side exits, but {} happened", after - before);
    result
}

/// Define objects that all read `@config` through one inherited method, each with a distinct
/// shape, so the read site sees more shapes than the profile has buckets.
const SHAPE_CHAIN_SETUP: &str = r#"
    class ShapeChainBase
      def read = @config
      def write(val) = @config = val
    end

    OBJS = (0...20).map do |i|
      klass = Class.new(ShapeChainBase)
      ivars = (0...i).map { |j| "@x#{j} = #{j}" }.join("; ")
      klass.class_eval("def initialize(n); #{ivars}; @config = n; end")
      klass.new(i)
    end
"#;

#[test]
fn test_getivar_shape_chain_falls_back_instead_of_exiting() {
    // Only `num_profiles` samples are taken, so a site that sees 20 shapes profiles a handful of
    // them and has to handle the rest at runtime. That must not be a side exit.
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("5.times { |i| OBJS[i].read }");
    assert_snapshot!(assert_no_shape_guard_exits("
        total = 0
        3.times { OBJS.each { |o| total += o.read } }
        total
    "), @"570");
}

#[test]
fn test_getivar_shape_chain_miss_takes_the_fallback_arm() {
    crate::options::enable_zjit_stats();
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("5.times { |i| OBJS[i].read }");
    let before = crate::state::ZJITState::get_counters().getivar_fallback_shape_chain_miss;
    assert_snapshot!(inspect("OBJS.map { |o| o.read }.sum"), @"190");
    let after = crate::state::ZJITState::get_counters().getivar_fallback_shape_chain_miss;
    assert!(after > before, "expected the generic ivar fallback arm to run, but the counter did not move");
}

#[test]
fn test_setivar_shape_chain_falls_back_instead_of_exiting() {
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("5.times { |i| OBJS[i].write(i) }");
    assert_snapshot!(assert_no_shape_guard_exits("
        OBJS.each_with_index { |o, i| o.write(i * 2) }
        OBJS.map { |o| o.read }.sum
    "), @"380");
}

#[test]
fn test_setivar_shape_chain_fallback_raises_on_frozen_receiver() {
    // The fallback arm is a generic SetIvar, which still has to raise on a frozen receiver.
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("5.times { |i| OBJS[i].write(i) }");
    assert_snapshot!(assert_no_shape_guard_exits(r#"
        frozen = OBJS.last.clone.freeze
        begin
          frozen.write(1)
          "no raise"
        rescue FrozenError
          "raised"
        end
    "#), @r#""raised""#);
}

#[test]
fn test_getivar_shape_chain_reads_ivar_defined_after_compile() {
    // An object whose @config is still undefined reads as nil, and reading it again after the
    // write (which moves the object to a shape the chain never saw) sees the new value.
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("5.times { |i| OBJS[i].read }");
    assert_snapshot!(assert_no_shape_guard_exits(r#"
        obj = ShapeChainBase.new
        before = obj.read
        obj.write(42)
        [before, obj.read]
    "#), @"[nil, 42]");
}

#[test]
fn test_ivar_shape_chain_falls_back_for_too_complex_shapes() {
    // Objects that outgrew their class's shape variation limit store ivars in a hash table, so
    // the shape chain cannot index them and they have to take the generic arm.
    set_call_threshold(6);
    eval(r#"
        class TooComplex
          def read = @config
          def write(val) = @config = val
        end

        COMPLEX = (0...20).map do |i|
          obj = TooComplex.new
          i.times { |j| obj.instance_variable_set("@x#{i}_#{j}", j) }
          obj.instance_variable_set(:@config, i)
          obj
        end
    "#);
    eval("5.times { |i| COMPLEX[i].read }");
    assert_snapshot!(assert_no_shape_guard_exits("
        COMPLEX.each_with_index { |o, i| o.write(i) }
        COMPLEX.map { |o| o.read }.sum
    "), @"190");
}

#[test]
fn test_ivar_cache_serves_the_shape_chain_fallback_inline() {
    // Once the table is warm, the shapes the guard chain has no arm for are read
    // in JIT code with no call: getivar_cache_hit is the inline path.
    crate::options::enable_zjit_stats();
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("5.times { |i| OBJS[i].read }");
    eval("OBJS.each { |o| o.read }"); // warm the table
    let before = crate::state::ZJITState::get_counters().getivar_cache_hit;
    assert_snapshot!(inspect("OBJS.map { |o| o.read }.sum"), @"190");
    let after = crate::state::ZJITState::get_counters().getivar_cache_hit;
    assert!(after > before, "expected the inline ivar table probe to hit, but the counter did not move");
}

#[test]
fn test_ivar_cache_serves_absent_ivars_as_nil() {
    // An absent ivar is cached as such and served inline, which is the whole
    // reason the probe has a nil arm.
    crate::options::enable_zjit_stats();
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("class ShapeChainBase; def missing = @never_assigned; end");
    eval("5.times { |i| OBJS[i].missing }");
    eval("OBJS.each { |o| o.missing }");
    let before = crate::state::ZJITState::get_counters().getivar_cache_hit;
    assert_snapshot!(inspect("OBJS.map { |o| o.missing }.uniq"), @"[nil]");
    let after = crate::state::ZJITState::get_counters().getivar_cache_hit;
    assert!(after > before, "expected absent ivars to be served by the inline probe");
}

#[test]
fn test_ivar_cache_sees_writes_that_transition_the_shape() {
    // The table is keyed on the whole shape id, so adding the ivar moves the
    // receiver to a key the table has never seen rather than hitting a stale
    // "absent" entry.
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("class ShapeChainBase; def late = @late; def set_late(v) = @late = v; end");
    eval("5.times { |i| OBJS[i].late }");
    eval("OBJS.each { |o| o.late }");
    assert_snapshot!(inspect(r#"
        before = OBJS.map { |o| o.late }
        OBJS.each_with_index { |o, i| o.set_late(i) }
        [before.uniq, OBJS.map { |o| o.late }.sum]
    "#), @"[[nil], 190]");
}

#[test]
fn test_ivar_cache_write_raises_on_a_receiver_frozen_after_warmup() {
    // Freezing sets a bit inside shape_id, so a frozen receiver misses whatever
    // entry its unfrozen self warmed, and the write still has to raise.
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("5.times { |i| OBJS[i].write(i) }");
    eval("OBJS.each_with_index { |o, i| o.write(i) }");
    assert_snapshot!(inspect(r#"
        obj = OBJS.last
        reads = [obj.read]
        obj.freeze
        reads << obj.read
        begin
          obj.write(99)
          reads << "no raise"
        rescue FrozenError
          reads << "raised"
        end
        reads
    "#), @r#"[19, 19, "raised"]"#);
}

#[test]
fn test_ivar_cache_still_reads_after_remove_instance_variable() {
    // remove_instance_variable transitions the object to a rebuilt shape, so the
    // entry warmed for the old shape cannot be reached with the new one.
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("5.times { |i| OBJS[i].read }");
    eval("OBJS.each { |o| o.read }");
    assert_snapshot!(inspect(r#"
        obj = OBJS.last
        before = obj.read
        obj.remove_instance_variable(:@config)
        after = obj.read
        obj.write(7)
        [before, after, obj.read]
    "#), @"[19, nil, 7]");
}

#[test]
fn test_ivar_cache_reads_class_and_module_ivars() {
    // Classes and modules get a table entry kind of their own, read through
    // rb_ivar_get_at_no_ractor_check while only one ractor exists.
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        module CacheMod; end
        class CacheCls; end
        CacheMod.instance_variable_set(:@a, 1)
        CacheCls.instance_variable_set(:@a, 2)
        def read(x) = x.instance_variable_get(:@a)
        5.times { [CacheMod, CacheCls].each { |x| read(x) } }
        [read(CacheMod), read(CacheCls), read(CacheMod.dup)]
    "#), @"[1, 2, 1]");
}

/// A getivar site whose profile filled up with shapes that stop appearing, followed by a long
/// run of shapes it has no bucket left to record. `observe_ivar_fallback` drops the cold buckets
/// so the live shapes can be recorded and the site respecialized; every read has to keep
/// answering correctly through the eviction, the refill and the recompile.
const IVAR_EVICTION_SETUP: &str = r#"
    class EvictBase
      def read = @config
      def write(val) = @config = val
    end

    def evict_shapes(prefix, count, first_ivar)
      (0...count).map do |i|
        klass = Class.new(EvictBase)
        ivars = (0...(first_ivar + i)).map { |j| "@#{prefix}#{j} = #{j}" }.join("; ")
        klass.class_eval("def initialize(n); #{ivars}; @config = n; end")
        klass.new(i)
      end
    end

    BOOT_OBJS = evict_shapes("boot", 10, 0)
    LIVE_OBJS = evict_shapes("live", 3, 30)
"#;

#[test]
fn test_getivar_profile_eviction_keeps_reads_correct() {
    set_call_threshold(6);
    eval(IVAR_EVICTION_SETUP);
    // Fill every profile bucket with shapes that never come back.
    eval("20.times { BOOT_OBJS.each { |o| o.read } }");
    // Then hammer the site with shapes the profile cannot hold. This is what triggers the
    // eviction, the refill and the recompile.
    eval("500.times { LIVE_OBJS.each { |o| o.read } }");
    assert_snapshot!(assert_no_shape_guard_exits("
        LIVE_OBJS.map { |o| o.read } + BOOT_OBJS.map { |o| o.read }
    "), @"[0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
}

#[test]
fn test_getivar_profile_eviction_runs_and_is_bounded() {
    crate::options::enable_zjit_stats();
    set_call_threshold(6);
    eval(IVAR_EVICTION_SETUP);
    eval("20.times { BOOT_OBJS.each { |o| o.read } }");
    let before = crate::state::ZJITState::get_counters().ivar_profile_evicted_count;
    eval("500.times { LIVE_OBJS.each { |o| o.read } }");
    let after = crate::state::ZJITState::get_counters().ivar_profile_evicted_count;
    assert!(after > before, "expected the crowded profile to drop its cold buckets");
    // MAX_IVAR_PROFILE_EVICTIONS per site, and this program only has the one.
    assert!(after - before <= 2, "expected at most two evictions, got {}", after - before);
}

#[test]
fn test_getivar_profile_eviction_reads_ivars_added_after_it() {
    // The refilled buckets are shapes, not objects: an object that transitions after the
    // recompile still has to read correctly off the fallback.
    set_call_threshold(6);
    eval(IVAR_EVICTION_SETUP);
    eval("20.times { BOOT_OBJS.each { |o| o.read } }");
    eval("500.times { LIVE_OBJS.each { |o| o.read } }");
    assert_snapshot!(assert_no_shape_guard_exits(r#"
        fresh = EvictBase.new
        before = fresh.read
        fresh.write(7)
        frozen = LIVE_OBJS.first.clone.freeze
        [before, fresh.read, frozen.read]
    "#), @"[nil, 7, 0]");
}

#[test]
fn test_setivar_profile_is_not_evicted() {
    // A setivar arm has to survive `prepare_optimized_setivar` as well as a shape match, so
    // refilling its buckets from the fallback is a worse bet than keeping what the profile saw.
    crate::options::enable_zjit_stats();
    set_call_threshold(6);
    eval(IVAR_EVICTION_SETUP);
    eval("20.times { BOOT_OBJS.each_with_index { |o, i| o.write(i) } }");
    let before = crate::state::ZJITState::get_counters().ivar_profile_evicted_count;
    eval("500.times { LIVE_OBJS.each_with_index { |o, i| o.write(i) } }");
    let after = crate::state::ZJITState::get_counters().ivar_profile_evicted_count;
    assert_eq!(before, after, "a setinstancevariable site must not evict its profile");
    assert_snapshot!(assert_no_shape_guard_exits("LIVE_OBJS.map { |o| o.read }"), @"[0, 1, 2]");
}

#[test]
fn test_ivar_cache_class_ivars_follow_shape_transitions() {
    // Class and module shapes get their own table entry kind, so a class that gains an ivar
    // after the site was compiled -- and one that is frozen, and a singleton class -- all have
    // to keep reading the right slot.
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        KLASSES = 12.times.map do |i|
          k = Class.new
          i.times { |j| k.instance_variable_set(:"@pad#{j}", j) }
          k.instance_variable_set(:@a, i)
          k
        end
        def read(x) = x.instance_variable_get(:@a)
        5.times { KLASSES.each { |k| read(k) } }

        grown = KLASSES.first
        grown.instance_variable_set(:@later, 99)
        singleton = Object.new.singleton_class
        singleton.instance_variable_set(:@a, :sing)
        frozen = Class.new
        frozen.instance_variable_set(:@a, :frz)
        frozen.freeze

        [KLASSES.map { |k| read(k) }, read(grown), read(singleton), read(frozen), read(Class.new)]
    "#), @"[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 0, :sing, :frz, nil]");
}

#[test]
fn test_ivar_cache_class_and_instance_ivars_share_a_table() {
    // One table per ivar name, so a class shape and an object shape land in the same table and
    // must not read each other's slot.
    set_call_threshold(2);
    assert_snapshot!(inspect(r#"
        class Holder; def initialize(v) = (@pad = 0; @a = v); end
        HOLDERS = 12.times.map { |i| Holder.new(i) }
        CLS = 12.times.map do |i|
          k = Class.new
          i.times { |j| k.instance_variable_set(:"@pad#{j}", j) }
          k.instance_variable_set(:@a, -i)
          k
        end
        def read(x) = x.instance_variable_get(:@a)
        5.times { (HOLDERS + CLS).each { |x| read(x) } }
        [HOLDERS.map { |h| read(h) }.sum, CLS.map { |k| read(k) }.sum]
    "#), @"[66, -66]");
}

#[test]
fn test_ivar_cache_reads_survive_gc_stress() {
    set_call_threshold(6);
    eval(SHAPE_CHAIN_SETUP);
    eval("5.times { |i| OBJS[i].read }");
    eval("OBJS.each { |o| o.read }");
    assert_snapshot!(inspect(r#"
        GC.stress = true
        sums = 2.times.map { OBJS.map { |o| o.read }.sum }
        GC.stress = false
        sums
    "#), @"[190, 190]");
}


#[test]
fn test_array_each_is_defined_in_ruby() {
    assert_snapshot!(inspect("Array.instance_method(:each).source_location&.first"), @r#""<internal:array>""#);
}

#[test]
fn test_forward_fallback_with_lightweight_frame_reads_cfp() {
    assert_snapshot!(inspect(r#"
      class Base
        def foo(...)
          "base"
        end
      end

      class Child < Base
        def foo(...)
          bar do
            super
          end
        end

        def bar
          yield
        end
      end

      c = Child.new
      100.times do
        Array.new(50) { |n| n * n }
        c.foo(1, 2, 3)
      end
      :done
    "#), @":done");
}

// --- Narrowed `test reg, imm` ---------------------------------------------------

// `test rdi, 7` is emitted as `test dil, 7` when only ZF is read. The bits above
// the immediate are masked off either way, so a receiver whose pointer has plenty
// of high bits set must still be classified as a heap object, and an immediate
// whose payload sets high bits must still be classified as one.
#[test]
fn test_narrowed_test_high_bits() {
    assert_snapshot!(inspect(r#"
        class Big; def go = :heap; end
        class Integer; def go = :int; end
        class Symbol; def go = :sym; end
        class Float; def go = :float; end
        class NilClass; def go = :nil; end
        class FalseClass; def go = :false; end
        class TrueClass; def go = :true; end
        def test(o) = o.go
        objs = 200.times.map { Big.new }
        20.times { test(objs.sample); test(1); test(:s) }
        # Large fixnums and negative ones set bits well above the tag byte; 1e300 is
        # outside the flonum range, so it is a heap Float.
        [test(objs.last), test(1 << 40), test(-(1 << 40)), test(0), test(:zz),
         test(1.5), test(1.5e300), test(nil), test(false), test(true)]
    "#), @"[:heap, :int, :int, :int, :sym, :float, :float, :nil, :false, :true]");
}

#[test]
fn test_send_with_profiled_method_name() {
    eval("
        class SendProfiled
          def double(n) = n * 2
        end
        def entry(c) = c.send(:double, 21)
        C1 = SendProfiled.new
        5.times { entry(C1) }
    ");
    assert_snapshot!(assert_compiles("entry(C1)"), @"42");
}

#[test]
fn test_send_with_two_profiled_method_names() {
    eval("
        class SendTwoNames
          def a(n) = n + 1
          def b(n) = n + 2
        end
        def call_by_name(c, name) = c.send(name, 10)
        C2 = SendTwoNames.new
        def entry = (1..10).map { |i| call_by_name(C2, i.even? ? :a : :b) }.uniq.sort
        5.times { entry }
    ");
    assert_snapshot!(assert_compiles("entry"), @"[11, 12]");
}

#[test]
fn test_send_can_call_private_method() {
    eval("
        class SendPrivate
          private def secret = :shh
        end
        def entry(c) = c.send(:secret)
        C3 = SendPrivate.new
        5.times { entry(C3) }
    ");
    assert_snapshot!(assert_compiles("entry(C3)"), @":shh");
}

#[test]
fn test_send_with_unprofiled_method_name_falls_back() {
    // The dispatch chain only covers the names seen while profiling; anything else has to
    // reach the generic send in the fall-through arm and still produce the right answer.
    eval("
        class SendUnprofiled
          def a = :a
          def b = :b
        end
        def entry(c, name) = c.send(name)
        C4 = SendUnprofiled.new
        20.times { entry(C4, :a) }
    ");
    assert_snapshot!(assert_compiles("[entry(C4, :a), entry(C4, :b)]"), @"[:a, :b]");
}

#[test]
fn test_send_with_string_method_name() {
    // String names are never recorded by the profiler, so this stays a dynamic send.
    eval("
        class SendStringName
          def a = :a
        end
        def entry(c, name) = c.send(name)
        C5 = SendStringName.new
        20.times { entry(C5, 'a') }
    ");
    assert_snapshot!(assert_compiles("entry(C5, 'a')"), @":a");
}

#[test]
fn test_send_to_cfunc() {
    eval("
        def entry(s) = s.send(:upcase)
        5.times { entry('ab') }
    ");
    assert_snapshot!(assert_compiles("entry('ab')"), @r#""AB""#);
}

#[test]
fn test_send_with_multiple_arguments() {
    eval("
        class SendManyArgs
          def add(a, b, c) = a + b + c
        end
        def entry(o) = o.send(:add, 1, 2, 3)
        C6 = SendManyArgs.new
        5.times { entry(C6) }
    ");
    assert_snapshot!(assert_compiles("entry(C6)"), @"6");
}

#[test]
fn test_send_picks_up_redefinition() {
    eval("
        class SendRedefined
          def a = 1
        end
        def entry(c) = c.send(:a)
        C7 = SendRedefined.new
        20.times { entry(C7) }
    ");
    assert_snapshot!(assert_compiles_allowing_exits("
        before = entry(C7)
        class SendRedefined
          def a = 2
        end
        [before, entry(C7)]
    "), @"[1, 2]");
}

#[test]
fn test_send_picks_up_redefinition_of_send_itself() {
    eval("
        class SendOverridden
          def a = :a
        end
        def entry(c) = c.send(:a)
        C9 = SendOverridden.new
        20.times { entry(C9) }
    ");
    assert_snapshot!(assert_compiles_allowing_exits("
        before = entry(C9)
        class SendOverridden
          def send(name) = :overridden
        end
        [before, entry(C9)]
    "), @"[:a, :overridden]");
}

#[test]
fn test_send_nested_is_not_specialized() {
    eval("
        class SendNested
          def a = :a
        end
        def entry(c) = c.send(:send, :a)
        C8 = SendNested.new
        20.times { entry(C8) }
    ");
    assert_snapshot!(assert_compiles("entry(C8)"), @":a");
}


/// A shared iterator whose `yield` is handed a different block ISEQ by every one of its callers,
/// which is what `gen_invoke_block_iseq_dynamic` exists for: no guard chain can cover 120
/// distinct blocks, so without it every one of these yields calls `rb_vm_invokeblock`.
///
/// `each_of` is padded past the inline threshold on purpose. That is the shape the optimization
/// is for -- `Array#each` is called from methods whose inline budget is long spent -- and an
/// inlined copy would resolve the `yield` to its caller's literal block instead.
const DYNAMIC_BLOCK_SETUP: &str = r#"
    PAD = (0...60).map { |k| "pad += #{k}" }.join("\n  ")
    eval(<<~RUBY)
      def each_of(a)
        i = 0
        n = a.size
        pad = 0
        #{PAD}
        while i < n
          yield a[i]
          i += 1
        end
        a
      end
    RUBY

    # Each caller closes over a literal block of its own, so the yield site above sees 120
    # different `captured->code.iseq` values.
    CALLERS = (0...120).map do |i|
      eval("proc { |arr, acc| each_of(arr) { |x| acc << x + #{i} } }")
    end

    def drive(times)
      acc = []
      times.times { CALLERS.each { |c| c.call([100], acc) } }
      acc
    end
"#;

#[test]
fn test_block_dynamic_dispatch_over_many_distinct_blocks() {
    crate::options::enable_zjit_stats();
    set_call_threshold(2);
    eval(DYNAMIC_BLOCK_SETUP);
    eval("drive(6)");
    let before = crate::state::ZJITState::get_counters().block_iseq_dynamic_optimized_send_count;
    assert_snapshot!(inspect("
        acc = drive(2)
        [acc.size, acc.first, acc.last, acc.uniq.size]
    "), @"[240, 100, 219, 120]");
    let after = crate::state::ZJITState::get_counters().block_iseq_dynamic_optimized_send_count;
    assert!(after > before,
        "expected the yield site to enter blocks it has no chain arm for, but the counter did not move");
}

#[test]
fn test_block_dynamic_dispatch_nil_fills_block_locals() {
    // `local_table_size > param.lead_num`: the dispatch has to nil-fill the block's own locals
    // the way `vm_push_frame` does, at an offset it only learns at run time. Each block declares
    // its locals with an assignment that never runs, so a slot left holding whatever was on the
    // VM stack shows up as a wrong value rather than as nil.
    crate::options::enable_zjit_stats();
    set_call_threshold(2);
    eval(DYNAMIC_BLOCK_SETUP);
    eval(r#"
        LOCAL_CALLERS = (0...10).map do |n|
          decls = (0...n).map { |j| "l#{j} = x if false" }.join("; ")
          reads = "[" + (0...n).map { |j| "l#{j}" }.join(", ") + "]"
          eval("proc { |arr, acc| each_of(arr) { |x| #{decls}; acc << #{reads} } }")
        end

        def drive_locals(times)
          acc = []
          times.times { LOCAL_CALLERS.each { |c| c.call([1], acc) } }
          acc
        end
    "#);
    eval("drive_locals(6)");
    let before = crate::state::ZJITState::get_counters().block_iseq_dynamic_optimized_send_count;
    assert_snapshot!(inspect("
        acc = drive_locals(2)
        [acc.size, acc[0], acc[3], acc[9]]
    "), @"[20, [], [nil, nil, nil], [nil, nil, nil, nil, nil, nil, nil, nil, nil]]");
    let after = crate::state::ZJITState::get_counters().block_iseq_dynamic_optimized_send_count;
    assert!(after > before, "expected blocks with their own locals to be entered directly");
}

#[test]
fn test_block_dynamic_dispatch_arity_mismatch_falls_back() {
    // Only an exact arity match dispatches directly. Everything else has to reach
    // `rb_vm_invokeblock`, which nil-fills, truncates and auto-splats.
    set_call_threshold(2);
    eval(DYNAMIC_BLOCK_SETUP);
    eval("drive(6)");
    assert_snapshot!(inspect("
        out = []
        each_of([1, 2]) { |a, b| out << [a, b] }
        each_of([[3, 4]]) { |a, b| out << [a, b] }
        each_of([5]) { out << :none }
        each_of([6]) { |a, b, c| out << [a, b, c] }
        out
    "), @"[[1, nil], [2, nil], [3, 4], :none, [6, nil, nil]]");
}

#[test]
fn test_block_dynamic_dispatch_break_next_and_return() {
    // A `throw` out of a frame this dispatch pushed unwinds through every JIT native frame back
    // to the enclosing `vm_exec`, which has to find a usable ISEQ on the pushed frame.
    set_call_threshold(2);
    eval(DYNAMIC_BLOCK_SETUP);
    eval("drive(6)");
    assert_snapshot!(inspect("
        def with_break(arr) = each_of(arr) { |x| break x * 10 if x > 1 }
        def with_next(arr)
          out = []
          each_of(arr) { |x| next if x.even?; out << x }
          out
        end
        def with_return(arr)
          each_of(arr) { |x| return x * 100 if x > 1 }
          :fell_through
        end
        [with_break([1, 2, 3]), with_next([1, 2, 3, 4]), with_return([1, 2, 3]), with_return([0])]
    "), @"[20, [1, 3], 200, :fell_through]");
}

#[test]
fn test_block_dynamic_dispatch_under_gc_stress() {
    // The frame this pushes is not published at `ec->cfp` until the callee's entry point runs,
    // so a collection anywhere across the push has to still see a walkable stack.
    set_call_threshold(2);
    eval(DYNAMIC_BLOCK_SETUP);
    eval("drive(6)");
    assert_snapshot!(inspect("
        GC.stress = true
        acc = drive(1)
        GC.stress = false
        [acc.size, acc.first, acc.last]
    "), @"[120, 100, 219]");
}

#[test]
fn test_block_dynamic_dispatch_flushed_by_tracepoint() {
    // Enabling a TracePoint clears every `jit_entry`, which is the only thing this path trusts
    // about the callee, so the dispatch has to fall back rather than enter stale code.
    set_call_threshold(2);
    eval(DYNAMIC_BLOCK_SETUP);
    eval("drive(6)");
    assert_snapshot!(inspect("
        seen = 0
        tp = TracePoint.new(:b_call) { |_| seen += 1 }
        traced = tp.enable { drive(1) }
        after = drive(1)
        [traced.size, seen > 0, after.size, after.last]
    "), @"[120, true, 120, 219]");
}

#[test]
fn test_block_dynamic_dispatch_survives_method_redefinition_mid_iteration() {
    // Redefining a method the blocks call, from inside one of them, invalidates compiled code
    // while frames this dispatch pushed are still on the stack.
    set_call_threshold(2);
    eval(DYNAMIC_BLOCK_SETUP);
    eval(r#"
        def tag(x) = x * 2
        REDEF_CALLERS = (0...40).map do |i|
          eval("proc { |arr, acc| each_of(arr) { |x| acc << tag(x) + #{i} } }")
        end
        def drive_redef(times)
          acc = []
          times.times { REDEF_CALLERS.each { |c| c.call([1], acc) } }
          acc
        end
    "#);
    eval("drive_redef(6)");
    assert_snapshot!(inspect("
        before = drive_redef(1)
        each_of([1]) { |_| def tag(x) = x * 1000 }
        after = drive_redef(1)
        [before.first, before.last, after.first, after.last]
    "), @"[2, 41, 1000, 1039]");
}

/// A `yield` site compiled from a profile that only ever saw ISEQ block handlers, which is then
/// handed `&:sym` handlers for the rest of the process. Nothing about the compiled dispatch
/// fails -- it branches to `rb_vm_invokeblock` rather than side-exiting -- so only
/// `rb_zjit_block_reprofile` can notice that it is serving the wrong traffic.
const BLOCK_RESPECIALIZE_SETUP: &str = r#"
    RPAD = (0...60).map { |k| "pad += #{k}" }.join("\n  ")
    eval(<<~RUBY)
      def each_of(a)
        i = 0
        n = a.size
        pad = 0
        #{RPAD}
        while i < n
          yield a[i]
          i += 1
        end
        a
      end
    RUBY

    BOOT_BLOCKS = (0...4).map do |i|
      eval("proc { |arr, acc| each_of(arr) { |x| acc << x + #{i} } }")
    end

    def boot(times)
      acc = []
      times.times { BOOT_BLOCKS.each { |c| c.call([1], acc) } }
      acc
    end

    def live(times)
      out = nil
      times.times { out = each_of(["a", "b", "c"].map(&:dup), &:upcase!) }
      out
    end
"#;

#[test]
fn test_block_respecialization_converges() {
    crate::options::enable_zjit_stats();
    set_call_threshold(2);
    eval(BLOCK_RESPECIALIZE_SETUP);
    // Freeze the yield site on a profile of ISEQ block handlers.
    eval("boot(20)");
    let before = crate::state::ZJITState::get_counters().block_respecialize_count;
    // Then give it nothing but Symbol handlers over a monomorphic receiver.
    eval("live(2000)");
    let after = crate::state::ZJITState::get_counters().block_respecialize_count;
    assert!(after > before,
        "expected the frozen yield site to earn a respecialization from its fallback traffic");
    assert!(after - before <= crate::payload::MAX_BLOCK_RESPECIALIZATIONS as u64,
        "expected at most {} respecializations, got {}",
        crate::payload::MAX_BLOCK_RESPECIALIZATIONS, after - before);

    // Stable from here: the respecialized dispatch serves the traffic, so no further windows
    // close and no further versions are asked for.
    let steady = crate::state::ZJITState::get_counters().block_respecialize_count;
    eval("live(20000)");
    assert_eq!(steady, crate::state::ZJITState::get_counters().block_respecialize_count,
        "a respecialized site must stop asking for versions once it stops falling back");

    assert_snapshot!(inspect("[live(1), boot(1).last]"), @r#"[["A", "B", "C"], 4]"#);
}

#[test]
fn test_block_respecialization_declines_unservable_handlers() {
    // A `yield` handed Proc block handlers has nothing a rebuilt dispatch could add an arm for,
    // so it must not spend versions rebuilding itself.
    crate::options::enable_zjit_stats();
    set_call_threshold(2);
    eval(BLOCK_RESPECIALIZE_SETUP);
    eval("boot(20)");
    let before = crate::state::ZJITState::get_counters().block_respecialize_count;
    assert_snapshot!(inspect(r#"
        blk = proc { |x| x.to_s }
        out = nil
        2000.times { out = each_of([1, 2], &blk) }
        out
    "#), @"[1, 2]");
    let after = crate::state::ZJITState::get_counters().block_respecialize_count;
    assert_eq!(before, after,
        "a fallback that only ever sees Proc handlers must not earn a respecialization");
}

// --- HasType fused into the branch that consumes it -----------------------------
//
// A CondBranch on a HasType now jumps out of the type checks directly instead of
// merging them into a 0/1 that the branch re-tests. These cover each arm of
// gen_has_type_branch() and, importantly, the values that must take the *false*
// path out of the heap-object arm: immediates and Qfalse, which are rejected
// before the class field is ever loaded.

// Every kind of receiver through one polymorphic call site. nil/false/true and
// the immediates must not have their RBasic read.
#[test]
fn test_has_type_branch_polymorphic_receivers() {
    assert_snapshot!(inspect(r#"
        class Wrapped; def kind = :obj; end
        def dispatch(o) = o.kind
        class Integer; def kind = :int; end
        class Symbol; def kind = :sym; end
        class Float; def kind = :float; end
        class NilClass; def kind = :nil; end
        class FalseClass; def kind = :false; end
        class TrueClass; def kind = :true; end
        class String; def kind = :str; end
        class Array; def kind = :ary; end
        vals = [1, :s, 1.5, nil, false, true, "x", [1], Wrapped.new]
        20.times { vals.each { |v| dispatch(v) } }
        vals.map { |v| dispatch(v) }
    "#), @"[:int, :sym, :float, :nil, :false, :true, :str, :ary, :obj]");
}

// A class-check arm whose receiver turns out to be an immediate or `false` must
// take the false edge. This is the case the fused form rules out with the
// `test`/`cmp` pair before touching RBASIC(val)->klass; getting it wrong reads a
// class field out of a tagged integer.
#[test]
fn test_has_type_branch_rejects_immediates_and_false() {
    assert_snapshot!(inspect(r#"
        class Box; def val = 1; end
        def dispatch(o) = o.val
        class Integer; def val = 2; end
        # Train the site on Box only, so Box is the single guarded class and every
        # other receiver has to fall out of the class check.
        b = Box.new
        20.times { dispatch(b) }
        [dispatch(b), dispatch(7), dispatch(-1)]
    "#), @"[1, 2, 2]");
}

// The builtin-type arm (T_ tag rather than an exact class) with a subclass
// receiver, plus the same immediate/false rejection.
#[test]
fn test_has_type_branch_builtin_type_arm() {
    assert_snapshot!(inspect(r#"
        class MyStr < String; end
        def sizeof(o) = o.size
        vals = ["abc", MyStr.new("wxyz"), [1, 2], {a: 1}]
        20.times { vals.each { |v| sizeof(v) } }
        vals.map { |v| sizeof(v) }
    "#), @"[3, 4, 2, 1]");
}

// The fused branch keeps the tested value live across the checks; a guard that
// fails on a later arm still has to be able to side-exit with correct state.
#[test]
fn test_has_type_branch_side_exit_state() {
    assert_snapshot!(inspect(r#"
        class A; def go(x) = x + 1; end
        class B; def go(x) = x + 2; end
        def test(o, x)
          y = x * 10
          z = o.go(x)
          [y, z, o.class.name]
        end
        20.times { |i| test(i.even? ? A.new : B.new, i) }
        # C was never profiled: the call site falls out of every fused type check.
        class C; def go(x) = x + 3; end
        [test(A.new, 1), test(B.new, 1), test(C.new, 1)]
    "#), @r#"[[10, 2, "A"], [10, 3, "B"], [10, 4, "C"]]"#);
}

// Fusing must not fire when the HasType result is used by something other than
// the branch, or the second consumer would read a value that was never
// materialized.
#[test]
fn test_has_type_branch_multi_use_not_fused() {
    assert_snapshot!(inspect(r#"
        class A; def go = 1; end
        class B; def go = 2; end
        def test(o)
          # `o.go` builds the fused type dispatch; storing the receiver keeps
          # other values from the same block live past the branch.
          r = o.go
          [r, o.class.name]
        end
        20.times { |i| test(i.even? ? A.new : B.new) }
        [test(A.new), test(B.new)]
    "#), @r#"[[1, "A"], [2, "B"]]"#);
}

// Under GC stress with a moving collector, the guarded class VALUE baked into the
// fused compare has to be updated like the unfused one was.
#[test]
fn test_has_type_branch_gc_compact() {
    assert_snapshot!(inspect(r#"
        class A; def go = :a; end
        class B; def go = :b; end
        def test(o) = o.go
        20.times { |i| test(i.even? ? A.new : B.new) }
        GC.compact
        [test(A.new), test(B.new), test(A.new)]
    "#), @"[:a, :b, :a]");
}

#[test]
fn test_side_exits_are_emitted_into_the_outlined_region() {
    // Side exits belong in the outlined half of the code region, so that a
    // function's body stays contiguous with the next function's. Compile a method
    // with plenty of guards and check that the bytes landed on the cold side of
    // the split without leaving a hole in the hot side.
    set_call_threshold(2);
    with_rubyvm(|| {
        let cb = crate::state::ZJITState::get_code_block();
        assert!(cb.has_outlined_region(), "ZJIT's code region should be split");
        let inlined_before = cb.inlined_code_size();
        let outlined_before = cb.outlined_code_size();

        // Type guards on an untyped parameter give this a side exit per operation.
        eval("
            def guarded(a, b) = a + b + a * b - a
            guarded(1, 2)
            guarded(1, 2)
        ");

        let cb = crate::state::ZJITState::get_code_block();
        assert!(cb.inlined_code_size() > inlined_before,
            "the function body should have been written to the inlined half");
        assert!(cb.outlined_code_size() > outlined_before,
            "the side exits should have been written to the outlined half");

        // The two halves are far enough apart that the exits cannot have been
        // mistaken for inline code, and close enough for a rel32 branch.
        let distance = cb.outlined_write_ptr().as_offset() - cb.get_write_ptr().as_offset();
        assert!(distance > 0, "the outlined half should sit above the inlined half");
        assert!(distance < i32::MAX as i64,
            "hot-to-cold branches have to stay in rel32 range, got a gap of {distance}");
    });
}

#[test]
fn test_side_exit_still_reached_from_outlined_region() {
    // Taking an exit that now lives megabytes away from the guard that jumps to it
    // must still land the interpreter on the right frame.
    set_call_threshold(2);
    assert_snapshot!(inspect("
        def add(a, b) = a + b
        add(1, 2)
        add(1, 2)
        [add(1, 2), add('x', 'y'), add(1.5, 2.5), add(1, 2)]
    "), @r#"[3, "xy", 4.0, 3]"#);
}

#[test]
fn test_outlined_side_exit_survives_compaction() {
    // A side exit bakes the VALUEs that were on the VM stack into its own code, so
    // GC compaction has to rewrite those in the outlined half too. That means
    // mark_all_writable has to cover the outlined arena: without it the write goes
    // to a page that is executable and not writable.
    set_call_threshold(2);
    with_rubyvm(|| {
        let out = inspect("
            BAKED_IN_EXIT = 'baked'
            def stacked(a) = [BAKED_IN_EXIT, a + 1]
            stacked(1)
            stacked(1)
            GC.compact
            [stacked(1), stacked(2.5)]
        ");
        assert_eq!(r#"[["baked", 2], ["baked", 3.5]]"#, out);

        // Make sure that had something to survive: the exit captures the constant
        // that was on the VM stack, so the ISEQ owns GC offsets that point into the
        // outlined half. Nothing is mapped between the end of the inlined code and
        // the start of the outlined half, so an offset past the inlined write
        // position is an offset in the outlined half.
        let cb = crate::state::ZJITState::get_code_block();
        let inlined_end = cb.get_write_ptr().as_offset();
        let iseq = get_method_iseq("self", "stacked");
        let outlined_gc_offsets = get_or_create_iseq_payload(iseq).versions.iter()
            .flat_map(|version| unsafe { version.as_ref() }.gc_offsets.offsets().iter())
            .filter(|offset| offset.as_offset() >= inlined_end)
            .count();
        assert!(outlined_gc_offsets > 0,
            "the side exit should have baked VALUEs into the outlined half");
    });
}

#[test]
fn test_invalidation_jumps_from_the_inlined_to_the_outlined_half() {
    // Invalidation overwrites a patch point in the hot half with a jump to the
    // patch point's side exit, which now lives in the cold half. The jump has to
    // reach it, and with_write_ptr has to hand back the range it really wrote so
    // that remove_gc_offsets drops the right offsets.
    set_call_threshold(2);
    assert_snapshot!(inspect("
        INVALIDATED_CONST = 1
        def read_const = INVALIDATED_CONST + 1
        read_const
        read_const
        before = read_const
        Object.send(:remove_const, :INVALIDATED_CONST)
        INVALIDATED_CONST = 10
        [before, read_const]
    "), @"[2, 11]");
}

// Tests for the redundant spill elimination in gen_save_sp() and gen_spill_locals(). Each of
// these programs lets the JIT skip a cfp->sp store or a local spill store, and then observes
// the frame from the interpreter, where a wrongly skipped store shows up as a stale local.

#[test]
fn test_spill_elision_rescue_reads_locals() {
    // A rescue in this frame runs in the interpreter and reads the locals out of their slots,
    // so every local the raising call could expose must be resident when it runs.
    assert_snapshot!(inspect("
        def raiser = raise('boom')
        def test(o)
          a = 1
          b = 2
          begin
            o.send(:itself)
            a = 3
            b = 4
            o.send(:raiser)
            a = 99
          rescue RuntimeError
            [a, b]
          end
        end
        o = Object.new
        [test(o), test(o), test(o), test(o)]
    "), @"[[3, 4], [3, 4], [3, 4], [3, 4]]");
}

#[test]
fn test_spill_elision_block_writes_caller_local() {
    // A block that shares this frame's EP writes the caller's local slot directly, so the JIT
    // must not assume the slot still holds what it last spilled there.
    assert_snapshot!(inspect("
        def twice
          yield
          yield
        end
        def test(o)
          x = 0
          y = 0
          o.send(:itself)
          twice { x += 1 }
          o.send(:itself)
          y = x + 1
          o.send(:itself)
          [x, y]
        end
        o = Object.new
        [test(o), test(o), test(o), test(o)]
    "), @"[[2, 3], [2, 3], [2, 3], [2, 3]]");
}

#[test]
fn test_spill_elision_binding_reads_locals() {
    // Binding reads the locals out of the frame, which requires them to be resident at the
    // call that creates it.
    assert_snapshot!(inspect("
        def test(o)
          a = 1
          b = 2
          o.send(:itself)
          a = 3
          o.send(:itself)
          bind = binding
          [bind.local_variable_get(:a), bind.local_variable_get(:b)]
        end
        o = Object.new
        [test(o), test(o), test(o)]
    "), @"[[3, 2], [3, 2], [3, 2]]");
}

#[test]
fn test_spill_elision_loop_keeps_locals_correct() {
    // Repeated dynamic sends in a loop: the locals that never change are spilled once, but a
    // Binding at the end must still see every local.
    assert_snapshot!(inspect("
        def test(o)
          a = 1
          b = 2
          c = 0
          i = 0
          while i < 5
            o.send(:itself)
            o.send(:itself)
            c = c + a + b
            i += 1
          end
          bind = binding
          [a, b, c, i, bind.local_variable_get(:c)]
        end
        o = Object.new
        [test(o), test(o), test(o)]
    "), @"[[1, 2, 15, 5, 15], [1, 2, 15, 5, 15], [1, 2, 15, 5, 15]]");
}

#[test]
fn test_spill_elision_ensure_reads_locals_under_gc_stress() {
    // An ensure that runs in the interpreter while GC is stressed reads the locals out of the
    // slots, so a skipped spill would surface as a stale or dead object.
    assert_snapshot!(inspect(r#"
        def raiser = raise('boom')
        def test(o, n)
          a = "a" * n
          b = [a]
          r = nil
          begin
            o.send(:itself)
            a = a + "!"
            o.send(:raiser)
          rescue RuntimeError
          ensure
            r = [a.size, b.size, b[0].size]
          end
          r
        end
        o = Object.new
        3.times { test(o, 3) }
        GC.stress = true
        out = 3.times.map { |i| test(o, i + 1) }
        GC.stress = false
        GC.start
        out
    "#), @"[[2, 1, 1], [3, 1, 2], [4, 1, 3]]");
}
