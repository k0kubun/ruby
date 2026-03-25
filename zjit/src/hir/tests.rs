#[cfg(test)]
use super::*;

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use insta::assert_snapshot;

    #[track_caller]
    fn hir_string(method: &str) -> String {
        let iseq = crate::cruby::with_rubyvm(|| get_method_iseq("self", method));
        unsafe { crate::cruby::rb_zjit_profile_disable(iseq) };
        let function = iseq_to_hir(iseq).unwrap();
        format!("{}", FunctionPrinter::with_snapshot(&function))
    }

    #[track_caller]
    fn optimized_hir_string(method: &str) -> String {
        let iseq = crate::cruby::with_rubyvm(|| get_proc_iseq(&format!("{}.method(:{})", "self", method)));
        unsafe { crate::cruby::rb_zjit_profile_disable(iseq) };
        let mut function = iseq_to_hir(iseq).unwrap();
        function.optimize();
        function.validate().unwrap();
        format!("{}", FunctionPrinter::with_snapshot(&function))
    }

    #[test]
    fn test_remove_redundant_patch_points() {
        eval("
            def test = 1 + 2 + 3
            test
            test
        ");
        assert_snapshot!(optimized_hir_string("test"), @"
        fn test@<compiled>:2:
        bb0():
          Entries bb1, bb2
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v10:Any = Snapshot FrameState { pc: 0x1000, stack: [], locals: [] }
          PatchPoint NoTracePoint
          v12:Fixnum[1] = Const Value(1)
          v14:Fixnum[2] = Const Value(2)
          v15:Any = Snapshot FrameState { pc: 0x1008, stack: [v12, v14], locals: [] }
          PatchPoint MethodRedefined(Integer@0x1010, +@0x1018, cme:0x1020)
          v35:Fixnum[6] = Const Value(6)
          v23:Any = Snapshot FrameState { pc: 0x1048, stack: [v35], locals: [] }
          CheckInterrupts
          Return v35
        ");
    }

    #[test]
    fn test_new_array_with_elements() {
        eval("def test(a, b) = [a, b]");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb0():
          Entries bb1, bb2
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v16:Any = Snapshot FrameState { pc: 0x1008, stack: [], locals: [a=v14, b=v15] }
          v17:Any = Snapshot FrameState { pc: 0x1010, stack: [], locals: [a=v14, b=v15] }
          PatchPoint NoTracePoint
          v19:Any = Snapshot FrameState { pc: 0x1018, stack: [v14], locals: [a=v14, b=v15] }
          v20:Any = Snapshot FrameState { pc: 0x1020, stack: [v14, v15], locals: [a=v14, b=v15] }
          v21:ArrayExact = NewArray v14, v15
          v22:Any = Snapshot FrameState { pc: 0x1028, stack: [v21], locals: [a=v14, b=v15] }
          PatchPoint NoTracePoint
          CheckInterrupts
          Return v21
        ");
    }

    #[test]
    fn test_send_direct_with_reordered_kwargs_has_snapshot() {
        eval("
            def foo(a:, b:, c:) = [a, b, c]
            def test = foo(c: 3, a: 1, b: 2)
            test
            test
        ");
        assert_snapshot!(optimized_hir_string("test"), @"
        fn test@<compiled>:3:
        bb0():
          Entries bb1, bb2
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v10:Any = Snapshot FrameState { pc: 0x1000, stack: [], locals: [] }
          PatchPoint NoTracePoint
          v13:Fixnum[3] = Const Value(3)
          v15:Fixnum[1] = Const Value(1)
          v17:Fixnum[2] = Const Value(2)
          v18:Any = Snapshot FrameState { pc: 0x1008, stack: [v8, v13, v15, v17], locals: [] }
          v25:Any = Snapshot FrameState { pc: 0x1008, stack: [v8, v15, v17, v13], locals: [] }
          PatchPoint MethodRedefined(Object@0x1010, foo@0x1018, cme:0x1020)
          v27:ObjectSubclass[class_exact*:Object@VALUE(0x1010)] = GuardType v8, ObjectSubclass[class_exact*:Object@VALUE(0x1010)]
          v28:BasicObject = SendDirect v27, 0x1048, :foo (0x1058), v15, v17, v13
          v20:Any = Snapshot FrameState { pc: 0x1060, stack: [v28], locals: [] }
          PatchPoint NoTracePoint
          CheckInterrupts
          Return v28
        ");
    }

    #[test]
    fn test_send_direct_with_kwargs_in_order_has_snapshot() {
        eval("
            def foo(a:, b:) = [a, b]
            def test = foo(a: 1, b: 2)
            test
            test
        ");
        assert_snapshot!(optimized_hir_string("test"), @"
        fn test@<compiled>:3:
        bb0():
          Entries bb1, bb2
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v10:Any = Snapshot FrameState { pc: 0x1000, stack: [], locals: [] }
          PatchPoint NoTracePoint
          v13:Fixnum[1] = Const Value(1)
          v15:Fixnum[2] = Const Value(2)
          v16:Any = Snapshot FrameState { pc: 0x1008, stack: [v8, v13, v15], locals: [] }
          PatchPoint MethodRedefined(Object@0x1010, foo@0x1018, cme:0x1020)
          v24:ObjectSubclass[class_exact*:Object@VALUE(0x1010)] = GuardType v8, ObjectSubclass[class_exact*:Object@VALUE(0x1010)]
          v25:BasicObject = SendDirect v24, 0x1048, :foo (0x1058), v13, v15
          v18:Any = Snapshot FrameState { pc: 0x1060, stack: [v25], locals: [] }
          PatchPoint NoTracePoint
          CheckInterrupts
          Return v25
        ");
    }

    #[test]
    fn test_send_direct_with_many_kwargs_no_reorder_snapshot() {
        eval("
            def foo(five, six, a:, b:, c:, d:, e:, f:) = [a, b, c, d, five, six, e, f]
            def test = foo(5, 6, d: 4, c: 3, a: 1, b: 2, e: 7, f: 8)
            test
            test
        ");
        assert_snapshot!(optimized_hir_string("test"), @"
        fn test@<compiled>:3:
        bb0():
          Entries bb1, bb2
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v10:Any = Snapshot FrameState { pc: 0x1000, stack: [], locals: [] }
          PatchPoint NoTracePoint
          v13:Fixnum[5] = Const Value(5)
          v15:Fixnum[6] = Const Value(6)
          v17:Fixnum[4] = Const Value(4)
          v19:Fixnum[3] = Const Value(3)
          v21:Fixnum[1] = Const Value(1)
          v23:Fixnum[2] = Const Value(2)
          v25:Fixnum[7] = Const Value(7)
          v27:Fixnum[8] = Const Value(8)
          v28:Any = Snapshot FrameState { pc: 0x1008, stack: [v8, v13, v15, v17, v19, v21, v23, v25, v27], locals: [] }
          v29:BasicObject = Send v8, :foo, v13, v15, v17, v19, v21, v23, v25, v27 # SendFallbackReason: Too many arguments for LIR
          v30:Any = Snapshot FrameState { pc: 0x1010, stack: [v29], locals: [] }
          PatchPoint NoTracePoint
          CheckInterrupts
          Return v29
        ");
    }
}

#[cfg(test)]
pub mod hir_build_tests {
    use super::*;
    use crate::options::set_call_threshold;
    use insta::assert_snapshot;

    fn iseq_contains_opcode(iseq: IseqPtr, expected_opcode: u32) -> bool {
        let iseq_size = unsafe { get_iseq_encoded_size(iseq) };
        let mut insn_idx = 0;
        while insn_idx < iseq_size {
            // Get the current pc and opcode
            let pc = unsafe { rb_iseq_pc_at_idx(iseq, insn_idx) };

            // try_into() call below is unfortunate. Maybe pick i32 instead of usize for opcodes.
            let opcode: u32 = unsafe { rb_iseq_opcode_at_pc(iseq, pc) }
                .try_into()
                .unwrap();
            if opcode == expected_opcode {
                return true;
            }
            insn_idx += insn_len(opcode as usize);
        }
        false
    }

    #[track_caller]
    pub fn assert_contains_opcode(method: &str, opcode: u32) {
        let iseq = crate::cruby::with_rubyvm(|| get_method_iseq("self", method));
        unsafe { crate::cruby::rb_zjit_profile_disable(iseq) };
        assert!(iseq_contains_opcode(iseq, opcode), "iseq {method} does not contain {}", insn_name(opcode as usize));
    }

    #[track_caller]
    fn assert_contains_opcodes(method: &str, opcodes: &[u32]) {
        let iseq = crate::cruby::with_rubyvm(|| get_method_iseq("self", method));
        unsafe { crate::cruby::rb_zjit_profile_disable(iseq) };
        for &opcode in opcodes {
            assert!(iseq_contains_opcode(iseq, opcode), "iseq {method} does not contain {}", insn_name(opcode as usize));
        }
    }

    /// Combine multiple hir_string() results to match all of them at once, which allows
    /// us to avoid running the set of zjit-test -> zjit-test-update multiple times.
    #[macro_export]
    macro_rules! hir_strings {
        ($( $s:expr ),+ $(,)?) => {{
            vec![$( hir_string($s) ),+].join("\n")
        }};
    }

    #[track_caller]
    fn hir_string(method: &str) -> String {
        hir_string_proc(&format!("{}.method(:{})", "self", method))
    }

    #[track_caller]
    fn hir_string_proc(proc: &str) -> String {
        let iseq = crate::cruby::with_rubyvm(|| get_proc_iseq(proc));
        unsafe { crate::cruby::rb_zjit_profile_disable(iseq) };
        let function = iseq_to_hir(iseq).unwrap();
        hir_string_function(&function)
    }

    #[track_caller]
    fn hir_string_function(function: &Function) -> String {
        format!("{}", FunctionPrinter::without_snapshot(function))
    }

    #[track_caller]
    fn assert_compile_fails(method: &str, reason: ParseError) {
        let iseq = crate::cruby::with_rubyvm(|| get_method_iseq("self", method));
        unsafe { crate::cruby::rb_zjit_profile_disable(iseq) };
        let result = iseq_to_hir(iseq);
        assert!(result.is_err(), "Expected an error but successfully compiled to HIR: {}", FunctionPrinter::without_snapshot(&result.unwrap()));
        assert_eq!(result.unwrap_err(), reason);
    }

    #[test]
    fn test_compile_optional() {
        eval("def test(x=1) = 123");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          BumpSP
          v5:CPtr = LoadPC
          v6:CPtr[CPtr(0x1008)] = Const CPtr(0x1010)
          v7:CBool = IsBitEqual v5, v6
          IfTrue v7, bb3(v1, v3)
          Jump bb5(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v12:BasicObject = LoadArg :self@0
          v13:NilClass = Const Value(nil)
          Jump bb3(v12, v13)
        bb3(v20:BasicObject, v21:BasicObject):
          v24:Fixnum[1] = Const Value(1)
          Jump bb5(v20, v24)
        bb4():
          EntryPoint JIT(1)
          BumpSP
          v17:BasicObject = LoadArg :self@0
          v18:BasicObject = LoadArg :x@1
          Jump bb5(v17, v18)
        bb5(v27:BasicObject, v28:BasicObject):
          v32:Fixnum[123] = Const Value(123)
          CheckInterrupts
          Return v32
        ");
    }

    #[test]
    fn test_putobject() {
        eval("def test = 123");
        assert_contains_opcode("test", YARVINSN_putobject);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Fixnum[123] = Const Value(123)
          CheckInterrupts
          Return v12
        ");
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
            test(1)
        "#);
        assert_contains_opcode("test", YARVINSN_checkmatch);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :o@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :o@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v16:NilClass = Const Value(nil)
          v20:BasicObject = GetConstantPath 0x1008
          v22:BasicObject = CheckMatch v12, v20, CASE
          CheckInterrupts
          v25:CBool = Test v22
          v26:Truthy = RefineType v22, Truthy
          IfTrue v25, bb4(v11, v12, v16, v12)
          v28:Falsy = RefineType v22, Falsy
          v33:Fixnum[2] = Const Value(2)
          CheckInterrupts
          Return v33
        bb4(v38:BasicObject, v39:BasicObject, v40:NilClass, v41:BasicObject):
          v46:Fixnum[1] = Const Value(1)
          CheckInterrupts
          Return v46
        ");
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
            test(1)
        "#);
        assert_contains_opcode("test", YARVINSN_checkmatch);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :o@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :o@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:ArrayExact[VALUE(0x1008)] = Const Value(VALUE(0x1008))
          v19:ArrayExact = ArrayDup v18
          v21:BasicObject = CheckMatch v12, v19, CASE|ARRAY
          CheckInterrupts
          v24:CBool = Test v21
          v25:Truthy = RefineType v21, Truthy
          IfTrue v24, bb4(v11, v12, v12)
          v27:Falsy = RefineType v21, Falsy
          v31:Fixnum[2] = Const Value(2)
          CheckInterrupts
          Return v31
        bb4(v36:BasicObject, v37:BasicObject, v38:BasicObject):
          v43:Fixnum[1] = Const Value(1)
          CheckInterrupts
          Return v43
        ");
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
            test
        "#);
        assert_contains_opcode("test", YARVINSN_checkmatch);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:4:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:NilClass = Const Value(nil)
          v14:ArrayExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          v15:ArrayExact = ArrayDup v14
          v17:BasicObject = CheckMatch v12, v15, WHEN|ARRAY
          CheckInterrupts
          v20:CBool = Test v17
          v21:Truthy = RefineType v17, Truthy
          IfTrue v20, bb4(v8)
          v23:Falsy = RefineType v17, Falsy
          v26:Fixnum[2] = Const Value(2)
          CheckInterrupts
          Return v26
        bb4(v31:BasicObject):
          v35:Fixnum[1] = Const Value(1)
          CheckInterrupts
          Return v35
        ");
    }

    #[test]
    fn test_new_array() {
        eval("def test = []");
        assert_contains_opcode("test", YARVINSN_newarray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:ArrayExact = NewArray
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_new_array_with_element() {
        eval("def test(a) = [a]");
        assert_contains_opcode("test", YARVINSN_newarray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:ArrayExact = NewArray v12
          CheckInterrupts
          Return v17
        ");
    }

    #[test]
    fn test_new_array_with_elements() {
        eval("def test(a, b) = [a, b]");
        assert_contains_opcode("test", YARVINSN_newarray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v21:ArrayExact = NewArray v14, v15
          CheckInterrupts
          Return v21
        ");
    }

    #[test]
    fn test_new_range_inclusive_with_one_element() {
        eval("def test(a) = (a..10)");
        assert_contains_opcode("test", YARVINSN_newrange);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:Fixnum[10] = Const Value(10)
          v19:RangeExact = NewRange v12 NewRangeInclusive v17
          CheckInterrupts
          Return v19
        ");
    }

    #[test]
    fn test_new_range_inclusive_with_two_elements() {
        eval("def test(a, b) = (a..b)");
        assert_contains_opcode("test", YARVINSN_newrange);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v21:RangeExact = NewRange v14 NewRangeInclusive v15
          CheckInterrupts
          Return v21
        ");
    }

    #[test]
    fn test_new_range_exclusive_with_one_element() {
        eval("def test(a) = (a...10)");
        assert_contains_opcode("test", YARVINSN_newrange);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:Fixnum[10] = Const Value(10)
          v19:RangeExact = NewRange v12 NewRangeExclusive v17
          CheckInterrupts
          Return v19
        ");
    }

    #[test]
    fn test_new_range_exclusive_with_two_elements() {
        eval("def test(a, b) = (a...b)");
        assert_contains_opcode("test", YARVINSN_newrange);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v21:RangeExact = NewRange v14 NewRangeExclusive v15
          CheckInterrupts
          Return v21
        ");
    }

    #[test]
    fn test_array_dup() {
        eval("def test = [1, 2, 3]");
        assert_contains_opcode("test", YARVINSN_duparray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:ArrayExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          v13:ArrayExact = ArrayDup v12
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_hash_dup() {
        eval("def test = {a: 1, b: 2}");
        assert_contains_opcode("test", YARVINSN_duphash);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:HashExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          v13:HashExact = HashDup v12
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_new_hash_empty() {
        eval("def test = {}");
        assert_contains_opcode("test", YARVINSN_newhash);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:HashExact = NewHash
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_new_hash_with_elements() {
        eval("def test(aval, bval) = {a: aval, b: bval}");
        assert_contains_opcode("test", YARVINSN_newhash);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :aval@0x1000
          v4:BasicObject = LoadField v2, :bval@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :aval@1
          v11:BasicObject = LoadArg :bval@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v19:StaticSymbol[:a] = Const Value(VALUE(0x1008))
          v22:StaticSymbol[:b] = Const Value(VALUE(0x1010))
          v25:HashExact = NewHash v19: v14, v22: v15
          CheckInterrupts
          Return v25
        ");
    }

    #[test]
    fn test_string_copy() {
        eval("def test = \"hello\"");
        assert_contains_opcode("test", YARVINSN_putchilledstring);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:StringExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          v13:StringExact = StringCopy v12
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_bignum() {
        eval("def test = 999999999999999999999999999999999999");
        assert_contains_opcode("test", YARVINSN_putobject);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Bignum[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_flonum() {
        eval("def test = 1.5");
        assert_contains_opcode("test", YARVINSN_putobject);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Flonum[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_heap_float() {
        eval("def test = 1.7976931348623157e+308");
        assert_contains_opcode("test", YARVINSN_putobject);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:HeapFloat[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_static_sym() {
        eval("def test = :foo");
        assert_contains_opcode("test", YARVINSN_putobject);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:StaticSymbol[:foo] = Const Value(VALUE(0x1000))
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_opt_plus() {
        eval("def test = 1+2");
        assert_contains_opcode("test", YARVINSN_opt_plus);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Fixnum[1] = Const Value(1)
          v14:Fixnum[2] = Const Value(2)
          v17:BasicObject = Send v12, :+, v14 # SendFallbackReason: Uncategorized(opt_plus)
          CheckInterrupts
          Return v17
        ");
    }

    #[test]
    fn test_opt_hash_freeze() {
        eval("
            def test = {}.freeze
        ");
        assert_contains_opcode("test", YARVINSN_opt_hash_freeze);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          PatchPoint BOPRedefined(HASH_REDEFINED_OP_FLAG, BOP_FREEZE)
          v13:HashExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_opt_hash_freeze_rewritten() {
        eval("
            class Hash
              def freeze; 5; end
            end
            def test = {}.freeze
        ");
        assert_contains_opcode("test", YARVINSN_opt_hash_freeze);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:5:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          SideExit PatchPoint(BOPRedefined(HASH_REDEFINED_OP_FLAG, BOP_FREEZE))
        ");
    }

    #[test]
    fn test_opt_ary_freeze() {
        eval("
            def test = [].freeze
        ");
        assert_contains_opcode("test", YARVINSN_opt_ary_freeze);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          PatchPoint BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_FREEZE)
          v13:ArrayExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_opt_ary_freeze_rewritten() {
        eval("
            class Array
              def freeze; 5; end
            end
            def test = [].freeze
        ");
        assert_contains_opcode("test", YARVINSN_opt_ary_freeze);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:5:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          SideExit PatchPoint(BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_FREEZE))
        ");
    }

    #[test]
    fn test_opt_str_freeze() {
        eval("
            def test = ''.freeze
        ");
        assert_contains_opcode("test", YARVINSN_opt_str_freeze);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          PatchPoint BOPRedefined(STRING_REDEFINED_OP_FLAG, BOP_FREEZE)
          v13:StringExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_opt_str_freeze_rewritten() {
        eval("
            class String
              def freeze; 5; end
            end
            def test = ''.freeze
        ");
        assert_contains_opcode("test", YARVINSN_opt_str_freeze);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:5:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          SideExit PatchPoint(BOPRedefined(STRING_REDEFINED_OP_FLAG, BOP_FREEZE))
        ");
    }

    #[test]
    fn test_opt_str_uminus() {
        eval("
            def test = -''
        ");
        assert_contains_opcode("test", YARVINSN_opt_str_uminus);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          PatchPoint BOPRedefined(STRING_REDEFINED_OP_FLAG, BOP_UMINUS)
          v13:StringExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_opt_str_uminus_rewritten() {
        eval("
            class String
              def -@; 5; end
            end
            def test = -''
        ");
        assert_contains_opcode("test", YARVINSN_opt_str_uminus);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:5:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          SideExit PatchPoint(BOPRedefined(STRING_REDEFINED_OP_FLAG, BOP_UMINUS))
        ");
    }

    #[test]
    fn test_setlocal_getlocal() {
        eval("
            def test
              a = 1
              a
            end
        ");
        assert_contains_opcodes("test", &[YARVINSN_getlocal_WC_0, YARVINSN_setlocal_WC_0]);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v2)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v7:BasicObject = LoadArg :self@0
          v8:NilClass = Const Value(nil)
          Jump bb3(v7, v8)
        bb3(v10:BasicObject, v11:NilClass):
          v15:Fixnum[1] = Const Value(1)
          CheckInterrupts
          Return v15
        ");
    }

    #[test]
    fn test_nested_setlocal_getlocal() {
        eval("
          l3 = 3
          _unused = _unused1 = nil
          1.times do |l2|
            _ = nil
            l2 = 2
            1.times do |l1|
              l1 = 1
              define_method(:test) do
                l1 = l2
                l2 = l1 + l2
                l3 = l2 + l3
              end
            end
          end
        ");
        assert_contains_opcodes(
            "test",
            &[YARVINSN_getlocal_WC_1, YARVINSN_setlocal_WC_1,
              YARVINSN_getlocal, YARVINSN_setlocal]);
        assert_snapshot!(hir_string("test"), @"
        fn block (3 levels) in <compiled>@<compiled>:10:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:CPtr = GetEP 2
          v13:BasicObject = LoadField v12, :l2@0x1000
          SetLocal :l1, l1, EP@3, v13
          v18:CPtr = GetEP 1
          v19:BasicObject = LoadField v18, :l1@0x1001
          v21:CPtr = GetEP 2
          v22:BasicObject = LoadField v21, :l2@0x1000
          v25:BasicObject = Send v19, :+, v22 # SendFallbackReason: Uncategorized(opt_plus)
          SetLocal :l2, l2, EP@4, v25
          v30:CPtr = GetEP 2
          v31:BasicObject = LoadField v30, :l2@0x1000
          v33:CPtr = GetEP 3
          v34:BasicObject = LoadField v33, :l3@0x1002
          v37:BasicObject = Send v31, :+, v34 # SendFallbackReason: Uncategorized(opt_plus)
          SetLocal :l3, l3, EP@5, v37
          CheckInterrupts
          Return v37
        "
        );
    }

    #[test]
    fn test_setlocal_in_default_args() {
        eval("
            def test(a = (b = 1)) = [a, b]
        ");
        assert_contains_opcode("test", YARVINSN_setlocal_WC_0);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:NilClass = Const Value(nil)
          BumpSP
          v6:CPtr = LoadPC
          v7:CPtr[CPtr(0x1008)] = Const CPtr(0x1010)
          v8:CBool = IsBitEqual v6, v7
          IfTrue v8, bb3(v1, v3, v4)
          Jump bb5(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v13:BasicObject = LoadArg :self@0
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v13, v14, v15)
        bb3(v23:BasicObject, v24:BasicObject, v25:NilClass):
          v29:Fixnum[1] = Const Value(1)
          Jump bb5(v23, v29, v29)
        bb4():
          EntryPoint JIT(1)
          BumpSP
          v19:BasicObject = LoadArg :self@0
          v20:BasicObject = LoadArg :a@1
          v21:NilClass = Const Value(nil)
          Jump bb5(v19, v20, v21)
        bb5(v34:BasicObject, v35:BasicObject, v36:NilClass|Fixnum):
          v42:ArrayExact = NewArray v35, v36
          CheckInterrupts
          Return v42
        ");
    }

    #[test]
    fn test_setlocal_in_default_args_with_tracepoint() {
        eval("
            def test(a = (b = 1)) = [a, b]
            TracePoint.new(:line) {}.enable
            test
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:NilClass = Const Value(nil)
          BumpSP
          v6:CPtr = LoadPC
          v7:CPtr[CPtr(0x1008)] = Const CPtr(0x1010)
          v8:CBool = IsBitEqual v6, v7
          IfTrue v8, bb3(v1, v3, v4)
          Jump bb5(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v13:BasicObject = LoadArg :self@0
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v13, v14, v15)
        bb3(v23:BasicObject, v24:BasicObject, v25:NilClass):
          SideExit UnhandledYARVInsn(trace_putobject_INT2FIX_1_)
        bb4():
          EntryPoint JIT(1)
          BumpSP
          v19:BasicObject = LoadArg :self@0
          v20:BasicObject = LoadArg :a@1
          v21:NilClass = Const Value(nil)
          Jump bb5(v19, v20, v21)
        bb5(v30:BasicObject, v31:BasicObject, v32:NilClass):
          v38:ArrayExact = NewArray v31, v32
          CheckInterrupts
          Return v38
        ");
    }

    #[test]
    fn test_setlocal_in_default_args_with_side_exit() {
        eval("
            def test(a = (def foo = nil)) = a
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          v5:CPtr = LoadPC
          v6:CPtr[CPtr(0x1008)] = Const CPtr(0x1010)
          v7:CBool = IsBitEqual v5, v6
          IfTrue v7, bb3(v1, v3)
          Jump bb5(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v12:BasicObject = LoadArg :self@0
          v13:NilClass = Const Value(nil)
          Jump bb3(v12, v13)
        bb3(v20:BasicObject, v21:BasicObject):
          SideExit UnhandledYARVInsn(definemethod)
        bb4():
          EntryPoint JIT(1)
          BumpSP
          v17:BasicObject = LoadArg :self@0
          v18:BasicObject = LoadArg :a@1
          Jump bb5(v17, v18)
        bb5(v26:BasicObject, v27:BasicObject):
          CheckInterrupts
          Return v27
        ");
    }

    #[test]
    fn test_setlocal_cyclic_default_args() {
        eval("
            def test = proc { |a=a| a }
        ");
        assert_snapshot!(hir_string_proc("test"), @"
        fn block in test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:NilClass = Const Value(nil)
          Jump bb3(v8, v9)
        bb4():
          EntryPoint JIT(1)
          BumpSP
          v13:BasicObject = LoadArg :self@0
          v14:BasicObject = LoadArg :a@1
          Jump bb3(v13, v14)
        bb3(v16:BasicObject, v17:BasicObject):
          CheckInterrupts
          Return v17
        ");
    }

    #[test]
    fn defined_ivar() {
        eval("
            def test = defined?(@foo)
        ");
        assert_contains_opcode("test", YARVINSN_definedivar);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:StringExact|NilClass = DefinedIvar v8, :@foo
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn if_defined_ivar() {
        eval("
            def test
              if defined?(@foo)
                3
              else
                4
              end
            end
        ");
        assert_contains_opcode("test", YARVINSN_definedivar);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:TrueClass|NilClass = DefinedIvar v8, :@foo
          CheckInterrupts
          v15:CBool = Test v12
          v16:NilClass = RefineType v12, Falsy
          IfFalse v15, bb4(v8)
          v18:TrueClass = RefineType v12, Truthy
          v21:Fixnum[3] = Const Value(3)
          CheckInterrupts
          Return v21
        bb4(v26:BasicObject):
          v30:Fixnum[4] = Const Value(4)
          CheckInterrupts
          Return v30
        ");
    }

    #[test]
    fn defined() {
        eval("
            def test = return defined?(SeaChange), defined?(favourite), defined?($ruby)
        ");
        assert_contains_opcode("test", YARVINSN_defined);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:NilClass = Const Value(nil)
          v14:StringExact|NilClass = Defined constant, v12
          v17:StringExact|NilClass = Defined func, v8
          v19:NilClass = Const Value(nil)
          v21:StringExact|NilClass = Defined global-variable, v19
          v23:ArrayExact = NewArray v14, v17, v21
          CheckInterrupts
          Return v23
        ");
    }

    #[test]
    fn defined_yield_in_method_local_iseq_returns_defined() {
        eval("
            def test = defined?(yield)
        ");
        assert_contains_opcode("test", YARVINSN_defined);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:NilClass = Const Value(nil)
          v14:StringExact|NilClass = Defined yield, v12
          CheckInterrupts
          Return v14
        ");
    }

    #[test]
    fn defined_yield_in_non_method_local_iseq_returns_nil() {
        eval("
            define_method(:test) { defined?(yield) }
        ");
        assert_contains_opcode("test", YARVINSN_defined);
        assert_snapshot!(hir_string("test"), @"
        fn block in <compiled>@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:NilClass = Const Value(nil)
          v14:NilClass = Const Value(nil)
          CheckInterrupts
          Return v14
        ");
    }

    #[test]
    fn test_return_const() {
        eval("
            def test(cond)
              if cond
                3
              else
                4
              end
            end
        ");
        assert_contains_opcode("test", YARVINSN_leave);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :cond@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :cond@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          CheckInterrupts
          v18:CBool = Test v12
          v19:Falsy = RefineType v12, Falsy
          IfFalse v18, bb4(v11, v19)
          v21:Truthy = RefineType v12, Truthy
          v24:Fixnum[3] = Const Value(3)
          CheckInterrupts
          Return v24
        bb4(v29:BasicObject, v30:Falsy):
          v34:Fixnum[4] = Const Value(4)
          CheckInterrupts
          Return v34
        ");
    }

    #[test]
    fn test_merge_const() {
        eval("
            def test(cond)
              if cond
                result = 3
              else
                result = 4
              end
              result
            end
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :cond@0x1000
          v4:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :cond@1
          v11:NilClass = Const Value(nil)
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:NilClass):
          CheckInterrupts
          v21:CBool = Test v14
          v22:Falsy = RefineType v14, Falsy
          IfFalse v21, bb4(v13, v22, v15)
          v24:Truthy = RefineType v14, Truthy
          v27:Fixnum[3] = Const Value(3)
          CheckInterrupts
          Jump bb5(v13, v24, v27)
        bb4(v32:BasicObject, v33:Falsy, v34:NilClass):
          v38:Fixnum[4] = Const Value(4)
          Jump bb5(v32, v33, v38)
        bb5(v41:BasicObject, v42:BasicObject, v43:Fixnum):
          CheckInterrupts
          Return v43
        ");
    }

    #[test]
    fn test_opt_plus_fixnum() {
        eval("
            def test(a, b) = a + b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_plus);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :+, v15 # SendFallbackReason: Uncategorized(opt_plus)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_opt_minus_fixnum() {
        eval("
            def test(a, b) = a - b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_minus);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :-, v15 # SendFallbackReason: Uncategorized(opt_minus)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_opt_mult_fixnum() {
        eval("
            def test(a, b) = a * b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_mult);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :*, v15 # SendFallbackReason: Uncategorized(opt_mult)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_opt_div_fixnum() {
        eval("
            def test(a, b) = a / b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_div);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :/, v15 # SendFallbackReason: Uncategorized(opt_div)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_opt_mod_fixnum() {
        eval("
            def test(a, b) = a % b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_mod);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :%, v15 # SendFallbackReason: Uncategorized(opt_mod)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_opt_eq_fixnum() {
        eval("
            def test(a, b) = a == b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_eq);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :==, v15 # SendFallbackReason: Uncategorized(opt_eq)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_opt_neq_fixnum() {
        eval("
            def test(a, b) = a != b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_neq);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :!=, v15 # SendFallbackReason: Uncategorized(opt_neq)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_opt_lt_fixnum() {
        eval("
            def test(a, b) = a < b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_lt);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :<, v15 # SendFallbackReason: Uncategorized(opt_lt)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_opt_le_fixnum() {
        eval("
            def test(a, b) = a <= b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_le);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :<=, v15 # SendFallbackReason: Uncategorized(opt_le)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_opt_gt_fixnum() {
        eval("
            def test(a, b) = a > b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_gt);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :>, v15 # SendFallbackReason: Uncategorized(opt_gt)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_loop() {
        eval("
            def test
              result = 0
              times = 10
              while times > 0
                result = result + 1
                times = times - 1
              end
              result
            end
            test
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:NilClass = Const Value(nil)
          v3:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v2, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:NilClass = Const Value(nil)
          v10:NilClass = Const Value(nil)
          Jump bb3(v8, v9, v10)
        bb3(v12:BasicObject, v13:NilClass, v14:NilClass):
          v18:Fixnum[0] = Const Value(0)
          v22:Fixnum[10] = Const Value(10)
          CheckInterrupts
          Jump bb5(v12, v18, v22)
        bb5(v28:BasicObject, v29:BasicObject, v30:BasicObject):
          v34:Fixnum[0] = Const Value(0)
          v37:BasicObject = Send v30, :>, v34 # SendFallbackReason: Uncategorized(opt_gt)
          CheckInterrupts
          v40:CBool = Test v37
          v41:Truthy = RefineType v37, Truthy
          IfTrue v40, bb4(v28, v29, v30)
          v43:Falsy = RefineType v37, Falsy
          v45:NilClass = Const Value(nil)
          CheckInterrupts
          Return v29
        bb4(v53:BasicObject, v54:BasicObject, v55:BasicObject):
          v60:Fixnum[1] = Const Value(1)
          v63:BasicObject = Send v54, :+, v60 # SendFallbackReason: Uncategorized(opt_plus)
          v68:Fixnum[1] = Const Value(1)
          v71:BasicObject = Send v55, :-, v68 # SendFallbackReason: Uncategorized(opt_minus)
          Jump bb5(v53, v63, v71)
        ");
    }

    #[test]
    fn test_opt_ge_fixnum() {
        eval("
            def test(a, b) = a >= b
            test(1, 2); test(1, 2)
        ");
        assert_contains_opcode("test", YARVINSN_opt_ge);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :>=, v15 # SendFallbackReason: Uncategorized(opt_ge)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn test_display_types() {
        eval("
            def test
              cond = true
              if cond
                3
              else
                4
              end
            end
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v2)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v7:BasicObject = LoadArg :self@0
          v8:NilClass = Const Value(nil)
          Jump bb3(v7, v8)
        bb3(v10:BasicObject, v11:NilClass):
          v15:TrueClass = Const Value(true)
          CheckInterrupts
          v21:CBool[true] = Test v15
          v22 = RefineType v15, Falsy
          IfFalse v21, bb4(v10, v22)
          v24:TrueClass = RefineType v15, Truthy
          v27:Fixnum[3] = Const Value(3)
          CheckInterrupts
          Return v27
        bb4(v32, v33):
          v37 = Const Value(4)
          CheckInterrupts
          Return v37
        ");
    }

    #[test]
    fn test_send_without_block() {
        eval("
            def bar(a, b)
              a+b
            end
            def test
              bar(2, 3)
            end
        ");
        assert_contains_opcode("test", YARVINSN_opt_send_without_block);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:6:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v13:Fixnum[2] = Const Value(2)
          v15:Fixnum[3] = Const Value(3)
          v17:BasicObject = Send v8, :bar, v13, v15 # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v17
        ");
    }

    #[test]
    fn test_send_with_block() {
        eval("
            def test(a)
              a.each {|item|
                item
              }
            end
            test([1,2,3])
        ");
        assert_contains_opcode("test", YARVINSN_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:BasicObject = Send v12, 0x1008, :each # SendFallbackReason: Uncategorized(send)
          v18:CPtr = GetEP 0
          v19:BasicObject = LoadField v18, :a@0x1030
          CheckInterrupts
          Return v17
        ");
    }

    #[test]
    fn test_intern_interpolated_symbol() {
        eval(r#"
            def test
              :"foo#{123}"
            end
        "#);
        assert_contains_opcode("test", YARVINSN_intern);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:StringExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          v14:Fixnum[123] = Const Value(123)
          v17:BasicObject = ObjToString v14
          v19:String = AnyToString v14, str: v17
          v21:StringExact = StringConcat v12, v19
          v23:Symbol = StringIntern v21
          CheckInterrupts
          Return v23
        ");
    }

    #[test]
    fn different_objects_get_addresses() {
        eval("def test = unknown_method([0], [1], '2', '2')");

        // The 2 string literals have the same address because they're deduped.
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v13:ArrayExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          v14:ArrayExact = ArrayDup v13
          v16:ArrayExact[VALUE(0x1008)] = Const Value(VALUE(0x1008))
          v17:ArrayExact = ArrayDup v16
          v19:StringExact[VALUE(0x1010)] = Const Value(VALUE(0x1010))
          v20:StringExact = StringCopy v19
          v22:StringExact[VALUE(0x1010)] = Const Value(VALUE(0x1010))
          v23:StringExact = StringCopy v22
          v25:BasicObject = Send v8, :unknown_method, v14, v17, v20, v23 # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v25
        ");
    }

    #[test]
    fn test_cant_compile_splat() {
        eval("
            def test(a) = foo(*a)
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:ArrayExact = ToArray v12
          v20:BasicObject = Send v11, :foo, v18 # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v20
        ");
    }

    #[test]
    fn test_compile_block_arg() {
        eval("
            def test(a) = foo(&a)
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:BasicObject = Send v11, 0x1008, :foo, v12 # SendFallbackReason: Uncategorized(send)
          CheckInterrupts
          Return v18
        ");
    }

    #[test]
    fn test_cant_compile_kwarg() {
        eval("
            def test(a) = foo(a: 1)
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:Fixnum[1] = Const Value(1)
          v19:BasicObject = Send v11, :foo, v17 # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v19
        ");
    }

    #[test]
    fn test_cant_compile_kw_splat() {
        eval("
            def test(a) = foo(**a)
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:BasicObject = Send v11, :foo, v12 # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v18
        ");
    }

    // TODO(max): Figure out how to generate a call with TAILCALL flag

    #[test]
    fn test_compile_super() {
        eval("
            def test = super()
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v13:BasicObject = InvokeSuper v8, 0x1000 # SendFallbackReason: Uncategorized(invokesuper)
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_compile_zsuper() {
        eval("
            def test = super
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v13:BasicObject = InvokeSuper v8, 0x1000 # SendFallbackReason: Uncategorized(invokesuper)
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_cant_compile_super_nil_blockarg() {
        eval("
            def test = super(&nil)
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v13:NilClass = Const Value(nil)
          v15:BasicObject = InvokeSuper v8, 0x1000, v13 # SendFallbackReason: Uncategorized(invokesuper)
          CheckInterrupts
          Return v15
        ");
    }

    #[test]
    fn test_compile_super_forward() {
        eval("
            def test(...) = super(...)
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :...@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :...@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:BasicObject = InvokeSuperForward v11, 0x1008, v12 # SendFallbackReason: Uncategorized(invokesuperforward)
          CheckInterrupts
          Return v18
        ");
    }

    #[test]
    fn test_compile_super_forward_with_block() {
        eval("
            def test(...) = super { |x| x }
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :...@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :...@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:BasicObject = InvokeSuperForward v11, 0x1008, v12 # SendFallbackReason: Uncategorized(invokesuperforward)
          v19:CPtr = GetEP 0
          v20:BasicObject = LoadField v19, :...@0x1010
          CheckInterrupts
          Return v18
        ");
    }

    #[test]
    fn test_compile_super_forward_with_use() {
        eval("
            def test(...) = super(...) + 1
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :...@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :...@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:BasicObject = InvokeSuperForward v11, 0x1008, v12 # SendFallbackReason: Uncategorized(invokesuperforward)
          v20:Fixnum[1] = Const Value(1)
          v23:BasicObject = Send v18, :+, v20 # SendFallbackReason: Uncategorized(opt_plus)
          CheckInterrupts
          Return v23
        ");
    }

    #[test]
    fn test_compile_super_forward_with_arg() {
        eval("
            def test(...) = super(1, ...)
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :...@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :...@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:Fixnum[1] = Const Value(1)
          v20:BasicObject = InvokeSuperForward v11, 0x1008, v17, v12 # SendFallbackReason: Uncategorized(invokesuperforward)
          CheckInterrupts
          Return v20
        ");
    }

    #[test]
    fn test_compile_forwardable() {
        eval("def forwardable(...) = nil");
        assert_snapshot!(hir_string("forwardable"), @"
        fn forwardable@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :...@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :...@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v16:NilClass = Const Value(nil)
          CheckInterrupts
          Return v16
        ");
    }

    // TODO(max): Figure out how to generate a call with OPT_SEND flag

    #[test]
    fn test_cant_compile_kw_splat_mut() {
        eval("
            def test(a) = foo **a, b: 1
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:Class[VMFrozenCore] = Const Value(VALUE(0x1008))
          v19:HashExact = NewHash
          PatchPoint NoEPEscape(test)
          v24:BasicObject = Send v17, :core#hash_merge_kwd, v19, v12 # SendFallbackReason: Uncategorized(opt_send_without_block)
          v26:Class[VMFrozenCore] = Const Value(VALUE(0x1008))
          v29:StaticSymbol[:b] = Const Value(VALUE(0x1010))
          v31:Fixnum[1] = Const Value(1)
          v33:BasicObject = Send v26, :core#hash_merge_ptr, v24, v29, v31 # SendFallbackReason: Uncategorized(opt_send_without_block)
          v35:BasicObject = Send v11, :foo, v33 # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v35
        ");
    }

    #[test]
    fn test_cant_compile_splat_mut() {
        eval("
            def test(*) = foo *, 1
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:ArrayExact = LoadField v2, :*@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :*@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:ArrayExact = ToNewArray v12
          v20:Fixnum[1] = Const Value(1)
          ArrayPush v18, v20
          v24:BasicObject = Send v11, :foo, v18 # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v24
        ");
    }

    #[test]
    fn test_compile_forwarding() {
        eval("
            def test(...) = foo(...)
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :...@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :...@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:BasicObject = SendForward v11, 0x1008, :foo, v12 # SendFallbackReason: Uncategorized(sendforward)
          CheckInterrupts
          Return v18
        ");
    }

    #[test]
    fn test_compile_triple_dots_with_positional_args() {
        eval("
            def test(a, ...) = foo(a, ...)
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:ArrayExact = LoadField v2, :*@0x1001
          v5:BasicObject = LoadField v2, :**@0x1002
          v6:BasicObject = LoadField v2, :&@0x1003
          v7:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6, v7)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v12:BasicObject = LoadArg :self@0
          v13:BasicObject = LoadArg :a@1
          v14:BasicObject = LoadArg :*@2
          v15:BasicObject = LoadArg :**@3
          v16:BasicObject = LoadArg :&@4
          v17:NilClass = Const Value(nil)
          Jump bb3(v12, v13, v14, v15, v16, v17)
        bb3(v19:BasicObject, v20:BasicObject, v21:BasicObject, v22:BasicObject, v23:BasicObject, v24:NilClass):
          v31:ArrayExact = ToArray v21
          PatchPoint NoEPEscape(test)
          v36:CPtr = GetEP 0
          v37:CInt64 = LoadField v36, :_env_data_index_flags@0x1004
          v38:CInt64 = GuardNoBitsSet v37, VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM=CUInt64(512)
          v39:CInt64 = LoadField v36, :_env_data_index_specval@0x1005
          v40:CInt64 = GuardAnyBitSet v39, CUInt64(1)
          v41:ObjectSubclass[BlockParamProxy] = Const Value(VALUE(0x1008))
          SideExit SplatKwNotProfiled
        ");
    }

    #[test]
    fn test_opt_new() {
        eval("
            class C; end
            def test = C.new
        ");
        assert_contains_opcode("test", YARVINSN_opt_new);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:BasicObject = GetConstantPath 0x1000
          v14:NilClass = Const Value(nil)
          v17:CBool = IsMethodCFunc v12, :new
          IfFalse v17, bb4(v8, v14, v12)
          v19:HeapBasicObject = ObjectAlloc v12
          v21:BasicObject = Send v19, :initialize # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Jump bb5(v8, v19, v21)
        bb4(v25:BasicObject, v26:NilClass, v27:BasicObject):
          v30:BasicObject = Send v27, :new # SendFallbackReason: Uncategorized(opt_send_without_block)
          Jump bb5(v25, v30, v26)
        bb5(v33:BasicObject, v34:BasicObject, v35:BasicObject):
          CheckInterrupts
          Return v34
        ");
    }

    #[test]
    fn test_opt_newarray_send_max_no_elements() {
        eval("
            def test = [].max
        ");
        // TODO(max): Rewrite to nil
        assert_contains_opcode("test", YARVINSN_opt_newarray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          PatchPoint BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_MAX)
          v13:BasicObject = ArrayMax
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_opt_newarray_send_max() {
        eval("
            def test(a,b) = [a,b].max
        ");
        assert_contains_opcode("test", YARVINSN_opt_newarray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          PatchPoint BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_MAX)
          v22:BasicObject = ArrayMax v14, v15
          CheckInterrupts
          Return v22
        ");
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
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:9:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          SideExit PatchPoint(BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_MAX))
        ");
    }

    #[test]
    fn test_opt_newarray_send_min() {
        eval("
            def test(a,b)
              sum = a+b
              result = [a,b].min
              puts [1,2,3]
              result
            end
        ");
        assert_contains_opcode("test", YARVINSN_opt_newarray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          v5:NilClass = Const Value(nil)
          v6:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v11:BasicObject = LoadArg :self@0
          v12:BasicObject = LoadArg :a@1
          v13:BasicObject = LoadArg :b@2
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v11, v12, v13, v14, v15)
        bb3(v17:BasicObject, v18:BasicObject, v19:BasicObject, v20:NilClass, v21:NilClass):
          v28:BasicObject = Send v18, :+, v19 # SendFallbackReason: Uncategorized(opt_plus)
          SideExit UnhandledNewarraySend(MIN)
        ");
    }

    #[test]
    fn test_opt_newarray_send_hash() {
        eval("
            def test(a,b)
              sum = a+b
              result = [a,b].hash
              puts [1,2,3]
              result
            end
        ");
        assert_contains_opcode("test", YARVINSN_opt_newarray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          v5:NilClass = Const Value(nil)
          v6:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v11:BasicObject = LoadArg :self@0
          v12:BasicObject = LoadArg :a@1
          v13:BasicObject = LoadArg :b@2
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v11, v12, v13, v14, v15)
        bb3(v17:BasicObject, v18:BasicObject, v19:BasicObject, v20:NilClass, v21:NilClass):
          v28:BasicObject = Send v18, :+, v19 # SendFallbackReason: Uncategorized(opt_plus)
          PatchPoint BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_HASH)
          v35:Fixnum = ArrayHash v18, v19
          PatchPoint NoEPEscape(test)
          v42:ArrayExact[VALUE(0x1008)] = Const Value(VALUE(0x1008))
          v43:ArrayExact = ArrayDup v42
          v45:BasicObject = Send v17, :puts, v43 # SendFallbackReason: Uncategorized(opt_send_without_block)
          PatchPoint NoEPEscape(test)
          CheckInterrupts
          Return v35
        ");
    }

    #[test]
    fn test_opt_newarray_send_hash_redefined() {
        eval("
            Array.class_eval { def hash = 42 }

            def test(a,b)
              sum = a+b
              result = [a,b].hash
              puts [1,2,3]
              result
            end
        ");
        assert_contains_opcode("test", YARVINSN_opt_newarray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:5:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          v5:NilClass = Const Value(nil)
          v6:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v11:BasicObject = LoadArg :self@0
          v12:BasicObject = LoadArg :a@1
          v13:BasicObject = LoadArg :b@2
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v11, v12, v13, v14, v15)
        bb3(v17:BasicObject, v18:BasicObject, v19:BasicObject, v20:NilClass, v21:NilClass):
          v28:BasicObject = Send v18, :+, v19 # SendFallbackReason: Uncategorized(opt_plus)
          SideExit PatchPoint(BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_HASH))
        ");
    }

    #[test]
    fn test_opt_newarray_send_pack() {
        eval("
            def test(a,b)
              sum = a+b
              result = [a,b].pack 'C'
              puts [1,2,3]
              result
            end
        ");
        assert_contains_opcode("test", YARVINSN_opt_newarray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          v5:NilClass = Const Value(nil)
          v6:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v11:BasicObject = LoadArg :self@0
          v12:BasicObject = LoadArg :a@1
          v13:BasicObject = LoadArg :b@2
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v11, v12, v13, v14, v15)
        bb3(v17:BasicObject, v18:BasicObject, v19:BasicObject, v20:NilClass, v21:NilClass):
          v28:BasicObject = Send v18, :+, v19 # SendFallbackReason: Uncategorized(opt_plus)
          v34:StringExact[VALUE(0x1008)] = Const Value(VALUE(0x1008))
          v35:StringExact = StringCopy v34
          SideExit UnhandledNewarraySend(PACK)
        ");
    }

    #[test]
    fn test_opt_newarray_send_pack_buffer() {
        eval(r#"
            def test(a,b)
              sum = a+b
              buf = ""
              [a,b].pack 'C', buffer: buf
              buf
            end
        "#);
        assert_contains_opcode("test", YARVINSN_opt_newarray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          v5:NilClass = Const Value(nil)
          v6:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v11:BasicObject = LoadArg :self@0
          v12:BasicObject = LoadArg :a@1
          v13:BasicObject = LoadArg :b@2
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v11, v12, v13, v14, v15)
        bb3(v17:BasicObject, v18:BasicObject, v19:BasicObject, v20:NilClass, v21:NilClass):
          v28:BasicObject = Send v18, :+, v19 # SendFallbackReason: Uncategorized(opt_plus)
          v32:StringExact[VALUE(0x1008)] = Const Value(VALUE(0x1008))
          v33:StringExact = StringCopy v32
          v39:StringExact[VALUE(0x1010)] = Const Value(VALUE(0x1010))
          v40:StringExact = StringCopy v39
          v42:CPtr = GetEP 0
          v43:BasicObject = LoadField v42, :buf@0x1018
          PatchPoint BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_PACK)
          v46:String = ArrayPackBuffer v18, v19, fmt: v40, buf: v43
          PatchPoint NoEPEscape(test)
          CheckInterrupts
          Return v33
        ");
    }

    #[test]
    fn test_opt_newarray_send_pack_buffer_redefined() {
        eval(r#"
            class Array
              def pack(fmt, buffer: nil) = 5
            end
            def test(a,b)
              sum = a+b
              buf = ""
              [a,b].pack 'C', buffer: buf
              buf
            end
        "#);
        assert_contains_opcode("test", YARVINSN_opt_newarray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:6:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          v5:NilClass = Const Value(nil)
          v6:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v11:BasicObject = LoadArg :self@0
          v12:BasicObject = LoadArg :a@1
          v13:BasicObject = LoadArg :b@2
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v11, v12, v13, v14, v15)
        bb3(v17:BasicObject, v18:BasicObject, v19:BasicObject, v20:NilClass, v21:NilClass):
          v28:BasicObject = Send v18, :+, v19 # SendFallbackReason: Uncategorized(opt_plus)
          v32:StringExact[VALUE(0x1008)] = Const Value(VALUE(0x1008))
          v33:StringExact = StringCopy v32
          v39:StringExact[VALUE(0x1010)] = Const Value(VALUE(0x1010))
          v40:StringExact = StringCopy v39
          v42:CPtr = GetEP 0
          v43:BasicObject = LoadField v42, :buf@0x1018
          SideExit PatchPoint(BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_PACK))
        ");
    }

    #[test]
    fn test_opt_newarray_send_include_p() {
        eval("
            def test(a,b)
              sum = a+b
              result = [a,b].include? b
              puts [1,2,3]
              result
            end
        ");
        assert_contains_opcode("test", YARVINSN_opt_newarray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          v5:NilClass = Const Value(nil)
          v6:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v11:BasicObject = LoadArg :self@0
          v12:BasicObject = LoadArg :a@1
          v13:BasicObject = LoadArg :b@2
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v11, v12, v13, v14, v15)
        bb3(v17:BasicObject, v18:BasicObject, v19:BasicObject, v20:NilClass, v21:NilClass):
          v28:BasicObject = Send v18, :+, v19 # SendFallbackReason: Uncategorized(opt_plus)
          PatchPoint BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_INCLUDE_P)
          v36:BoolExact = ArrayInclude v18, v19 | v19
          PatchPoint NoEPEscape(test)
          v43:ArrayExact[VALUE(0x1008)] = Const Value(VALUE(0x1008))
          v44:ArrayExact = ArrayDup v43
          v46:BasicObject = Send v17, :puts, v44 # SendFallbackReason: Uncategorized(opt_send_without_block)
          PatchPoint NoEPEscape(test)
          CheckInterrupts
          Return v36
        ");
    }

    #[test]
    fn test_opt_newarray_send_include_p_redefined() {
        eval("
            class Array
              alias_method :old_include?, :include?
              def include?(x)
                old_include?(x)
              end
            end

            def test(a,b)
              sum = a+b
              result = [a,b].include? b
              puts [1,2,3]
              result
            end
        ");
        assert_contains_opcode("test", YARVINSN_opt_newarray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:10:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          v5:NilClass = Const Value(nil)
          v6:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v11:BasicObject = LoadArg :self@0
          v12:BasicObject = LoadArg :a@1
          v13:BasicObject = LoadArg :b@2
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v11, v12, v13, v14, v15)
        bb3(v17:BasicObject, v18:BasicObject, v19:BasicObject, v20:NilClass, v21:NilClass):
          v28:BasicObject = Send v18, :+, v19 # SendFallbackReason: Uncategorized(opt_plus)
          SideExit PatchPoint(BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_INCLUDE_P))
        ");
    }

    #[test]
    fn test_opt_duparray_send_include_p() {
        eval("
            def test(x)
              [:a, :b].include?(x)
            end
        ");
        assert_contains_opcode("test", YARVINSN_opt_duparray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :x@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          PatchPoint BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_INCLUDE_P)
          v18:BoolExact = DupArrayInclude VALUE(0x1008) | v12
          CheckInterrupts
          Return v18
        ");
    }

    #[test]
    fn test_opt_duparray_send_include_p_redefined() {
        eval("
            class Array
              alias_method :old_include?, :include?
              def include?(x)
                old_include?(x)
              end
            end
            def test(x)
              [:a, :b].include?(x)
            end
        ");
        assert_contains_opcode("test", YARVINSN_opt_duparray_send);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:9:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :x@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          SideExit PatchPoint(BOPRedefined(ARRAY_REDEFINED_OP_FLAG, BOP_INCLUDE_P))
        ");
    }

    #[test]
    fn test_opt_length() {
        eval("
            def test(a,b) = [a,b].length
        ");
        assert_contains_opcode("test", YARVINSN_opt_length);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v21:ArrayExact = NewArray v14, v15
          v24:BasicObject = Send v21, :length # SendFallbackReason: Uncategorized(opt_length)
          CheckInterrupts
          Return v24
        ");
    }

    #[test]
    fn test_opt_size() {
        eval("
            def test(a,b) = [a,b].size
        ");
        assert_contains_opcode("test", YARVINSN_opt_size);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v21:ArrayExact = NewArray v14, v15
          v24:BasicObject = Send v21, :size # SendFallbackReason: Uncategorized(opt_size)
          CheckInterrupts
          Return v24
        ");
    }

    #[test]
    fn test_getconstant() {
        eval("
            def test(klass)
              klass::ARGV
            end
        ");
        assert_contains_opcode("test", YARVINSN_getconstant);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :klass@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :klass@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:FalseClass = Const Value(false)
          v19:BasicObject = GetConstant v12, :ARGV, v17
          CheckInterrupts
          Return v19
        ");
    }

    #[test]
    fn test_getinstancevariable() {
        eval("
            def test = @foo
            test
        ");
        assert_contains_opcode("test", YARVINSN_getinstancevariable);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          PatchPoint SingleRactorMode
          v13:BasicObject = GetIvar v8, :@foo
          CheckInterrupts
          Return v13
        ");
    }

    #[test]
    fn test_setinstancevariable() {
        eval("
            def test = @foo = 1
            test
        ");
        assert_contains_opcode("test", YARVINSN_setinstancevariable);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Fixnum[1] = Const Value(1)
          PatchPoint SingleRactorMode
          SetIvar v8, :@foo, v12
          v17:HeapBasicObject = RefineType v8, HeapBasicObject
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_set_ivar_rescue_frozen() {
        let result = eval("
            class Foo
              attr_accessor :bar
              def initialize
                @bar = 1
                freeze
              end
            end

            def test(foo)
              begin
                foo.bar = 2
              rescue FrozenError
              end
            end

            foo = Foo.new
            test(foo)
            test(foo)

            foo.bar
        ");
        assert_eq!(VALUE::fixnum_from_usize(1), result);
    }

    #[test]
    fn test_getclassvariable() {
        eval("
            class Foo
              def self.test = @@foo
            end
        ");
        let iseq = crate::cruby::with_rubyvm(|| get_method_iseq("Foo", "test"));
        assert!(iseq_contains_opcode(iseq, YARVINSN_getclassvariable), "iseq Foo.test does not contain getclassvariable");
        let function = iseq_to_hir(iseq).unwrap();
        assert_snapshot!(hir_string_function(&function), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:BasicObject = GetClassVar :@@foo
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_setclassvariable() {
        eval("
            class Foo
              def self.test = @@foo = 42
            end
        ");
        let iseq = crate::cruby::with_rubyvm(|| get_method_iseq("Foo", "test"));
        assert!(iseq_contains_opcode(iseq, YARVINSN_setclassvariable), "iseq Foo.test does not contain setclassvariable");
        let function = iseq_to_hir(iseq).unwrap();
        assert_snapshot!(hir_string_function(&function), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Fixnum[42] = Const Value(42)
          SetClassVar :@@foo, v12
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_setglobal() {
        eval("
            def test = $foo = 1
            test
        ");
        assert_contains_opcode("test", YARVINSN_setglobal);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Fixnum[1] = Const Value(1)
          SetGlobal :$foo, v12
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_getglobal() {
        eval("
            def test = $foo
            test
        ");
        assert_contains_opcode("test", YARVINSN_getglobal);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:BasicObject = GetGlobal :$foo
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_getblockparam() {
        eval("
            def test(&block) = block
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :block@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :block@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v16:CPtr = GetEP 0
          v17:CBool = IsBlockParamModified v16
          IfTrue v17, bb4(v11, v12)
          Jump bb5(v11, v12)
        bb4(v18:BasicObject, v19:BasicObject):
          v26:CPtr = GetEP 0
          v27:BasicObject = LoadField v26, :block@0x1001
          Jump bb6(v18, v27, v27)
        bb5(v21:BasicObject, v22:BasicObject):
          v29:BasicObject = GetBlockParam :block, l0, EP@3
          Jump bb6(v21, v29, v29)
        bb6(v31:BasicObject, v32:BasicObject, v33:BasicObject):
          CheckInterrupts
          Return v33
        ");
    }

    #[test]
    fn test_getblockparam_nested_block() {
        eval("
            def test(&block)
              proc do
                block
              end
            end
        ");
        assert_snapshot!(hir_string_proc("test"), @"
        fn block in test@<compiled>:4:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:CPtr = GetEP 1
          v13:CBool = IsBlockParamModified v12
          IfTrue v13, bb4(v8)
          Jump bb5(v8)
        bb4(v14:BasicObject):
          v20:CPtr = GetEP 1
          v21:BasicObject = LoadField v20, :block@0x1000
          Jump bb6(v14, v21)
        bb5(v16:BasicObject):
          v23:BasicObject = GetBlockParam :block, l1, EP@3
          Jump bb6(v16, v23)
        bb6(v25:BasicObject, v26:BasicObject):
          CheckInterrupts
          Return v26
        ");
    }

    #[test]
    fn test_splatkw_unprofiled_side_exits() {
        eval("
            def foo(**kw, &b) = kw
            def test(**kw, &b) = foo(**kw, &b)
        ");
        assert_contains_opcode("test", YARVINSN_splatkw);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :kw@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :kw@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v21:CPtr = GetEP 0
          v22:CInt64 = LoadField v21, :_env_data_index_flags@0x1002
          v23:CInt64 = GuardNoBitsSet v22, VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM=CUInt64(512)
          v24:CInt64 = LoadField v21, :_env_data_index_specval@0x1003
          v25:CInt64 = GuardAnyBitSet v24, CUInt64(1)
          v26:ObjectSubclass[BlockParamProxy] = Const Value(VALUE(0x1008))
          SideExit SplatKwNotProfiled
        ");
    }

    #[test]
    fn test_splatkw_nil_guards_nil() {
        eval("
            def foo(a, ...) = a
            def test(a, ...) = foo(a, ...)
            test(1)
        ");
        assert_contains_opcode("test", YARVINSN_splatkw);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:ArrayExact = LoadField v2, :*@0x1001
          v5:BasicObject = LoadField v2, :**@0x1002
          v6:BasicObject = LoadField v2, :&@0x1003
          v7:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6, v7)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v12:BasicObject = LoadArg :self@0
          v13:BasicObject = LoadArg :a@1
          v14:BasicObject = LoadArg :*@2
          v15:BasicObject = LoadArg :**@3
          v16:BasicObject = LoadArg :&@4
          v17:NilClass = Const Value(nil)
          Jump bb3(v12, v13, v14, v15, v16, v17)
        bb3(v19:BasicObject, v20:BasicObject, v21:BasicObject, v22:BasicObject, v23:BasicObject, v24:NilClass):
          v31:ArrayExact = ToArray v21
          PatchPoint NoEPEscape(test)
          v36:CPtr = GetEP 0
          v37:CInt64 = LoadField v36, :_env_data_index_flags@0x1004
          v38:CInt64 = GuardNoBitsSet v37, VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM=CUInt64(512)
          v39:CInt64 = LoadField v36, :_env_data_index_specval@0x1005
          v40:CInt64[0] = GuardBitEquals v39, CInt64(0)
          v41:NilClass = Const Value(nil)
          v43:NilClass = GuardType v22, NilClass
          v45:BasicObject = Send v19, 0x1004, :foo, v20, v31, v43, v41 # SendFallbackReason: Uncategorized(send)
          CheckInterrupts
          Return v45
        ");
    }

    #[test]
    fn test_splatkw_empty_hash_guards_hash() {
        eval("
            def foo(**kw, &b) = kw
            def test(**kw, &b) = foo(**kw, &b)
            test(&proc {})
        ");
        assert_contains_opcode("test", YARVINSN_splatkw);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :kw@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :kw@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v21:CPtr = GetEP 0
          v22:CInt64 = LoadField v21, :_env_data_index_flags@0x1002
          v23:CInt64 = GuardNoBitsSet v22, VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM=CUInt64(512)
          v24:CInt64 = LoadField v21, :_env_data_index_specval@0x1003
          v25:CInt64 = GuardAnyBitSet v24, CUInt64(1)
          v26:ObjectSubclass[BlockParamProxy] = Const Value(VALUE(0x1008))
          v28:HashExact = GuardType v14, HashExact
          v30:BasicObject = Send v13, 0x1002, :foo, v28, v26 # SendFallbackReason: Uncategorized(send)
          CheckInterrupts
          Return v30
        ");
    }

    #[test]
    fn test_splatkw_hash_guards_hash() {
        eval("
            def foo(**kw, &b) = kw
            def test(**kw, &b) = foo(**kw, &b)
            test(a: 1, &proc {})
        ");
        assert_contains_opcode("test", YARVINSN_splatkw);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :kw@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :kw@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v21:CPtr = GetEP 0
          v22:CInt64 = LoadField v21, :_env_data_index_flags@0x1002
          v23:CInt64 = GuardNoBitsSet v22, VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM=CUInt64(512)
          v24:CInt64 = LoadField v21, :_env_data_index_specval@0x1003
          v25:CInt64 = GuardAnyBitSet v24, CUInt64(1)
          v26:ObjectSubclass[BlockParamProxy] = Const Value(VALUE(0x1008))
          v28:HashExact = GuardType v14, HashExact
          v30:BasicObject = Send v13, 0x1002, :foo, v28, v26 # SendFallbackReason: Uncategorized(send)
          CheckInterrupts
          Return v30
        ");
    }

    #[test]
    fn test_splatkw_polymorphic_side_exits() {
        set_call_threshold(3);
        eval("
            def foo(a, ...) = a
            def test(a, ...) = foo(a, ...)
            test(1)
            test(1, b: 2)
        ");
        assert_contains_opcode("test", YARVINSN_splatkw);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:ArrayExact = LoadField v2, :*@0x1001
          v5:BasicObject = LoadField v2, :**@0x1002
          v6:BasicObject = LoadField v2, :&@0x1003
          v7:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6, v7)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v12:BasicObject = LoadArg :self@0
          v13:BasicObject = LoadArg :a@1
          v14:BasicObject = LoadArg :*@2
          v15:BasicObject = LoadArg :**@3
          v16:BasicObject = LoadArg :&@4
          v17:NilClass = Const Value(nil)
          Jump bb3(v12, v13, v14, v15, v16, v17)
        bb3(v19:BasicObject, v20:BasicObject, v21:BasicObject, v22:BasicObject, v23:BasicObject, v24:NilClass):
          v31:ArrayExact = ToArray v21
          PatchPoint NoEPEscape(test)
          v36:CPtr = GetEP 0
          v37:CInt64 = LoadField v36, :_env_data_index_flags@0x1004
          v38:CInt64 = GuardNoBitsSet v37, VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM=CUInt64(512)
          v39:CInt64 = LoadField v36, :_env_data_index_specval@0x1005
          v40:CInt64[0] = GuardBitEquals v39, CInt64(0)
          v41:NilClass = Const Value(nil)
          SideExit SplatKwPolymorphic
        ");
    }

    #[test]
    fn test_splatkw_with_non_hash_side_exits() {
        eval("
            def foo(a:) = a
            def test(obj, &block) = foo(**obj, &block)
            obj = Object.new
            def obj.to_hash = { a: 1 }
            test(obj) { 2 }
        ");
        assert_contains_opcode("test", YARVINSN_splatkw);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :obj@0x1000
          v4:BasicObject = LoadField v2, :block@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :obj@1
          v11:BasicObject = LoadArg :block@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v21:CPtr = GetEP 0
          v22:CInt64 = LoadField v21, :_env_data_index_flags@0x1002
          v23:CInt64 = GuardNoBitsSet v22, VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM=CUInt64(512)
          v24:CInt64 = LoadField v21, :_env_data_index_specval@0x1003
          v25:CInt64 = GuardAnyBitSet v24, CUInt64(1)
          v26:ObjectSubclass[BlockParamProxy] = Const Value(VALUE(0x1008))
          SideExit SplatKwNotNilOrHash
        ");
    }

    #[test]
    fn test_splatarray_mut() {
        eval("
            def test(a) = [*a]
        ");
        assert_contains_opcode("test", YARVINSN_splatarray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:ArrayExact = ToNewArray v12
          CheckInterrupts
          Return v17
        ");
    }

    #[test]
    fn test_concattoarray() {
        eval("
            def test(a) = [1, *a]
        ");
        assert_contains_opcode("test", YARVINSN_concattoarray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v16:Fixnum[1] = Const Value(1)
          v18:ArrayExact = NewArray v16
          v21:ArrayExact = ToArray v12
          ArrayExtend v18, v21
          CheckInterrupts
          Return v18
        ");
    }

    #[test]
    fn test_pushtoarray_one_element() {
        eval("
            def test(a) = [*a, 1]
        ");
        assert_contains_opcode("test", YARVINSN_pushtoarray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:ArrayExact = ToNewArray v12
          v19:Fixnum[1] = Const Value(1)
          ArrayPush v17, v19
          CheckInterrupts
          Return v17
        ");
    }

    #[test]
    fn test_pushtoarray_multiple_elements() {
        eval("
            def test(a) = [*a, 1, 2, 3]
        ");
        assert_contains_opcode("test", YARVINSN_pushtoarray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :a@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v17:ArrayExact = ToNewArray v12
          v19:Fixnum[1] = Const Value(1)
          v21:Fixnum[2] = Const Value(2)
          v23:Fixnum[3] = Const Value(3)
          ArrayPush v17, v19
          ArrayPush v17, v21
          ArrayPush v17, v23
          CheckInterrupts
          Return v17
        ");
    }

    #[test]
    fn test_aset() {
        eval("
            def test(a, b) = a[b] = 1
        ");
        assert_contains_opcode("test", YARVINSN_opt_aset);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v19:NilClass = Const Value(nil)
          v23:Fixnum[1] = Const Value(1)
          v27:BasicObject = Send v14, :[]=, v15, v23 # SendFallbackReason: Uncategorized(opt_aset)
          CheckInterrupts
          Return v23
        ");
    }

    #[test]
    fn test_aref() {
        eval("
            def test(a, b) = a[b]
        ");
        assert_contains_opcode("test", YARVINSN_opt_aref);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :a@0x1000
          v4:BasicObject = LoadField v2, :b@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :a@1
          v11:BasicObject = LoadArg :b@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :[], v15 # SendFallbackReason: Uncategorized(opt_aref)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn opt_empty_p() {
        eval("
            def test(x) = x.empty?
        ");
        assert_contains_opcode("test", YARVINSN_opt_empty_p);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :x@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:BasicObject = Send v12, :empty? # SendFallbackReason: Uncategorized(opt_empty_p)
          CheckInterrupts
          Return v18
        ");
    }

    #[test]
    fn opt_succ() {
        eval("
            def test(x) = x.succ
        ");
        assert_contains_opcode("test", YARVINSN_opt_succ);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :x@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:BasicObject = Send v12, :succ # SendFallbackReason: Uncategorized(opt_succ)
          CheckInterrupts
          Return v18
        ");
    }

    #[test]
    fn opt_and() {
        eval("
            def test(x, y) = x & y
        ");
        assert_contains_opcode("test", YARVINSN_opt_and);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          v4:BasicObject = LoadField v2, :y@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :x@1
          v11:BasicObject = LoadArg :y@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :&, v15 # SendFallbackReason: Uncategorized(opt_and)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn opt_or() {
        eval("
            def test(x, y) = x | y
        ");
        assert_contains_opcode("test", YARVINSN_opt_or);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          v4:BasicObject = LoadField v2, :y@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :x@1
          v11:BasicObject = LoadArg :y@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :|, v15 # SendFallbackReason: Uncategorized(opt_or)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    fn opt_not() {
        eval("
            def test(x) = !x
        ");
        assert_contains_opcode("test", YARVINSN_opt_not);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :x@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v18:BasicObject = Send v12, :! # SendFallbackReason: Uncategorized(opt_not)
          CheckInterrupts
          Return v18
        ");
    }

    #[test]
    fn opt_regexpmatch2() {
        eval("
            def test(regexp, matchee) = regexp =~ matchee
        ");
        assert_contains_opcode("test", YARVINSN_opt_regexpmatch2);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :regexp@0x1000
          v4:BasicObject = LoadField v2, :matchee@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :regexp@1
          v11:BasicObject = LoadArg :matchee@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v22:BasicObject = Send v14, :=~, v15 # SendFallbackReason: Uncategorized(opt_regexpmatch2)
          CheckInterrupts
          Return v22
        ");
    }

    #[test]
    // Tests for ConstBase requires either constant or class definition, both
    // of which can't be performed inside a method.
    fn test_putspecialobject_vm_core_and_cbase() {
        eval("
            def test
              alias aliased __callee__
            end
        ");
        assert_contains_opcode("test", YARVINSN_putspecialobject);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Class[VMFrozenCore] = Const Value(VALUE(0x1000))
          v14:BasicObject = PutSpecialObject CBase
          v16:StaticSymbol[:aliased] = Const Value(VALUE(0x1008))
          v18:StaticSymbol[:__callee__] = Const Value(VALUE(0x1010))
          v20:BasicObject = Send v12, :core#set_method_alias, v14, v16, v18 # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v20
        ");
    }

    #[test]
    fn opt_reverse() {
        eval("
            def reverse_odd
              a, b, c = @a, @b, @c
              [a, b, c]
            end

            def reverse_even
              a, b, c, d = @a, @b, @c, @d
              [a, b, c, d]
            end
        ");
        assert_contains_opcode("reverse_odd", YARVINSN_opt_reverse);
        assert_contains_opcode("reverse_even", YARVINSN_opt_reverse);
        assert_snapshot!(hir_strings!("reverse_odd", "reverse_even"), @"
        fn reverse_odd@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:NilClass = Const Value(nil)
          v3:NilClass = Const Value(nil)
          v4:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v2, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:NilClass = Const Value(nil)
          v11:NilClass = Const Value(nil)
          v12:NilClass = Const Value(nil)
          Jump bb3(v9, v10, v11, v12)
        bb3(v14:BasicObject, v15:NilClass, v16:NilClass, v17:NilClass):
          PatchPoint SingleRactorMode
          v22:BasicObject = GetIvar v14, :@a
          PatchPoint SingleRactorMode
          v25:BasicObject = GetIvar v14, :@b
          PatchPoint SingleRactorMode
          v28:BasicObject = GetIvar v14, :@c
          PatchPoint NoEPEscape(reverse_odd)
          v40:ArrayExact = NewArray v22, v25, v28
          CheckInterrupts
          Return v40

        fn reverse_even@<compiled>:8:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:NilClass = Const Value(nil)
          v3:NilClass = Const Value(nil)
          v4:NilClass = Const Value(nil)
          v5:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v2, v3, v4, v5)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v10:BasicObject = LoadArg :self@0
          v11:NilClass = Const Value(nil)
          v12:NilClass = Const Value(nil)
          v13:NilClass = Const Value(nil)
          v14:NilClass = Const Value(nil)
          Jump bb3(v10, v11, v12, v13, v14)
        bb3(v16:BasicObject, v17:NilClass, v18:NilClass, v19:NilClass, v20:NilClass):
          PatchPoint SingleRactorMode
          v25:BasicObject = GetIvar v16, :@a
          PatchPoint SingleRactorMode
          v28:BasicObject = GetIvar v16, :@b
          PatchPoint SingleRactorMode
          v31:BasicObject = GetIvar v16, :@c
          PatchPoint SingleRactorMode
          v34:BasicObject = GetIvar v16, :@d
          PatchPoint NoEPEscape(reverse_even)
          v48:ArrayExact = NewArray v25, v28, v31, v34
          CheckInterrupts
          Return v48
        ");
    }

    #[test]
    fn test_branchnil() {
        eval("
        def test(x) = x&.itself
        ");
        assert_contains_opcode("test", YARVINSN_branchnil);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :x@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          CheckInterrupts
          v19:CBool = IsNil v12
          v20:NilClass = Const Value(nil)
          IfTrue v19, bb4(v11, v20, v20)
          v22:NotNil = RefineType v12, NotNil
          v24:BasicObject = Send v22, :itself # SendFallbackReason: Uncategorized(opt_send_without_block)
          Jump bb4(v11, v22, v24)
        bb4(v26:BasicObject, v27:BasicObject, v28:BasicObject):
          CheckInterrupts
          Return v28
        ");
    }

    #[test]
    fn test_infer_nilability_from_branchif() {
        eval("
        def test(x)
          if x
            x&.itself
          else
            4
          end
        end
        ");
        assert_contains_opcode("test", YARVINSN_branchnil);
        // Note that IsNil has as its operand a value that we know statically *cannot* be nil
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :x@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          CheckInterrupts
          v18:CBool = Test v12
          v19:Falsy = RefineType v12, Falsy
          IfFalse v18, bb4(v11, v19)
          v21:Truthy = RefineType v12, Truthy
          CheckInterrupts
          v27:CBool[false] = IsNil v21
          v28:NilClass = Const Value(nil)
          IfTrue v27, bb5(v11, v28, v28)
          v30:Truthy = RefineType v21, NotNil
          v32:BasicObject = Send v30, :itself # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v32
        bb4(v37:BasicObject, v38:Falsy):
          v42:Fixnum[4] = Const Value(4)
          Jump bb5(v37, v38, v42)
        bb5(v44:BasicObject, v45:Falsy, v46:Fixnum[4]):
          CheckInterrupts
          Return v46
        ");
    }

    #[test]
    fn test_infer_truthiness_from_branch() {
        eval("
        def test(x)
          if x
            if x
              if x
                3
              else
                4
              end
            else
              5
            end
          else
            6
          end
        end
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :x@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          CheckInterrupts
          v18:CBool = Test v12
          v19:Falsy = RefineType v12, Falsy
          IfFalse v18, bb6(v11, v19)
          v21:Truthy = RefineType v12, Truthy
          CheckInterrupts
          v26:CBool[true] = Test v21
          v27 = RefineType v21, Falsy
          IfFalse v26, bb5(v11, v27)
          v29:Truthy = RefineType v21, Truthy
          CheckInterrupts
          v34:CBool[true] = Test v29
          v35 = RefineType v29, Falsy
          IfFalse v34, bb4(v11, v35)
          v37:Truthy = RefineType v29, Truthy
          v40:Fixnum[3] = Const Value(3)
          CheckInterrupts
          Return v40
        bb6(v45:BasicObject, v46:Falsy):
          v50:Fixnum[6] = Const Value(6)
          CheckInterrupts
          Return v50
        bb5(v55, v56):
          v60 = Const Value(5)
          CheckInterrupts
          Return v60
        bb4(v65, v66):
          v70 = Const Value(4)
          CheckInterrupts
          Return v70
        ");
    }

    #[test]
    fn test_invokebuiltin_delegate_annotated() {
        assert_contains_opcode("Float", YARVINSN_opt_invokebuiltin_delegate_leave);
        assert_snapshot!(hir_string("Float"), @"
        fn Float@<internal:kernel>:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :arg@0x1000
          v4:BasicObject = LoadField v2, :exception@0x1001
          v5:BasicObject = LoadField v2, :<empty>@0x1002
          BumpSP
          Jump bb3(v1, v3, v4, v5)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v10:BasicObject = LoadArg :self@0
          v11:BasicObject = LoadArg :arg@1
          v12:BasicObject = LoadArg :exception@2
          v13:CPtr = GetEP 0
          v14:BasicObject = LoadField v13, :<empty>@0x1003
          Jump bb3(v10, v11, v12, v14)
        bb3(v16:BasicObject, v17:BasicObject, v18:BasicObject, v19:BasicObject):
          v23:Float = InvokeBuiltin rb_f_float, v16, v17, v18
          Jump bb4(v16, v17, v18, v19, v23)
        bb4(v25:BasicObject, v26:BasicObject, v27:BasicObject, v28:BasicObject, v29:Float):
          CheckInterrupts
          Return v29
        ");
    }

    #[test]
    fn test_invokebuiltin_cexpr_annotated() {
        assert_contains_opcode("class", YARVINSN_opt_invokebuiltin_delegate_leave);
        assert_snapshot!(hir_string("class"), @"
        fn class@<internal:kernel>:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:HeapObject = InvokeBuiltin leaf <inline_expr>, v8
          Jump bb4(v8, v12)
        bb4(v14:BasicObject, v15:HeapObject):
          CheckInterrupts
          Return v15
        ");
    }

    #[test]
    fn test_invokebuiltin_delegate_with_args() {
        // Using an unannotated builtin to test InvokeBuiltin generation
        let iseq = crate::cruby::with_rubyvm(|| get_method_iseq("Dir", "open"));
        assert!(iseq_contains_opcode(iseq, YARVINSN_opt_invokebuiltin_delegate), "iseq Dir.open does not contain invokebuiltin");
        let function = iseq_to_hir(iseq).unwrap();
        assert_snapshot!(hir_string_function(&function), @"
        fn open@<internal:dir>:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :name@0x1000
          v4:BasicObject = LoadField v2, :encoding@0x1001
          v5:BasicObject = LoadField v2, :<empty>@0x1002
          v6:BasicObject = LoadField v2, :block@0x1003
          v7:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6, v7)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v12:BasicObject = LoadArg :self@0
          v13:BasicObject = LoadArg :name@1
          v14:BasicObject = LoadArg :encoding@2
          v15:CPtr = GetEP 0
          v16:BasicObject = LoadField v15, :<empty>@0x1003
          v17:BasicObject = LoadArg :block@3
          v18:NilClass = Const Value(nil)
          Jump bb3(v12, v13, v14, v16, v17, v18)
        bb3(v20:BasicObject, v21:BasicObject, v22:BasicObject, v23:BasicObject, v24:BasicObject, v25:NilClass):
          v29:BasicObject = InvokeBuiltin dir_s_open, v20, v21, v22
          PatchPoint NoEPEscape(open)
          v35:CPtr = GetEP 0
          v36:CInt64 = LoadField v35, :_env_data_index_flags@0x1004
          v37:CInt64 = GuardNoBitsSet v36, VM_FRAME_FLAG_MODIFIED_BLOCK_PARAM=CUInt64(512)
          v38:CInt64 = LoadField v35, :_env_data_index_specval@0x1005
          v39:CInt64 = GuardAnyBitSet v38, CUInt64(1)
          v40:ObjectSubclass[BlockParamProxy] = Const Value(VALUE(0x1008))
          CheckInterrupts
          v43:CBool[true] = Test v40
          v44 = RefineType v40, Falsy
          IfFalse v43, bb4(v20, v21, v22, v23, v24, v29)
          v46:ObjectSubclass[BlockParamProxy] = RefineType v40, Truthy
          v50:BasicObject = InvokeBlock, v29 # SendFallbackReason: Uncategorized(invokeblock)
          v53:BasicObject = InvokeBuiltin dir_s_close, v20, v29
          CheckInterrupts
          Return v50
        bb4(v59, v60, v61, v62, v63, v64):
          CheckInterrupts
          Return v64
        ");
    }

    #[test]
    fn test_invokebuiltin_delegate_without_args() {
        let iseq = crate::cruby::with_rubyvm(|| get_method_iseq("GC", "enable"));
        assert!(iseq_contains_opcode(iseq, YARVINSN_opt_invokebuiltin_delegate_leave), "iseq GC.enable does not contain invokebuiltin");
        let function = iseq_to_hir(iseq).unwrap();
        assert_snapshot!(hir_string_function(&function), @"
        fn enable@<internal:gc>:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:BasicObject = InvokeBuiltin gc_enable, v8
          Jump bb4(v8, v12)
        bb4(v14:BasicObject, v15:BasicObject):
          CheckInterrupts
          Return v15
        ");
    }

    #[test]
    fn test_invokebuiltin_with_args() {
        let iseq = crate::cruby::with_rubyvm(|| get_method_iseq("GC", "start"));
        assert!(iseq_contains_opcode(iseq, YARVINSN_invokebuiltin), "iseq GC.start does not contain invokebuiltin");
        let function = iseq_to_hir(iseq).unwrap();
        assert_snapshot!(hir_string_function(&function), @"
        fn start@<internal:gc>:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :full_mark@0x1000
          v4:BasicObject = LoadField v2, :immediate_mark@0x1001
          v5:BasicObject = LoadField v2, :immediate_sweep@0x1002
          v6:BasicObject = LoadField v2, :<empty>@0x1003
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v11:BasicObject = LoadArg :self@0
          v12:BasicObject = LoadArg :full_mark@1
          v13:BasicObject = LoadArg :immediate_mark@2
          v14:BasicObject = LoadArg :immediate_sweep@3
          v15:CPtr = GetEP 0
          v16:BasicObject = LoadField v15, :<empty>@0x1004
          Jump bb3(v11, v12, v13, v14, v16)
        bb3(v18:BasicObject, v19:BasicObject, v20:BasicObject, v21:BasicObject, v22:BasicObject):
          v29:FalseClass = Const Value(false)
          v31:BasicObject = InvokeBuiltin gc_start_internal, v18, v19, v20, v21, v29
          CheckInterrupts
          Return v31
        ");
    }

    #[test]
    fn test_invoke_leaf_builtin_symbol_name() {
        let iseq = crate::cruby::with_rubyvm(|| get_instance_method_iseq("Symbol", "name"));
        let function = iseq_to_hir(iseq).unwrap();
        assert_snapshot!(hir_string_function(&function), @"
        fn name@<internal:symbol>:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:StringExact = InvokeBuiltin leaf <inline_expr>, v8
          Jump bb4(v8, v12)
        bb4(v14:BasicObject, v15:StringExact):
          CheckInterrupts
          Return v15
        ");
    }

    #[test]
    fn test_invoke_leaf_builtin_symbol_to_s() {
        let iseq = crate::cruby::with_rubyvm(|| get_instance_method_iseq("Symbol", "to_s"));
        let function = iseq_to_hir(iseq).unwrap();
        assert_snapshot!(hir_string_function(&function), @"
        fn to_s@<internal:symbol>:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:StringExact = InvokeBuiltin leaf <inline_expr>, v8
          Jump bb4(v8, v12)
        bb4(v14:BasicObject, v15:StringExact):
          CheckInterrupts
          Return v15
        ");
    }

    #[test]
    fn dupn() {
        eval("
            def test(x) = (x[0, 1] ||= 2)
        ");
        assert_contains_opcode("test", YARVINSN_dupn);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          BumpSP
          Jump bb3(v1, v3)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v8:BasicObject = LoadArg :self@0
          v9:BasicObject = LoadArg :x@1
          Jump bb3(v8, v9)
        bb3(v11:BasicObject, v12:BasicObject):
          v16:NilClass = Const Value(nil)
          v19:Fixnum[0] = Const Value(0)
          v21:Fixnum[1] = Const Value(1)
          v24:BasicObject = Send v12, :[], v19, v21 # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          v28:CBool = Test v24
          v29:Truthy = RefineType v24, Truthy
          IfTrue v28, bb4(v11, v12, v16, v12, v19, v21, v29)
          v31:Falsy = RefineType v24, Falsy
          v34:Fixnum[2] = Const Value(2)
          v37:BasicObject = Send v12, :[]=, v19, v21, v34 # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v34
        bb4(v43:BasicObject, v44:BasicObject, v45:NilClass, v46:BasicObject, v47:Fixnum[0], v48:Fixnum[1], v49:Truthy):
          CheckInterrupts
          Return v49
        ");
    }

    #[test]
    fn test_objtostring_anytostring() {
        eval("
            def test = \"#{1}\"
        ");
        assert_contains_opcode("test", YARVINSN_objtostring);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:StringExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          v14:Fixnum[1] = Const Value(1)
          v17:BasicObject = ObjToString v14
          v19:String = AnyToString v14, str: v17
          v21:StringExact = StringConcat v12, v19
          CheckInterrupts
          Return v21
        ");
    }

    #[test]
    fn test_string_concat() {
        eval(r##"
            def test = "#{1}#{2}#{3}"
        "##);
        assert_contains_opcode("test", YARVINSN_concatstrings);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Fixnum[1] = Const Value(1)
          v15:BasicObject = ObjToString v12
          v17:String = AnyToString v12, str: v15
          v19:Fixnum[2] = Const Value(2)
          v22:BasicObject = ObjToString v19
          v24:String = AnyToString v19, str: v22
          v26:Fixnum[3] = Const Value(3)
          v29:BasicObject = ObjToString v26
          v31:String = AnyToString v26, str: v29
          v33:StringExact = StringConcat v17, v24, v31
          CheckInterrupts
          Return v33
        ");
    }

    #[test]
    fn test_string_concat_empty() {
        eval(r##"
            def test = "#{}"
        "##);
        assert_contains_opcode("test", YARVINSN_concatstrings);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:StringExact[VALUE(0x1000)] = Const Value(VALUE(0x1000))
          v14:NilClass = Const Value(nil)
          v17:BasicObject = ObjToString v14
          v19:String = AnyToString v14, str: v17
          v21:StringExact = StringConcat v12, v19
          CheckInterrupts
          Return v21
        ");
    }

    #[test]
    fn test_toregexp() {
        eval(r##"
            def test = /#{1}#{2}#{3}/
        "##);
        assert_contains_opcode("test", YARVINSN_toregexp);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Fixnum[1] = Const Value(1)
          v15:BasicObject = ObjToString v12
          v17:String = AnyToString v12, str: v15
          v19:Fixnum[2] = Const Value(2)
          v22:BasicObject = ObjToString v19
          v24:String = AnyToString v19, str: v22
          v26:Fixnum[3] = Const Value(3)
          v29:BasicObject = ObjToString v26
          v31:String = AnyToString v26, str: v29
          v33:RegexpExact = ToRegexp v17, v24, v31
          CheckInterrupts
          Return v33
        ");
    }

    #[test]
    fn test_toregexp_with_options() {
        eval(r##"
            def test = /#{1}#{2}/mixn
        "##);
        assert_contains_opcode("test", YARVINSN_toregexp);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:Fixnum[1] = Const Value(1)
          v15:BasicObject = ObjToString v12
          v17:String = AnyToString v12, str: v15
          v19:Fixnum[2] = Const Value(2)
          v22:BasicObject = ObjToString v19
          v24:String = AnyToString v19, str: v22
          v26:RegexpExact = ToRegexp v17, v24, MULTILINE|IGNORECASE|EXTENDED|NOENCODING
          CheckInterrupts
          Return v26
        ");
    }

    #[test]
    fn throw() {
        eval("
            define_method(:throw_return) { return 1 }
            define_method(:throw_break) { break 2 }
        ");
        assert_contains_opcode("throw_return", YARVINSN_throw);
        assert_contains_opcode("throw_break", YARVINSN_throw);
        assert_snapshot!(hir_strings!("throw_return", "throw_break"), @"
        fn block in <compiled>@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v14:Fixnum[1] = Const Value(1)
          Throw TAG_RETURN, v14

        fn block in <compiled>@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v14:Fixnum[2] = Const Value(2)
          Throw TAG_BREAK, v14
        ");
    }

    #[test]
    fn test_invokeblock() {
        eval(r#"
            def test
              yield
            end
        "#);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:BasicObject = InvokeBlock # SendFallbackReason: Uncategorized(invokeblock)
          CheckInterrupts
          Return v12
        ");
    }

    #[test]
    fn test_invokeblock_with_args() {
        eval(r#"
            def test(x, y)
              yield x, y
            end
        "#);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :x@0x1000
          v4:BasicObject = LoadField v2, :y@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :x@1
          v11:BasicObject = LoadArg :y@2
          Jump bb3(v9, v10, v11)
        bb3(v13:BasicObject, v14:BasicObject, v15:BasicObject):
          v21:BasicObject = InvokeBlock, v14, v15 # SendFallbackReason: Uncategorized(invokeblock)
          CheckInterrupts
          Return v21
        ");
    }

    #[test]
    fn test_expandarray_no_splat() {
        eval(r#"
            def test(o)
              a, b = o
            end
        "#);
        assert_contains_opcode("test", YARVINSN_expandarray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :o@0x1000
          v4:NilClass = Const Value(nil)
          v5:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v10:BasicObject = LoadArg :self@0
          v11:BasicObject = LoadArg :o@1
          v12:NilClass = Const Value(nil)
          v13:NilClass = Const Value(nil)
          Jump bb3(v10, v11, v12, v13)
        bb3(v15:BasicObject, v16:BasicObject, v17:NilClass, v18:NilClass):
          v24:ArrayExact = GuardType v16, ArrayExact
          v25:CInt64 = ArrayLength v24
          v26:CInt64[2] = Const CInt64(2)
          v27:CInt64 = GuardGreaterEq v25, v26
          v28:CInt64[1] = Const CInt64(1)
          v29:BasicObject = ArrayAref v24, v28
          v30:CInt64[0] = Const CInt64(0)
          v31:BasicObject = ArrayAref v24, v30
          PatchPoint NoEPEscape(test)
          CheckInterrupts
          Return v16
        ");
    }

    #[test]
    fn test_expandarray_splat() {
        eval(r#"
            def test(o)
              a, *b = o
            end
        "#);
        assert_contains_opcode("test", YARVINSN_expandarray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :o@0x1000
          v4:NilClass = Const Value(nil)
          v5:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v10:BasicObject = LoadArg :self@0
          v11:BasicObject = LoadArg :o@1
          v12:NilClass = Const Value(nil)
          v13:NilClass = Const Value(nil)
          Jump bb3(v10, v11, v12, v13)
        bb3(v15:BasicObject, v16:BasicObject, v17:NilClass, v18:NilClass):
          SideExit UnhandledYARVInsn(expandarray)
        ");
    }

    #[test]
    fn test_expandarray_splat_post() {
        eval(r#"
            def test(o)
              a, *b, c = o
            end
        "#);
        assert_contains_opcode("test", YARVINSN_expandarray);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:3:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :o@0x1000
          v4:NilClass = Const Value(nil)
          v5:NilClass = Const Value(nil)
          v6:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v11:BasicObject = LoadArg :self@0
          v12:BasicObject = LoadArg :o@1
          v13:NilClass = Const Value(nil)
          v14:NilClass = Const Value(nil)
          v15:NilClass = Const Value(nil)
          Jump bb3(v11, v12, v13, v14, v15)
        bb3(v17:BasicObject, v18:BasicObject, v19:NilClass, v20:NilClass, v21:NilClass):
          SideExit UnhandledYARVInsn(expandarray)
        ");
    }

    #[test]
    fn test_checkkeyword_tests_fixnum_bit() {
        eval(r#"
            def test(kw: 1 + 1) = kw
        "#);
        assert_contains_opcode("test", YARVINSN_checkkeyword);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :kw@0x1000
          v4:BasicObject = LoadField v2, :<empty>@0x1001
          BumpSP
          Jump bb3(v1, v3, v4)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v9:BasicObject = LoadArg :self@0
          v10:BasicObject = LoadArg :kw@1
          v11:CPtr = GetEP 0
          v12:BasicObject = LoadField v11, :<empty>@0x1002
          Jump bb3(v9, v10, v12)
        bb3(v14:BasicObject, v15:BasicObject, v16:BasicObject):
          v19:BoolExact = FixnumBitCheck v16, 0
          CheckInterrupts
          v22:CBool = Test v19
          v23:TrueClass = RefineType v19, Truthy
          IfTrue v22, bb4(v14, v15, v16)
          v25:FalseClass = RefineType v19, Falsy
          v27:Fixnum[1] = Const Value(1)
          v29:Fixnum[1] = Const Value(1)
          v32:BasicObject = Send v27, :+, v29 # SendFallbackReason: Uncategorized(opt_plus)
          Jump bb4(v14, v32, v16)
        bb4(v35:BasicObject, v36:BasicObject, v37:BasicObject):
          CheckInterrupts
          Return v36
        ");
    }

    #[test]
    fn test_checkkeyword_too_many_keywords_causes_side_exit() {
        eval(r#"
            def test(k1: k1, k2: k2, k3: k3, k4: k4, k5: k5,
            k6: k6, k7: k7, k8: k8, k9: k9, k10: k10, k11: k11,
            k12: k12, k13: k13, k14: k14, k15: k15, k16: k16,
            k17: k17, k18: k18, k19: k19, k20: k20, k21: k21,
            k22: k22, k23: k23, k24: k24, k25: k25, k26: k26,
            k27: k27, k28: k28, k29: k29, k30: k30, k31: k31,
            k32: k32, k33: k33) = k1
        "#);
        assert_contains_opcode("test", YARVINSN_checkkeyword);
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:CPtr = LoadSP
          v3:BasicObject = LoadField v2, :k1@0x1000
          v4:BasicObject = LoadField v2, :k2@0x1001
          v5:BasicObject = LoadField v2, :k3@0x1002
          v6:BasicObject = LoadField v2, :k4@0x1003
          v7:BasicObject = LoadField v2, :k5@0x1004
          v8:BasicObject = LoadField v2, :k6@0x1005
          v9:BasicObject = LoadField v2, :k7@0x1006
          v10:BasicObject = LoadField v2, :k8@0x1007
          v11:BasicObject = LoadField v2, :k9@0x1008
          v12:BasicObject = LoadField v2, :k10@0x1009
          v13:BasicObject = LoadField v2, :k11@0x100a
          v14:BasicObject = LoadField v2, :k12@0x100b
          v15:BasicObject = LoadField v2, :k13@0x100c
          v16:BasicObject = LoadField v2, :k14@0x100d
          v17:BasicObject = LoadField v2, :k15@0x100e
          v18:BasicObject = LoadField v2, :k16@0x100f
          v19:BasicObject = LoadField v2, :k17@0x1010
          v20:BasicObject = LoadField v2, :k18@0x1011
          v21:BasicObject = LoadField v2, :k19@0x1012
          v22:BasicObject = LoadField v2, :k20@0x1013
          v23:BasicObject = LoadField v2, :k21@0x1014
          v24:BasicObject = LoadField v2, :k22@0x1015
          v25:BasicObject = LoadField v2, :k23@0x1016
          v26:BasicObject = LoadField v2, :k24@0x1017
          v27:BasicObject = LoadField v2, :k25@0x1018
          v28:BasicObject = LoadField v2, :k26@0x1019
          v29:BasicObject = LoadField v2, :k27@0x101a
          v30:BasicObject = LoadField v2, :k28@0x101b
          v31:BasicObject = LoadField v2, :k29@0x101c
          v32:BasicObject = LoadField v2, :k30@0x101d
          v33:BasicObject = LoadField v2, :k31@0x101e
          v34:BasicObject = LoadField v2, :k32@0x101f
          v35:BasicObject = LoadField v2, :k33@0x1020
          v36:BasicObject = LoadField v2, :<empty>@0x1021
          BumpSP
          Jump bb3(v1, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v41:BasicObject = LoadArg :self@0
          v42:BasicObject = LoadArg :k1@1
          v43:BasicObject = LoadArg :k2@2
          v44:BasicObject = LoadArg :k3@3
          v45:BasicObject = LoadArg :k4@4
          v46:BasicObject = LoadArg :k5@5
          v47:BasicObject = LoadArg :k6@6
          v48:BasicObject = LoadArg :k7@7
          v49:BasicObject = LoadArg :k8@8
          v50:BasicObject = LoadArg :k9@9
          v51:BasicObject = LoadArg :k10@10
          v52:BasicObject = LoadArg :k11@11
          v53:BasicObject = LoadArg :k12@12
          v54:BasicObject = LoadArg :k13@13
          v55:BasicObject = LoadArg :k14@14
          v56:BasicObject = LoadArg :k15@15
          v57:BasicObject = LoadArg :k16@16
          v58:BasicObject = LoadArg :k17@17
          v59:BasicObject = LoadArg :k18@18
          v60:BasicObject = LoadArg :k19@19
          v61:BasicObject = LoadArg :k20@20
          v62:BasicObject = LoadArg :k21@21
          v63:BasicObject = LoadArg :k22@22
          v64:BasicObject = LoadArg :k23@23
          v65:BasicObject = LoadArg :k24@24
          v66:BasicObject = LoadArg :k25@25
          v67:BasicObject = LoadArg :k26@26
          v68:BasicObject = LoadArg :k27@27
          v69:BasicObject = LoadArg :k28@28
          v70:BasicObject = LoadArg :k29@29
          v71:BasicObject = LoadArg :k30@30
          v72:BasicObject = LoadArg :k31@31
          v73:BasicObject = LoadArg :k32@32
          v74:BasicObject = LoadArg :k33@33
          v75:CPtr = GetEP 0
          v76:BasicObject = LoadField v75, :<empty>@0x1022
          Jump bb3(v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v76)
        bb3(v78:BasicObject, v79:BasicObject, v80:BasicObject, v81:BasicObject, v82:BasicObject, v83:BasicObject, v84:BasicObject, v85:BasicObject, v86:BasicObject, v87:BasicObject, v88:BasicObject, v89:BasicObject, v90:BasicObject, v91:BasicObject, v92:BasicObject, v93:BasicObject, v94:BasicObject, v95:BasicObject, v96:BasicObject, v97:BasicObject, v98:BasicObject, v99:BasicObject, v100:BasicObject, v101:BasicObject, v102:BasicObject, v103:BasicObject, v104:BasicObject, v105:BasicObject, v106:BasicObject, v107:BasicObject, v108:BasicObject, v109:BasicObject, v110:BasicObject, v111:BasicObject, v112:BasicObject):
          SideExit TooManyKeywordParameters
        ");
    }

    #[test]
    fn test_array_each() {
        assert_snapshot!(hir_string_proc("Array.instance_method(:each)"), @"
        fn each@<internal:array>:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          v2:NilClass = Const Value(nil)
          BumpSP
          Jump bb3(v1, v2)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v7:BasicObject = LoadArg :self@0
          v8:NilClass = Const Value(nil)
          Jump bb3(v7, v8)
        bb3(v10:BasicObject, v11:NilClass):
          v15:NilClass = Const Value(nil)
          v17:TrueClass|NilClass = Defined yield, v15
          v19:CBool = Test v17
          v20:NilClass = RefineType v17, Falsy
          IfFalse v19, bb4(v10, v11)
          v22:TrueClass = RefineType v17, Truthy
          Jump bb6(v10, v11)
        bb4(v25:BasicObject, v26:NilClass):
          v30:BasicObject = InvokeBuiltin <inline_expr>, v25
          Jump bb5(v25, v26, v30)
        bb5(v42:BasicObject, v43:NilClass, v44:BasicObject):
          CheckInterrupts
          Return v44
        bb6(v32:BasicObject, v33:NilClass):
          v37:Fixnum[0] = Const Value(0)
          Jump bb8(v32, v37)
        bb8(v50:BasicObject, v51:Fixnum):
          v54:BoolExact = InvokeBuiltin rb_jit_ary_at_end, v50, v51
          v56:CBool = Test v54
          v57:FalseClass = RefineType v54, Falsy
          IfFalse v56, bb7(v50, v51)
          v59:TrueClass = RefineType v54, Truthy
          v61:NilClass = Const Value(nil)
          CheckInterrupts
          Return v50
        bb7(v69:BasicObject, v70:Fixnum):
          v74:BasicObject = InvokeBuiltin rb_jit_ary_at, v69, v70
          v76:BasicObject = InvokeBlock, v74 # SendFallbackReason: Uncategorized(invokeblock)
          v80:Fixnum = InvokeBuiltin rb_jit_fixnum_inc, v69, v70
          PatchPoint NoEPEscape(each)
          Jump bb8(v69, v80)
        ");
    }

    #[test]
    fn test_induce_side_exit() {
        eval("
          class NonTopLexicalScope
            RubyVM = 0
            def test
              RubyVM::ZJIT.induce_side_exit! # lexical scope dependant -- should not recognize
              ::RubyVM::ZJIT.induce_side_exit!
            end
          end
        ");
        assert_snapshot!(hir_string_proc("NonTopLexicalScope.instance_method(:test)"), @"
        fn test@<compiled>:5:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:BasicObject = GetConstantPath 0x1000
          v14:BasicObject = Send v12, :induce_side_exit! # SendFallbackReason: Uncategorized(opt_send_without_block)
          v18:BasicObject = GetConstantPath 0x1000
          SideExit DirectiveInduced
        ");
    }

    #[test]
    fn test_induce_side_exit_sensitive_to_constant_state() {
        eval("
          def test = ::RubyVM::ZJIT.induce_side_exit!
        ");
        assert!(hir_string("test").contains("SideExit DirectiveInduced"));
        eval("
          class RubyVM
            remove_const(:ZJIT)
          end
        ");
        let hir_after_removal = hir_string("test");
        assert_eq!(false, hir_string("test").contains("SideExit DirectiveInduced"), "should not work when the constant lookup would fail");
        assert_snapshot!(hir_after_removal, @"
        fn test@<compiled>:2:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:BasicObject = GetConstantPath 0x1000
          v14:BasicObject = Send v12, :induce_side_exit! # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v14
        ");
    }

    #[test]
    fn test_induce_side_exit_doesnt_work_when_method_after_undef() {
        eval("
          class << RubyVM::ZJIT
            undef :induce_side_exit!
          end
          def test = ::RubyVM::ZJIT.induce_side_exit!
        ");
        assert_eq!(false, hir_string("test").contains("SideExit DirectiveInduced"), "should not work after undef");
    }

    #[test]
    fn test_induce_compile_failure_does_not_trigger_autoload() {
        eval("
          class RubyVM
            remove_const(:ZJIT)
            autoload :ZJIT, 'a-file-that-does-not-exist-as-a-trap'
          end
          def test = ::RubyVM::ZJIT.induce_compile_failure!
        ");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:6:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:BasicObject = GetConstantPath 0x1000
          v14:BasicObject = Send v12, :induce_compile_failure! # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v14
        ");
    }

    #[test]
    fn test_induce_compile_failure_checks_full_const_path() {
        eval("def test = ::RubyVM::ZJIT::TooDeep.induce_compile_failure!");
        assert_snapshot!(hir_string("test"), @"
        fn test@<compiled>:1:
        bb1():
          EntryPoint interpreter
          v1:BasicObject = LoadSelf
          BumpSP
          Jump bb3(v1)
        bb2():
          EntryPoint JIT(0)
          BumpSP
          v6:BasicObject = LoadArg :self@0
          Jump bb3(v6)
        bb3(v8:BasicObject):
          v12:BasicObject = GetConstantPath 0x1000
          v14:BasicObject = Send v12, :induce_compile_failure! # SendFallbackReason: Uncategorized(opt_send_without_block)
          CheckInterrupts
          Return v14
        ");
    }

    #[test]
    fn test_induce_compile_failure() {
        eval("def test = ::RubyVM::ZJIT.induce_compile_failure!");
        assert_compile_fails("test", ParseError::DirectiveInduced);
    }

    #[test]
    fn test_induce_breakpoint() {
        eval("def test = ::RubyVM::ZJIT.induce_breakpoint!");
        assert!(hir_string("test").contains("BreakPoint"));
    }

    #[test]
    fn test_induce_breakpoint_returns_nil() {
        eval("
          def test
            x = ::RubyVM::ZJIT.induce_breakpoint!
            x
          end
        ");
        let hir = hir_string("test");
        assert!(hir.contains("BreakPoint"));
        assert!(hir.contains("Return v"));
    }
}

 /// Test successor and predecessor set computations.
 #[cfg(test)]
 mod control_flow_info_tests {
     use super::*;

     fn edge(target: BlockId) -> BranchEdge {
         BranchEdge { target, args: vec![] }
     }

     #[test]
     fn test_linked_list() {
        let mut function = Function::new(std::ptr::null());

        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);
        let bb3 = function.new_block(0);

        function.push_insn(bb0, Insn::Jump(edge(bb1)));
        function.push_insn(bb1, Insn::Jump(edge(bb2)));
        function.push_insn(bb2, Insn::Jump(edge(bb3)));

        let retval = function.push_insn(bb3, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb3, Insn::Return { val: retval });

        function.seal_entries();
        let cfi = ControlFlowInfo::new(&function);

        assert!(cfi.is_preceded_by(bb1, bb2));
        assert!(cfi.is_succeeded_by(bb2, bb1));
        assert!(cfi.predecessors(bb3).eq([bb2]));
     }

     #[test]
     fn test_diamond() {
        let mut function = Function::new(std::ptr::null());

        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);
        let bb3 = function.new_block(0);

        let v1 = function.push_insn(bb0, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb0, Insn::IfTrue { val: v1, target: edge(bb2)});
        function.push_insn(bb0, Insn::Jump(edge(bb1)));
        function.push_insn(bb1, Insn::Jump(edge(bb3)));
        function.push_insn(bb2, Insn::Jump(edge(bb3)));

        let retval = function.push_insn(bb3, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb3, Insn::Return { val: retval });

        function.seal_entries();
        let cfi = ControlFlowInfo::new(&function);

        assert!(cfi.is_preceded_by(bb2, bb3));
        assert!(cfi.is_preceded_by(bb1, bb3));
        assert!(!cfi.is_preceded_by(bb0, bb3));
        assert!(cfi.is_succeeded_by(bb1, bb0));
        assert!(cfi.is_succeeded_by(bb3, bb1));
     }

     #[test]
     fn test_cfi_deduplicated_successors_and_predecessors() {
         let mut function = Function::new(std::ptr::null());

         let bb0 = function.entry_block;
         let bb1 = function.new_block(0);

         // Construct two separate jump instructions.
         let v1 = function.push_insn(bb0, Insn::Const { val: Const::Value(Qfalse) });
         let _ = function.push_insn(bb0, Insn::IfTrue { val: v1, target: edge(bb1)});
         function.push_insn(bb0, Insn::Jump(edge(bb1)));

         let retval = function.push_insn(bb1, Insn::Const { val: Const::CBool(true) });
         function.push_insn(bb1, Insn::Return { val: retval });

         function.seal_entries();
         let cfi = ControlFlowInfo::new(&function);

         assert_eq!(cfi.predecessors(bb1).collect::<Vec<_>>().len(), 1);
         assert_eq!(cfi.successors(bb0).collect::<Vec<_>>().len(), 1);
     }
 }

 /// Test dominator set computations.
 #[cfg(test)]
 mod dom_tests {
     use super::*;
     use insta::assert_snapshot;

     fn edge(target: BlockId) -> BranchEdge {
         BranchEdge { target, args: vec![] }
     }

     fn assert_dominators_contains_self(function: &Function, dominators: &Dominators) {
         for (i, _) in function.blocks.iter().enumerate() {
             // Ensure that each dominating set contains the block itself.
             assert!(dominators.is_dominated_by(BlockId(i), BlockId(i)));
         }
     }

     #[test]
     fn test_linked_list() {
         let mut function = Function::new(std::ptr::null());

         let entries = function.entries_block;
         let bb0 = function.entry_block;
         let bb1 = function.new_block(0);
         let bb2 = function.new_block(0);
         let bb3 = function.new_block(0);

         function.push_insn(bb0, Insn::Jump(edge(bb1)));
         function.push_insn(bb1, Insn::Jump(edge(bb2)));
         function.push_insn(bb2, Insn::Jump(edge(bb3)));

         let retval = function.push_insn(bb3, Insn::Const { val: Const::CBool(true) });
         function.push_insn(bb3, Insn::Return { val: retval });

         function.seal_entries();
         assert_snapshot!(format!("{}", FunctionPrinter::without_snapshot(&function)), @"
         fn <manual>:
         bb1():
           Jump bb2()
         bb2():
           Jump bb3()
         bb3():
           Jump bb4()
         bb4():
           v3:Any = Const CBool(true)
           Return v3
         ");

         let dominators = Dominators::new(&function);
         assert_dominators_contains_self(&function, &dominators);
         assert_eq!(dominators.dominators(bb0), vec![entries, bb0]);
         assert_eq!(dominators.dominators(bb1), vec![entries, bb0, bb1]);
         assert_eq!(dominators.dominators(bb2), vec![entries, bb0, bb1, bb2]);
         assert_eq!(dominators.dominators(bb3), vec![entries, bb0, bb1, bb2, bb3]);
     }

     #[test]
     fn test_diamond() {
        let mut function = Function::new(std::ptr::null());

        let entries = function.entries_block;
        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);
        let bb3 = function.new_block(0);

        let val = function.push_insn(bb0, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb0, Insn::IfTrue { val, target: edge(bb1)});
        function.push_insn(bb0, Insn::Jump(edge(bb2)));

        function.push_insn(bb2, Insn::Jump(edge(bb3)));
        function.push_insn(bb1, Insn::Jump(edge(bb3)));

        let retval = function.push_insn(bb3, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb3, Insn::Return { val: retval });

        function.seal_entries();
        assert_snapshot!(format!("{}", FunctionPrinter::without_snapshot(&function)), @"
        fn <manual>:
        bb1():
          v0:Any = Const Value(false)
          IfTrue v0, bb2()
          Jump bb3()
        bb2():
          Jump bb4()
        bb3():
          Jump bb4()
        bb4():
          v5:Any = Const CBool(true)
          Return v5
        ");

        let dominators = Dominators::new(&function);
        assert_dominators_contains_self(&function, &dominators);
        assert_eq!(dominators.dominators(bb0), vec![entries, bb0]);
        assert_eq!(dominators.dominators(bb1), vec![entries, bb0, bb1]);
        assert_eq!(dominators.dominators(bb2), vec![entries, bb0, bb2]);
        assert_eq!(dominators.dominators(bb3), vec![entries, bb0, bb3]);
     }

    #[test]
    fn test_complex_cfg() {
        let mut function = Function::new(std::ptr::null());

        let entries = function.entries_block;
        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);
        let bb3 = function.new_block(0);
        let bb4 = function.new_block(0);
        let bb5 = function.new_block(0);
        let bb6 = function.new_block(0);
        let bb7 = function.new_block(0);

        function.push_insn(bb0, Insn::Jump(edge(bb1)));

        let v0 = function.push_insn(bb1, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb1, Insn::IfTrue { val: v0, target: edge(bb2)});
        function.push_insn(bb1, Insn::Jump(edge(bb4)));

        function.push_insn(bb2, Insn::Jump(edge(bb3)));

        let v1 = function.push_insn(bb3, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb3, Insn::IfTrue { val: v1, target: edge(bb5)});
        function.push_insn(bb3, Insn::Jump(edge(bb7)));

        function.push_insn(bb4, Insn::Jump(edge(bb5)));

        function.push_insn(bb5, Insn::Jump(edge(bb6)));

        function.push_insn(bb6, Insn::Jump(edge(bb7)));

        let retval = function.push_insn(bb7, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb7, Insn::Return { val: retval });

        function.seal_entries();
        assert_snapshot!(format!("{}", FunctionPrinter::without_snapshot(&function)), @"
        fn <manual>:
        bb1():
          Jump bb2()
        bb2():
          v1:Any = Const Value(false)
          IfTrue v1, bb3()
          Jump bb5()
        bb3():
          Jump bb4()
        bb4():
          v5:Any = Const Value(false)
          IfTrue v5, bb6()
          Jump bb8()
        bb5():
          Jump bb6()
        bb6():
          Jump bb7()
        bb7():
          Jump bb8()
        bb8():
          v11:Any = Const CBool(true)
          Return v11
        ");

        let dominators = Dominators::new(&function);
        assert_dominators_contains_self(&function, &dominators);
        assert_eq!(dominators.dominators(bb0), vec![entries, bb0]);
        assert_eq!(dominators.dominators(bb1), vec![entries, bb0, bb1]);
        assert_eq!(dominators.dominators(bb2), vec![entries, bb0, bb1, bb2]);
        assert_eq!(dominators.dominators(bb3), vec![entries, bb0, bb1, bb2, bb3]);
        assert_eq!(dominators.dominators(bb4), vec![entries, bb0, bb1, bb4]);
        assert_eq!(dominators.dominators(bb5), vec![entries, bb0, bb1, bb5]);
        assert_eq!(dominators.dominators(bb6), vec![entries, bb0, bb1, bb5, bb6]);
        assert_eq!(dominators.dominators(bb7), vec![entries, bb0, bb1, bb7]);
    }

    #[test]
    fn test_back_edges() {
        let mut function = Function::new(std::ptr::null());

        let entries = function.entries_block;
        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);
        let bb3 = function.new_block(0);
        let bb4 = function.new_block(0);
        let bb5 = function.new_block(0);

        let v0 = function.push_insn(bb0, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb0, Insn::IfTrue { val: v0, target: edge(bb1)});
        function.push_insn(bb0, Insn::Jump(edge(bb4)));

        let v1 = function.push_insn(bb1, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb1, Insn::IfTrue { val: v1, target: edge(bb2)});
        function.push_insn(bb1, Insn::Jump(edge(bb3)));

        function.push_insn(bb2, Insn::Jump(edge(bb3)));

        function.push_insn(bb4, Insn::Jump(edge(bb5)));

        let v2 = function.push_insn(bb5, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb5, Insn::IfTrue { val: v2, target: edge(bb3)});
        function.push_insn(bb5, Insn::Jump(edge(bb4)));

        let retval = function.push_insn(bb3, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb3, Insn::Return { val: retval });

        function.seal_entries();
        assert_snapshot!(format!("{}", FunctionPrinter::without_snapshot(&function)), @"
        fn <manual>:
        bb1():
          v0:Any = Const Value(false)
          IfTrue v0, bb2()
          Jump bb5()
        bb2():
          v3:Any = Const Value(false)
          IfTrue v3, bb3()
          Jump bb4()
        bb3():
          Jump bb4()
        bb5():
          Jump bb6()
        bb6():
          v8:Any = Const Value(false)
          IfTrue v8, bb4()
          Jump bb5()
        bb4():
          v11:Any = Const CBool(true)
          Return v11
        ");

        let dominators = Dominators::new(&function);
        assert_dominators_contains_self(&function, &dominators);
        assert_eq!(dominators.dominators(bb0), vec![entries, bb0]);
        assert_eq!(dominators.dominators(bb1), vec![entries, bb0, bb1]);
        assert_eq!(dominators.dominators(bb2), vec![entries, bb0, bb1, bb2]);
        assert_eq!(dominators.dominators(bb3), vec![entries, bb0, bb3]);
        assert_eq!(dominators.dominators(bb4), vec![entries, bb0, bb4]);
        assert_eq!(dominators.dominators(bb5), vec![entries, bb0, bb4, bb5]);
    }

    #[test]
    fn test_multiple_entry_blocks() {
        let mut function = Function::new(std::ptr::null());

        let entries = function.entries_block;
        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        function.jit_entry_blocks.push(bb1);
        let bb2 = function.new_block(0);

        function.push_insn(bb0, Insn::Jump(edge(bb2)));

        function.push_insn(bb1, Insn::Jump(edge(bb2)));

        let retval = function.push_insn(bb2, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb2, Insn::Return { val: retval });

        function.seal_entries();
        assert_snapshot!(format!("{}", FunctionPrinter::without_snapshot(&function)), @"
        fn <manual>:
        bb1():
          Jump bb3()
        bb2():
          Jump bb3()
        bb3():
          v2:Any = Const CBool(true)
          Return v2
        ");

        let dominators = Dominators::new(&function);
        assert_dominators_contains_self(&function, &dominators);

        assert_eq!(dominators.dominators(bb0), vec![entries, bb0]);
        assert_eq!(dominators.dominators(bb1), vec![entries, bb1]);
        assert_eq!(dominators.dominators(bb2), vec![entries, bb2]);

        assert!(!dominators.is_dominated_by(bb1, bb2));
    }
 }

 /// Test loop information computation.
#[cfg(test)]
mod loop_info_tests {
    use super::*;
    use insta::assert_snapshot;

    fn edge(target: BlockId) -> BranchEdge {
        BranchEdge { target, args: vec![] }
    }

    #[test]
    fn test_loop_depth() {
        // ┌─────┐
        // │ bb0 │
        // └──┬──┘
        //    │
        // ┌──▼──┐      ┌─────┐
        // │ bb2 ◄──────┼ bb1 ◄─┐
        // └──┬──┘      └─────┘ │
        //    └─────────────────┘
        let mut function = Function::new(std::ptr::null());

        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);

        function.push_insn(bb0, Insn::Jump(edge(bb2)));

        let val = function.push_insn(bb0, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb2, Insn::IfTrue { val, target: edge(bb1)});
        let retval = function.push_insn(bb2, Insn::Const { val: Const::CBool(true) });
        let _ = function.push_insn(bb2, Insn::Return { val: retval });

        function.push_insn(bb1, Insn::Jump(edge(bb2)));

        function.seal_entries();
        let cfi = ControlFlowInfo::new(&function);
        let dominators = Dominators::new(&function);
        let loop_info = LoopInfo::new(&cfi, &dominators);

        assert_snapshot!(format!("{}", FunctionPrinter::without_snapshot(&function)), @"
        fn <manual>:
        bb1():
          Jump bb3()
          v1:Any = Const Value(false)
        bb3():
          IfTrue v1, bb2()
          v3:Any = Const CBool(true)
          Return v3
        bb2():
          Jump bb3()
        ");

        assert!(loop_info.is_loop_header(bb2));
        assert!(loop_info.is_back_edge_source(bb1));
        assert_eq!(loop_info.loop_depth(bb1), 1);
    }

    #[test]
    fn test_nested_loops() {
        // ┌─────┐
        // │ bb0 ◄─────┐
        // └──┬──┘     │
        //    │        │
        // ┌──▼──┐     │
        // │ bb1 ◄───┐ │
        // └──┬──┘   │ │
        //    │      │ │
        // ┌──▼──┐   │ │
        // │ bb2 ┼───┘ │
        // └──┬──┘     │
        //    │        │
        // ┌──▼──┐     │
        // │ bb3 ┼─────┘
        // └──┬──┘
        //    │
        // ┌──▼──┐
        // │ bb4 │
        // └─────┘
        let mut function = Function::new(std::ptr::null());

        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);
        let bb3 = function.new_block(0);
        let bb4 = function.new_block(0);

        function.push_insn(bb0, Insn::Jump(edge(bb1)));

        function.push_insn(bb1, Insn::Jump(edge(bb2)));

        let cond = function.push_insn(bb2, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb2, Insn::IfTrue { val: cond, target: edge(bb1) });
        function.push_insn(bb2, Insn::Jump(edge(bb3)));

        let cond = function.push_insn(bb3, Insn::Const { val: Const::Value(Qtrue) });
        let _ = function.push_insn(bb3, Insn::IfTrue { val: cond, target: edge(bb0) });
        function.push_insn(bb3, Insn::Jump(edge(bb4)));

        let retval = function.push_insn(bb4, Insn::Const { val: Const::CBool(true) });
        let _ = function.push_insn(bb4, Insn::Return { val: retval });

        function.seal_entries();
        let cfi = ControlFlowInfo::new(&function);
        let dominators = Dominators::new(&function);
        let loop_info = LoopInfo::new(&cfi, &dominators);

        assert_snapshot!(format!("{}", FunctionPrinter::without_snapshot(&function)), @"
        fn <manual>:
        bb1():
          Jump bb2()
        bb2():
          Jump bb3()
        bb3():
          v2:Any = Const Value(false)
          IfTrue v2, bb2()
          Jump bb4()
        bb4():
          v5:Any = Const Value(true)
          IfTrue v5, bb1()
          Jump bb5()
        bb5():
          v8:Any = Const CBool(true)
          Return v8
        ");

        assert!(loop_info.is_loop_header(bb0));
        assert!(loop_info.is_loop_header(bb1));

        assert_eq!(loop_info.loop_depth(bb0), 1);
        assert_eq!(loop_info.loop_depth(bb1), 2);
        assert_eq!(loop_info.loop_depth(bb2), 2);
        assert_eq!(loop_info.loop_depth(bb3), 1);
        assert_eq!(loop_info.loop_depth(bb4), 0);

        assert!(loop_info.is_back_edge_source(bb2));
        assert!(loop_info.is_back_edge_source(bb3));
    }

    #[test]
    fn test_complex_loops() {
        //        ┌─────┐
        // ┌──────► bb0 │
        // │      └──┬──┘
        // │    ┌────┴────┐
        // │ ┌──▼──┐   ┌──▼──┐
        // │ │ bb1 ◄─┐ │ bb3 ◄─┐
        // │ └──┬──┘ │ └──┬──┘ │
        // │    │    │    │    │
        // │ ┌──▼──┐ │ ┌──▼──┐ │
        // │ │ bb2 ┼─┘ │ bb4 ┼─┘
        // │ └──┬──┘   └──┬──┘
        // │    └────┬────┘
        // │      ┌──▼──┐
        // └──────┼ bb5 │
        //        └──┬──┘
        //           │
        //        ┌──▼──┐
        //        │ bb6 │
        //        └─────┘
        let mut function = Function::new(std::ptr::null());

        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);
        let bb3 = function.new_block(0);
        let bb4 = function.new_block(0);
        let bb5 = function.new_block(0);
        let bb6 = function.new_block(0);

        let cond = function.push_insn(bb0, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb0, Insn::IfTrue { val: cond, target: edge(bb1) });
        function.push_insn(bb0, Insn::Jump(edge(bb3)));

        function.push_insn(bb1, Insn::Jump(edge(bb2)));

        let _ = function.push_insn(bb2, Insn::IfTrue { val: cond, target: edge(bb1) });
        function.push_insn(bb2, Insn::Jump(edge(bb5)));

        function.push_insn(bb3, Insn::Jump(edge(bb4)));

        let _ = function.push_insn(bb4, Insn::IfTrue { val: cond, target: edge(bb3) });
        function.push_insn(bb4, Insn::Jump(edge(bb5)));

        let _ = function.push_insn(bb5, Insn::IfTrue { val: cond, target: edge(bb0) });
        function.push_insn(bb5, Insn::Jump(edge(bb6)));

        let retval = function.push_insn(bb6, Insn::Const { val: Const::CBool(true) });
        let _ = function.push_insn(bb6, Insn::Return { val: retval });

        function.seal_entries();
        let cfi = ControlFlowInfo::new(&function);
        let dominators = Dominators::new(&function);
        let loop_info = LoopInfo::new(&cfi, &dominators);

        assert_snapshot!(format!("{}", FunctionPrinter::without_snapshot(&function)), @"
        fn <manual>:
        bb1():
          v0:Any = Const Value(false)
          IfTrue v0, bb2()
          Jump bb4()
        bb2():
          Jump bb3()
        bb3():
          IfTrue v0, bb2()
          Jump bb6()
        bb4():
          Jump bb5()
        bb5():
          IfTrue v0, bb4()
          Jump bb6()
        bb6():
          IfTrue v0, bb1()
          Jump bb7()
        bb7():
          v11:Any = Const CBool(true)
          Return v11
        ");

        assert!(loop_info.is_loop_header(bb0));
        assert!(loop_info.is_loop_header(bb1));
        assert!(!loop_info.is_loop_header(bb2));
        assert!(loop_info.is_loop_header(bb3));
        assert!(!loop_info.is_loop_header(bb5));
        assert!(!loop_info.is_loop_header(bb4));
        assert!(!loop_info.is_loop_header(bb6));

        assert_eq!(loop_info.loop_depth(bb0), 1);
        assert_eq!(loop_info.loop_depth(bb1), 2);
        assert_eq!(loop_info.loop_depth(bb2), 2);
        assert_eq!(loop_info.loop_depth(bb3), 2);
        assert_eq!(loop_info.loop_depth(bb4), 2);
        assert_eq!(loop_info.loop_depth(bb5), 1);
        assert_eq!(loop_info.loop_depth(bb6), 0);

        assert!(loop_info.is_back_edge_source(bb2));
        assert!(loop_info.is_back_edge_source(bb4));
        assert!(loop_info.is_back_edge_source(bb5));
    }

    #[test]
    fn linked_list_non_loop() {
        // ┌─────┐
        // │ bb0 │
        // └──┬──┘
        //    │
        // ┌──▼──┐
        // │ bb1 │
        // └──┬──┘
        //    │
        // ┌──▼──┐
        // │ bb2 │
        // └─────┘
        let mut function = Function::new(std::ptr::null());

        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);

        let _ = function.push_insn(bb0, Insn::Jump(edge(bb1)));
        let _ = function.push_insn(bb1, Insn::Jump(edge(bb2)));

        let retval = function.push_insn(bb2, Insn::Const { val: Const::CBool(true) });
        let _ = function.push_insn(bb2, Insn::Return { val: retval });

        function.seal_entries();
        let cfi = ControlFlowInfo::new(&function);
        let dominators = Dominators::new(&function);
        let loop_info = LoopInfo::new(&cfi, &dominators);

        assert_snapshot!(format!("{}", FunctionPrinter::without_snapshot(&function)), @"
        fn <manual>:
        bb1():
          Jump bb2()
        bb2():
          Jump bb3()
        bb3():
          v2:Any = Const CBool(true)
          Return v2
        ");

        assert!(!loop_info.is_loop_header(bb0));
        assert!(!loop_info.is_loop_header(bb1));
        assert!(!loop_info.is_loop_header(bb2));

        assert!(!loop_info.is_back_edge_source(bb0));
        assert!(!loop_info.is_back_edge_source(bb1));
        assert!(!loop_info.is_back_edge_source(bb2));

        assert_eq!(loop_info.loop_depth(bb0), 0);
        assert_eq!(loop_info.loop_depth(bb1), 0);
        assert_eq!(loop_info.loop_depth(bb2), 0);
    }

    #[test]
    fn triple_nested_loop() {
        // ┌─────┐
        // │ bb0 ◄──┐
        // └──┬──┘  │
        //    │     │
        // ┌──▼──┐  │
        // │ bb1 ◄─┐│
        // └──┬──┘ ││
        //    │    ││
        // ┌──▼──┐ ││
        // │ bb2 ◄┐││
        // └──┬──┘│││
        //    │   │││
        // ┌──▼──┐│││
        // │ bb3 ┼┘││
        // └──┬──┘ ││
        //    │    ││
        // ┌──▼──┐ ││
        // │ bb4 ┼─┘│
        // └──┬──┘  │
        //    │     │
        // ┌──▼──┐  │
        // │ bb5 ┼──┘
        // └─────┘
        let mut function = Function::new(std::ptr::null());

        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);
        let bb3 = function.new_block(0);
        let bb4 = function.new_block(0);
        let bb5 = function.new_block(0);

        let cond = function.push_insn(bb0, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb0, Insn::Jump(edge(bb1)));
        let _ = function.push_insn(bb1, Insn::Jump(edge(bb2)));
        let _ = function.push_insn(bb2, Insn::Jump(edge(bb3)));
        let _ = function.push_insn(bb3, Insn::Jump(edge(bb4)));
        let _ = function.push_insn(bb3, Insn::IfTrue {val: cond, target: edge(bb2)});
        let _ = function.push_insn(bb4, Insn::Jump(edge(bb5)));
        let _ = function.push_insn(bb4, Insn::IfTrue {val: cond, target: edge(bb1)});
        let _ = function.push_insn(bb5, Insn::IfTrue {val: cond, target: edge(bb0)});

        function.seal_entries();
        assert_snapshot!(format!("{}", FunctionPrinter::without_snapshot(&function)), @"
        fn <manual>:
        bb1():
          v0:Any = Const Value(false)
          Jump bb2()
        bb2():
          Jump bb3()
        bb3():
          Jump bb4()
        bb4():
          Jump bb5()
          IfTrue v0, bb3()
        bb5():
          Jump bb6()
          IfTrue v0, bb2()
        bb6():
          IfTrue v0, bb1()
        ");

        let cfi = ControlFlowInfo::new(&function);
        let dominators = Dominators::new(&function);
        let loop_info = LoopInfo::new(&cfi, &dominators);

        assert!(!loop_info.is_back_edge_source(bb0));
        assert!(!loop_info.is_back_edge_source(bb1));
        assert!(!loop_info.is_back_edge_source(bb2));
        assert!(loop_info.is_back_edge_source(bb3));
        assert!(loop_info.is_back_edge_source(bb4));
        assert!(loop_info.is_back_edge_source(bb5));

        assert_eq!(loop_info.loop_depth(bb0), 1);
        assert_eq!(loop_info.loop_depth(bb1), 2);
        assert_eq!(loop_info.loop_depth(bb2), 3);
        assert_eq!(loop_info.loop_depth(bb3), 3);
        assert_eq!(loop_info.loop_depth(bb4), 2);
        assert_eq!(loop_info.loop_depth(bb5), 1);

        assert!(loop_info.is_loop_header(bb0));
        assert!(loop_info.is_loop_header(bb1));
        assert!(loop_info.is_loop_header(bb2));
        assert!(!loop_info.is_loop_header(bb3));
        assert!(!loop_info.is_loop_header(bb4));
        assert!(!loop_info.is_loop_header(bb5));
    }
 }

/// Test dumping to iongraph format.
#[cfg(test)]
mod iongraph_tests {
    use super::*;
    use insta::assert_snapshot;

    fn edge(target: BlockId) -> BranchEdge {
        BranchEdge { target, args: vec![] }
    }

    #[test]
    fn test_simple_function() {
        let mut function = Function::new(std::ptr::null());
        let bb0 = function.entry_block;

        let retval = function.push_insn(bb0, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb0, Insn::Return { val: retval });

        let json = function.to_iongraph_pass("simple");
        assert_snapshot!(json.to_string(), @r#"{"name":"simple", "mir":{"blocks":[{"ptr":4096, "id":0, "loopDepth":0, "attributes":[], "predecessors":[], "successors":[], "instructions":[]}]}, "lir":{"blocks":[]}}"#);
    }

    #[test]
    fn test_two_blocks() {
        let mut function = Function::new(std::ptr::null());
        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);

        function.push_insn(bb0, Insn::Jump(edge(bb1)));

        let retval = function.push_insn(bb1, Insn::Const { val: Const::CBool(false) });
        function.push_insn(bb1, Insn::Return { val: retval });

        let json = function.to_iongraph_pass("two_blocks");
        assert_snapshot!(json.to_string(), @r#"{"name":"two_blocks", "mir":{"blocks":[{"ptr":4096, "id":0, "loopDepth":0, "attributes":[], "predecessors":[], "successors":[], "instructions":[]}]}, "lir":{"blocks":[]}}"#);
    }

    #[test]
    fn test_multiple_instructions() {
        let mut function = Function::new(std::ptr::null());
        let bb0 = function.entry_block;

        let val1 = function.push_insn(bb0, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb0, Insn::Return { val: val1 });

        let json = function.to_iongraph_pass("multiple_instructions");
        assert_snapshot!(json.to_string(), @r#"{"name":"multiple_instructions", "mir":{"blocks":[{"ptr":4096, "id":0, "loopDepth":0, "attributes":[], "predecessors":[], "successors":[], "instructions":[]}]}, "lir":{"blocks":[]}}"#);
    }

    #[test]
    fn test_conditional_branch() {
        let mut function = Function::new(std::ptr::null());
        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);

        let cond = function.push_insn(bb0, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb0, Insn::IfTrue { val: cond, target: edge(bb1) });

        let retval1 = function.push_insn(bb0, Insn::Const { val: Const::CBool(false) });
        function.push_insn(bb0, Insn::Return { val: retval1 });

        let retval2 = function.push_insn(bb1, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb1, Insn::Return { val: retval2 });

        let json = function.to_iongraph_pass("conditional_branch");
        assert_snapshot!(json.to_string(), @r#"{"name":"conditional_branch", "mir":{"blocks":[{"ptr":4096, "id":0, "loopDepth":0, "attributes":[], "predecessors":[], "successors":[], "instructions":[]}]}, "lir":{"blocks":[]}}"#);
    }

    #[test]
    fn test_loop_structure() {
        let mut function = Function::new(std::ptr::null());

        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);

        function.push_insn(bb0, Insn::Jump(edge(bb2)));

        let val = function.push_insn(bb0, Insn::Const { val: Const::Value(Qfalse) });
        let _ = function.push_insn(bb2, Insn::IfTrue { val, target: edge(bb1)});
        let retval = function.push_insn(bb2, Insn::Const { val: Const::CBool(true) });
        let _ = function.push_insn(bb2, Insn::Return { val: retval });

        function.push_insn(bb1, Insn::Jump(edge(bb2)));

        let json = function.to_iongraph_pass("loop_structure");
        assert_snapshot!(json.to_string(), @r#"{"name":"loop_structure", "mir":{"blocks":[{"ptr":4096, "id":0, "loopDepth":0, "attributes":[], "predecessors":[], "successors":[], "instructions":[]}]}, "lir":{"blocks":[]}}"#);
    }

    #[test]
    fn test_multiple_successors() {
        let mut function = Function::new(std::ptr::null());
        let bb0 = function.entry_block;
        let bb1 = function.new_block(0);
        let bb2 = function.new_block(0);

        let cond = function.push_insn(bb0, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb0, Insn::IfTrue { val: cond, target: edge(bb1) });
        function.push_insn(bb0, Insn::Jump(edge(bb2)));

        let retval1 = function.push_insn(bb1, Insn::Const { val: Const::CBool(true) });
        function.push_insn(bb1, Insn::Return { val: retval1 });

        let retval2 = function.push_insn(bb2, Insn::Const { val: Const::CBool(false) });
        function.push_insn(bb2, Insn::Return { val: retval2 });

        let json = function.to_iongraph_pass("multiple_successors");
        assert_snapshot!(json.to_string(), @r#"{"name":"multiple_successors", "mir":{"blocks":[{"ptr":4096, "id":0, "loopDepth":0, "attributes":[], "predecessors":[], "successors":[], "instructions":[]}]}, "lir":{"blocks":[]}}"#);
    }
 }
