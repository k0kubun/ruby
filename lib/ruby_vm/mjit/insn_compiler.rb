module RubyVM::MJIT
  class InsnCompiler
    # @param ocb [CodeBlock]
    # @param exit_compiler [RubyVM::MJIT::ExitCompiler]
    def initialize(cb, ocb, exit_compiler)
      @ocb = ocb
      @exit_compiler = exit_compiler
      @invariants = Invariants.new(cb, ocb, exit_compiler)
      # freeze # workaround a binding.irb issue. TODO: resurrect this
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    # @param insn `RubyVM::MJIT::Instruction`
    def compile(jit, ctx, asm, insn)
      asm.incr_counter(:mjit_insns_count)
      asm.comment("Insn: #{insn.name}")

      # 14/101
      case insn.name
      when :nop then nop(jit, ctx, asm)
      # getlocal
      # setlocal
      # getblockparam
      # setblockparam
      # getblockparamproxy
      # getspecial
      # setspecial
      when :getinstancevariable then getinstancevariable(jit, ctx, asm)
      # setinstancevariable
      # getclassvariable
      # setclassvariable
      # opt_getconstant_path
      # getconstant
      # setconstant
      # getglobal
      # setglobal
      when :putnil then putnil(jit, ctx, asm)
      when :putself then putself(jit, ctx, asm)
      when :putobject then putobject(jit, ctx, asm)
      # putspecialobject
      # putstring
      # concatstrings
      # anytostring
      # toregexp
      # intern
      # newarray
      # newarraykwsplat
      # duparray
      # duphash
      # expandarray
      # concatarray
      # splatarray
      # newhash
      # newrange
      # pop
      # dup
      # dupn
      # swap
      # opt_reverse
      # topn
      # setn
      # adjuststack
      # defined
      # checkmatch
      # checkkeyword
      # checktype
      # defineclass
      # definemethod
      # definesmethod
      # send
      when :opt_send_without_block then opt_send_without_block(jit, ctx, asm)
      # objtostring
      # opt_str_freeze
      # opt_nil_p
      # opt_str_uminus
      # opt_newarray_max
      # opt_newarray_min
      # invokesuper
      # invokeblock
      when :leave then leave(jit, ctx, asm)
      # throw
      # jump
      # branchif
      when :branchunless then branchunless(jit, ctx, asm)
      # branchnil
      # once
      # opt_case_dispatch
      when :opt_plus then opt_plus(jit, ctx, asm)
      when :opt_minus then opt_minus(jit, ctx, asm)
      # opt_mult
      # opt_div
      # opt_mod
      # opt_eq
      # opt_neq
      when :opt_lt then opt_lt(jit, ctx, asm)
      # opt_le
      # opt_gt
      # opt_ge
      # opt_ltlt
      # opt_and
      # opt_or
      # opt_aref
      # opt_aset
      # opt_aset_with
      # opt_aref_with
      # opt_length
      # opt_size
      # opt_empty_p
      # opt_succ
      # opt_not
      # opt_regexpmatch2
      # invokebuiltin
      # opt_invokebuiltin_delegate
      # opt_invokebuiltin_delegate_leave
      when :getlocal_WC_0 then getlocal_WC_0(jit, ctx, asm)
      # setlocal_WC_0
      # setlocal_WC_1
      when :putobject_INT2FIX_0_ then putobject_INT2FIX_0_(jit, ctx, asm)
      when :putobject_INT2FIX_1_ then putobject_INT2FIX_1_(jit, ctx, asm)
      else CantCompile
      end
    end

    private

    #
    # Insns
    #

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def nop(jit, ctx, asm)
      # Do nothing
      KeepCompiling
    end

    # getlocal
    # setlocal
    # getblockparam
    # setblockparam
    # getblockparamproxy
    # getspecial
    # setspecial

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def getinstancevariable(jit, ctx, asm)
      # Specialize on compile-time receiver, and split a block for chain guards
      unless jit.at_current_insn?
        defer_compilation(jit, ctx, asm)
        return EndBlock
      end

      id = jit.operand(0)
      comptime_obj = jit.peek_at_self

      jit_getivar(jit, ctx, asm, comptime_obj, id)
    end

    # setinstancevariable
    # getclassvariable
    # setclassvariable
    # opt_getconstant_path
    # getconstant
    # setconstant
    # getglobal
    # setglobal

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def putnil(jit, ctx, asm)
      assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
      asm.mov([SP, C.VALUE.size * ctx.stack_size], Qnil)
      ctx.stack_push(1)
      KeepCompiling
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def putself(jit, ctx, asm)
      assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
      asm.mov(:rax, [CFP, C.rb_control_frame_t.offsetof(:self)])
      asm.mov([SP, C.VALUE.size * ctx.stack_size], :rax)
      ctx.stack_push(1)
      KeepCompiling
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def putobject(jit, ctx, asm, val: jit.operand(0))
      # Push it to the stack
      # TODO: GC offsets
      assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
      if asm.imm32?(val)
        asm.mov([SP, C.VALUE.size * ctx.stack_size], val)
      else # 64-bit immediates can't be directly written to memory
        asm.mov(:rax, val)
        asm.mov([SP, C.VALUE.size * ctx.stack_size], :rax)
      end

      ctx.stack_push(1)
      KeepCompiling
    end

    # putspecialobject
    # putstring
    # concatstrings
    # anytostring
    # toregexp
    # intern
    # newarray
    # newarraykwsplat
    # duparray
    # duphash
    # expandarray
    # concatarray
    # splatarray
    # newhash
    # newrange
    # pop
    # dup
    # dupn
    # swap
    # opt_reverse
    # topn
    # setn
    # adjuststack
    # defined
    # checkmatch
    # checkkeyword
    # checktype
    # defineclass
    # definemethod
    # definesmethod
    # send

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    # @param cd `RubyVM::MJIT::CPointer::Struct_rb_call_data`
    def opt_send_without_block(jit, ctx, asm)
      cd = C.rb_call_data.new(jit.operand(0))
      jit_call_method(jit, ctx, asm, cd)
    end

    # objtostring
    # opt_str_freeze
    # opt_nil_p
    # opt_str_uminus
    # opt_newarray_max
    # opt_newarray_min
    # invokesuper
    # invokeblock

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def leave(jit, ctx, asm)
      assert_equal(ctx.stack_size, 1)

      jit_check_ints(jit, ctx, asm)

      asm.comment('pop stack frame')
      asm.lea(:rax, [CFP, C.rb_control_frame_t.size])
      asm.mov(CFP, :rax)
      asm.mov([EC, C.rb_execution_context_t.offsetof(:cfp)], :rax)

      # Return a value (for compile_leave_exit)
      ret_opnd = ctx.stack_pop
      asm.mov(:rax, ret_opnd)

      # Set caller's SP and push a value to its stack (for JIT)
      asm.mov(SP, [CFP, C.rb_control_frame_t.offsetof(:sp)]) # Note: SP is in the position after popping a receiver and arguments
      asm.mov([SP], :rax)

      # Jump to cfp->jit_return
      asm.jmp([CFP, -C.rb_control_frame_t.size + C.rb_control_frame_t.offsetof(:jit_return)])

      EndBlock
    end

    # throw
    # jump
    # branchif

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def branchunless(jit, ctx, asm)
      # TODO: check ints for backward branches
      # TODO: skip check for known truthy

      # This `test` sets ZF only for Qnil and Qfalse, which let jz jump.
      asm.test([SP, C.VALUE.size * (ctx.stack_size - 1)], ~Qnil)
      ctx.stack_pop(1)

      # Set stubs
      branch_stub = BranchStub.new(
        iseq: jit.iseq,
        shape: Default,
        target0: BranchTarget.new(ctx:, pc: jit.pc + C.VALUE.size * (jit.insn.len + jit.operand(0))), # branch target
        target1: BranchTarget.new(ctx:, pc: jit.pc + C.VALUE.size * jit.insn.len),                    # fallthrough
      )
      branch_stub.target0.address = Assembler.new.then do |ocb_asm|
        @exit_compiler.compile_branch_stub(ctx, ocb_asm, branch_stub, true)
        @ocb.write(ocb_asm)
      end
      branch_stub.target1.address = Assembler.new.then do |ocb_asm|
        @exit_compiler.compile_branch_stub(ctx, ocb_asm, branch_stub, false)
        @ocb.write(ocb_asm)
      end

      # Jump to target0 on jz
      branch_stub.compile = proc do |branch_asm|
        branch_asm.comment("branchunless #{branch_stub.shape}")
        branch_asm.stub(branch_stub) do
          case branch_stub.shape
          in Default
            branch_asm.jz(branch_stub.target0.address)
            branch_asm.jmp(branch_stub.target1.address)
          in Next0
            branch_asm.jnz(branch_stub.target1.address)
          in Next1
            branch_asm.jz(branch_stub.target0.address)
          end
        end
      end
      branch_stub.compile.call(asm)

      EndBlock
    end

    # branchnil
    # once
    # opt_case_dispatch

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def opt_plus(jit, ctx, asm)
      unless jit.at_current_insn?
        defer_compilation(jit, ctx, asm)
        return EndBlock
      end

      comptime_recv = jit.peek_at_stack(1)
      comptime_obj  = jit.peek_at_stack(0)

      if fixnum?(comptime_recv) && fixnum?(comptime_obj)
        # Generate a side exit before popping operands
        side_exit = side_exit(jit, ctx)

        unless @invariants.assume_bop_not_redefined(jit, C.INTEGER_REDEFINED_OP_FLAG, C.BOP_PLUS)
          return CantCompile
        end

        obj_opnd  = ctx.stack_pop
        recv_opnd = ctx.stack_pop

        asm.comment('guard recv is fixnum') # TODO: skip this with type information
        asm.test(recv_opnd, C.RUBY_FIXNUM_FLAG)
        asm.jz(side_exit)

        asm.comment('guard obj is fixnum') # TODO: skip this with type information
        asm.test(obj_opnd, C.RUBY_FIXNUM_FLAG)
        asm.jz(side_exit)

        asm.mov(:rax, recv_opnd)
        asm.sub(:rax, 1) # untag
        asm.mov(:rcx, obj_opnd)
        asm.add(:rax, :rcx)
        asm.jo(side_exit)

        dst_opnd = ctx.stack_push
        asm.mov(dst_opnd, :rax)

        KeepCompiling
      else
        CantCompile # TODO: delegate to send
      end
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def opt_minus(jit, ctx, asm)
      unless jit.at_current_insn?
        defer_compilation(jit, ctx, asm)
        return EndBlock
      end

      assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
      comptime_recv = jit.peek_at_stack(1)
      comptime_obj  = jit.peek_at_stack(0)

      if fixnum?(comptime_recv) && fixnum?(comptime_obj)
        unless @invariants.assume_bop_not_redefined(jit, C.INTEGER_REDEFINED_OP_FLAG, C.BOP_MINUS)
          return CantCompile
        end

        assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
        recv_index = ctx.stack_size - 2
        obj_index  = ctx.stack_size - 1

        asm.comment('guard recv is fixnum') # TODO: skip this with type information
        asm.test([SP, C.VALUE.size * recv_index], C.RUBY_FIXNUM_FLAG)
        asm.jz(side_exit(jit, ctx))

        asm.comment('guard obj is fixnum') # TODO: skip this with type information
        asm.test([SP, C.VALUE.size * obj_index], C.RUBY_FIXNUM_FLAG)
        asm.jz(side_exit(jit, ctx))

        asm.mov(:rax, [SP, C.VALUE.size * recv_index])
        asm.mov(:rcx, [SP, C.VALUE.size * obj_index])
        asm.sub(:rax, :rcx)
        asm.jo(side_exit(jit, ctx))
        asm.add(:rax, 1) # re-tag
        asm.mov([SP, C.VALUE.size * recv_index], :rax)

        ctx.stack_pop(1)
        KeepCompiling
      else
        CantCompile # TODO: delegate to send
      end
    end

    # opt_mult
    # opt_div
    # opt_mod
    # opt_eq
    # opt_neq

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def opt_lt(jit, ctx, asm)
      unless jit.at_current_insn?
        defer_compilation(jit, ctx, asm)
        return EndBlock
      end

      assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
      comptime_recv = jit.peek_at_stack(1)
      comptime_obj  = jit.peek_at_stack(0)

      if fixnum?(comptime_recv) && fixnum?(comptime_obj)
        unless @invariants.assume_bop_not_redefined(jit, C.INTEGER_REDEFINED_OP_FLAG, C.BOP_LT)
          return CantCompile
        end

        assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
        recv_index = ctx.stack_size - 2
        obj_index  = ctx.stack_size - 1

        asm.comment('guard recv is fixnum') # TODO: skip this with type information
        asm.test([SP, C.VALUE.size * recv_index], C.RUBY_FIXNUM_FLAG)
        asm.jz(side_exit(jit, ctx))

        asm.comment('guard obj is fixnum') # TODO: skip this with type information
        asm.test([SP, C.VALUE.size * obj_index], C.RUBY_FIXNUM_FLAG)
        asm.jz(side_exit(jit, ctx))

        asm.mov(:rax, [SP, C.VALUE.size * obj_index])
        asm.cmp([SP, C.VALUE.size * recv_index], :rax)
        asm.mov(:rax, Qfalse)
        asm.mov(:rcx, Qtrue)
        asm.cmovl(:rax, :rcx)
        asm.mov([SP, C.VALUE.size * recv_index], :rax)

        ctx.stack_pop(1)
        KeepCompiling
      else
        CantCompile # TODO: delegate to send
      end
    end

    # opt_le
    # opt_gt
    # opt_ge
    # opt_ltlt
    # opt_and
    # opt_or
    # opt_aref
    # opt_aset
    # opt_aset_with
    # opt_aref_with
    # opt_length
    # opt_size
    # opt_empty_p
    # opt_succ
    # opt_not
    # opt_regexpmatch2
    # invokebuiltin
    # opt_invokebuiltin_delegate
    # opt_invokebuiltin_delegate_leave

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def getlocal_WC_0(jit, ctx, asm)
      # Get operands
      idx = jit.operand(0)
      level = 0

      # Get EP
      asm.mov(:rax, [CFP, C.rb_control_frame_t.offsetof(:ep)])

      # Get a local variable
      asm.mov(:rax, [:rax, -idx * C.VALUE.size])

      # Push it to the stack
      asm.mov([SP, C.VALUE.size * ctx.stack_size], :rax)
      ctx.stack_push(1)
      KeepCompiling
    end

    # getlocal_WC_1
    # setlocal_WC_0
    # setlocal_WC_1

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def putobject_INT2FIX_0_(jit, ctx, asm)
      putobject(jit, ctx, asm, val: C.to_value(0))
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def putobject_INT2FIX_1_(jit, ctx, asm)
      putobject(jit, ctx, asm, val: C.to_value(1))
    end

    #
    # Helpers
    #

    # @param asm [RubyVM::MJIT::Assembler]
    def guard_object_is_heap(asm, object_opnd, side_exit)
      asm.comment('guard object is heap')
      # Test that the object is not an immediate
      asm.test(object_opnd, C.RUBY_IMMEDIATE_MASK)
      asm.jnz(side_exit)

      # Test that the object is not false
      asm.cmp(object_opnd, Qfalse)
      asm.je(side_exit)
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def jit_chain_guard(opcode, jit, ctx, asm, side_exit, limit: 10)
      assert_equal(opcode, :jne) # TODO: support more

      if ctx.chain_depth < limit
        deeper = ctx.dup
        deeper.chain_depth += 1

        branch_stub = BranchStub.new(
          iseq: jit.iseq,
          shape: Default,
          target0: BranchTarget.new(ctx: deeper, pc: jit.pc),
        )
        branch_stub.target0.address = Assembler.new.then do |ocb_asm|
          @exit_compiler.compile_branch_stub(deeper, ocb_asm, branch_stub, true)
          @ocb.write(ocb_asm)
        end
        branch_stub.compile = proc do |branch_asm|
          branch_asm.comment('jit_chain_guard')
          branch_asm.stub(branch_stub) do
            case branch_stub.shape
            in Default
              asm.jne(branch_stub.target0.address)
            end
          end
        end
        branch_stub.compile.call(asm)
      else
        asm.jne(side_exit)
      end
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def jump_to_next_insn(jit, ctx, asm)
      reset_depth = ctx.dup
      reset_depth.chain_depth = 0

      next_pc = jit.pc + jit.insn.len * C.VALUE.size
      stub_next_block(jit.iseq, next_pc, reset_depth, asm, comment: 'jump_to_next_insn')
    end

    # rb_vm_check_ints
    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def jit_check_ints(jit, ctx, asm)
      asm.comment('RUBY_VM_CHECK_INTS(ec)')
      asm.mov(:eax, [EC, C.rb_execution_context_t.offsetof(:interrupt_flag)])
      asm.test(:eax, :eax)
      asm.jnz(side_exit(jit, ctx))
    end

    # vm_getivar
    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def jit_getivar(jit, ctx, asm, comptime_obj, ivar_id)
      side_exit = side_exit(jit, ctx)
      starting_ctx = ctx.dup # copy for jit_chain_guard

      # Guard not special const
      if C.SPECIAL_CONST_P(comptime_obj)
        asm.incr_counter(:getivar_special_const)
        return CantCompile
      end
      asm.mov(:rax, [CFP, C.rb_control_frame_t.offsetof(:self)])
      guard_object_is_heap(asm, :rax, counted_exit(side_exit, :getivar_not_heap))

      case C.BUILTIN_TYPE(comptime_obj)
      when C.T_OBJECT
        # This is the only supported case for now (ROBJECT_IVPTR)
      else
        asm.incr_counter(:getivar_not_t_object)
        return CantCompile
      end

      shape_id = C.rb_shape_get_shape_id(comptime_obj)
      if shape_id == C.OBJ_TOO_COMPLEX_SHAPE_ID
        asm.incr_counter(:getivar_too_complex)
        return CantCompile
      end

      asm.comment('guard shape')
      asm.cmp(DwordPtr[:rax, C.rb_shape_id_offset], shape_id)
      jit_chain_guard(:jne, jit, starting_ctx, asm, counted_exit(side_exit, :getivar_megamorphic))

      index = C.rb_shape_get_iv_index(shape_id, ivar_id)
      if index
        # See ROBJECT_IVPTR
        if C.FL_TEST_RAW(comptime_obj, C.ROBJECT_EMBED)
          # Access embedded array
          asm.mov(:rax, [:rax, C.RObject.offsetof(:as, :ary) + (index * C.VALUE.size)])
        else
          # Pull out an ivar table on heap
          asm.mov(:rax, [:rax, C.RObject.offsetof(:as, :heap, :ivptr)])
          # Read the table
          asm.mov(:rax, [:rax, index * C.VALUE.size])
        end
        val_opnd = :rax
      else
        val_opnd = Qnil
      end

      stack_opnd = ctx.stack_push
      asm.mov(stack_opnd, val_opnd)

      # Let guard chains share the same successor
      jump_to_next_insn(jit, ctx, asm)
      EndBlock
    end

    # vm_call_method (vm_sendish -> vm_call_general -> vm_call_method)
    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    # @param cd `RubyVM::MJIT::CPointer::Struct_rb_call_data`
    def jit_call_method(jit, ctx, asm, cd)
      ci = cd.ci
      argc = C.vm_ci_argc(ci)
      mid = C.vm_ci_mid(ci)
      flags = C.vm_ci_flag(ci)

      unless jit.at_current_insn?
        defer_compilation(jit, ctx, asm)
        return EndBlock
      end

      if flags & C.VM_CALL_KW_SPLAT != 0
        # recv_index calculation may not work for this
        asm.incr_counter(:send_kw_splat)
        return CantCompile
      end
      assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
      recv_depth = argc + ((flags & C.VM_CALL_ARGS_BLOCKARG == 0) ? 0 : 1)
      recv_index = ctx.stack_size - 1 - recv_depth

      comptime_recv = jit.peek_at_stack(recv_depth)
      comptime_recv_klass = C.rb_class_of(comptime_recv)

      # Guard the receiver class (part of vm_search_method_fastpath)
      if comptime_recv_klass.singleton_class?
        asm.comment('guard known object with singleton class')
        asm.mov(:rax, C.to_value(comptime_recv))
        asm.cmp([SP, C.VALUE.size * recv_index], :rax)
        asm.jne(side_exit(jit, ctx))
      else
        # TODO: support more classes
        asm.incr_counter(:send_guard_known_object)
        return CantCompile
      end

      # Do method lookup (vm_cc_cme(cc) != NULL)
      cme = C.rb_callable_method_entry(comptime_recv_klass, mid)
      if cme.nil?
        asm.incr_counter(:send_missing_cme)
        return CantCompile # We don't support vm_call_method_name
      end

      # The main check of vm_call_method before vm_call_method_each_type
      case C.METHOD_ENTRY_VISI(cme)
      when C.METHOD_VISI_PUBLIC
        # You can always call public methods
      when C.METHOD_VISI_PRIVATE
        # Allow only callsites without a receiver
        if flags & C.VM_CALL_FCALL == 0
          asm.incr_counter(:send_private)
          return CantCompile
        end
      when C.METHOD_VISI_PROTECTED
        asm.incr_counter(:send_protected)
        return CantCompile # TODO: support this
      else
        # TODO: Change them to a constant and use case-in instead
        raise 'unreachable'
      end

      # Invalidate on redefinition (part of vm_search_method_fastpath)
      @invariants.assume_method_lookup_stable(jit, cme)

      jit_call_method_each_type(jit, ctx, asm, ci, argc, flags, cme)
    end

    # vm_call_method_each_type
    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def jit_call_method_each_type(jit, ctx, asm, ci, argc, flags, cme)
      case cme.def.type
      when C.VM_METHOD_TYPE_ISEQ
        jit_call_iseq_setup(jit, ctx, asm, ci, cme, flags, argc)
      else
        asm.incr_counter(:send_not_iseq)
        return CantCompile
      end
    end

    # vm_call_iseq_setup
    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def jit_call_iseq_setup(jit, ctx, asm, ci, cme, flags, argc)
      iseq = def_iseq_ptr(cme.def)
      opt_pc = jit_callee_setup_arg(jit, ctx, asm, ci, flags, iseq)
      if opt_pc == CantCompile
        # We hit some unsupported path of vm_callee_setup_arg
        return CantCompile
      end

      if flags & C.VM_CALL_TAILCALL != 0
        # We don't support vm_call_iseq_setup_tailcall
        asm.incr_counter(:send_tailcall)
        return CantCompile
      end
      jit_call_iseq_setup_normal(jit, ctx, asm, ci, cme, flags, argc, iseq)
    end

    # vm_call_iseq_setup_normal (vm_call_iseq_setup_2 -> vm_call_iseq_setup_normal)
    def jit_call_iseq_setup_normal(jit, ctx, asm, ci, cme, flags, argc, iseq)
      # Save caller SP and PC before pushing a callee frame for backtrace and side exits
      asm.comment('save SP to caller CFP')
      assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
      sp_index = ctx.stack_size - 1 - argc - ((flags & C.VM_CALL_ARGS_BLOCKARG == 0) ? 0 : 1) # Pop receiver and arguments for side exits
      asm.lea(:rax, [SP, C.VALUE.size * sp_index])
      asm.mov([CFP, C.rb_control_frame_t.offsetof(:sp)], :rax)

      asm.comment('save PC to caller CFP')
      next_pc = jit.pc + jit.insn.len * C.VALUE.size # Use the next one for backtrace and side exits
      asm.mov(:rax, next_pc)
      asm.mov([CFP, C.rb_control_frame_t.offsetof(:pc)], :rax)

      frame_type = C.VM_FRAME_MAGIC_METHOD | C.VM_ENV_FLAG_LOCAL
      jit_push_frame(jit, ctx, asm, ci, cme, flags, argc, iseq, frame_type, next_pc)
    end

    # vm_push_frame
    #
    # Frame structure:
    # | args | locals | cme/cref | block_handler/prev EP | frame type (EP here) | stack bottom (SP here)
    def jit_push_frame(jit, ctx, asm, ci, cme, flags, argc, iseq, frame_type, next_pc)
      # TODO: stack overflow check

      local_size = iseq.body.local_table_size - iseq.body.param.size
      local_size.times do |i|
        asm.comment('set local variables') if i == 0
        assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
        local_index = ctx.stack_size + i
        asm.mov([SP, C.VALUE.size * local_index], Qnil)
      end

      assert_equal(ctx.sp_offset, ctx.stack_size) # TODO: support SP motion
      sp_offset = ctx.stack_size + local_size + 3
      asm.add(SP, C.VALUE.size * sp_offset)

      asm.comment('set cme')
      asm.mov(:rax, cme.to_i)
      asm.mov([SP, C.VALUE.size * -3], :rax)

      asm.comment('set specval')
      asm.mov([SP, C.VALUE.size * -2], C.VM_BLOCK_HANDLER_NONE)

      asm.comment('set frame type')
      asm.mov([SP, C.VALUE.size * -1], frame_type)

      asm.comment('move CFP register to callee CFP')
      asm.sub(CFP, C.rb_control_frame_t.size);

      # Not setting PC since JIT code will do that as needed
      asm.comment('set SP to callee CFP')
      asm.mov([CFP, C.rb_control_frame_t.offsetof(:sp)], SP)
      asm.comment('set ISEQ to callee CFP')
      asm.mov(:rax, iseq.to_i)
      asm.mov([CFP, C.rb_control_frame_t.offsetof(:iseq)], :rax)
      asm.comment('set self to callee CFP')
      self_index = -(1 + argc + ((flags & C.VM_CALL_ARGS_BLOCKARG == 0) ? 0 : 1) + local_size + 3)
      asm.mov(:rax, [SP, C.VALUE.size * self_index])
      asm.mov([CFP, C.rb_control_frame_t.offsetof(:self)], :rax)
      asm.comment('set EP to callee CFP')
      asm.lea(:rax, [SP, C.VALUE.size * -1])
      asm.mov([CFP, C.rb_control_frame_t.offsetof(:ep)], :rax)
      asm.comment('set block_code to callee CFP')
      asm.mov([CFP, C.rb_control_frame_t.offsetof(:block_code)], 0)
      asm.comment('set BP to callee CFP')
      asm.mov([CFP, C.rb_control_frame_t.offsetof(:__bp__)], SP) # TODO: get rid of this!!

      # Stub cfp->jit_return
      return_ctx = ctx.dup
      return_ctx.stack_size -= argc + ((flags & C.VM_CALL_ARGS_BLOCKARG == 0) ? 0 : 1) # Pop args
      return_ctx.sp_offset = 1 # SP is in the position after popping a receiver and arguments
      branch_stub = BranchStub.new(
        iseq: jit.iseq,
        shape: Default,
        target0: BranchTarget.new(ctx: return_ctx, pc: next_pc),
      )
      branch_stub.target0.address = Assembler.new.then do |ocb_asm|
        @exit_compiler.compile_branch_stub(return_ctx, ocb_asm, branch_stub, true)
        @ocb.write(ocb_asm)
      end
      branch_stub.compile = proc do |branch_asm|
        branch_asm.comment('set jit_return to callee CFP')
        branch_asm.stub(branch_stub) do
          case branch_stub.shape
          in Default
            branch_asm.mov(:rax, branch_stub.target0.address)
            branch_asm.mov([CFP, C.rb_control_frame_t.offsetof(:jit_return)], :rax)
          end
        end
      end
      branch_stub.compile.call(asm)

      asm.comment('set callee CFP to ec->cfp')
      asm.mov([EC, C.rb_execution_context_t.offsetof(:cfp)], CFP)

      # Jump to a stub for the callee ISEQ
      callee_ctx = Context.new
      stub_next_block(iseq, iseq.body.iseq_encoded.to_i, callee_ctx, asm)

      EndBlock
    end

    # vm_callee_setup_arg: Set up args and return opt_pc (or CantCompile)
    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def jit_callee_setup_arg(jit, ctx, asm, ci, flags, iseq)
      if flags & C.VM_CALL_KW_SPLAT == 0
        if C.rb_simple_iseq_p(iseq)
          if jit_caller_setup_arg(jit, ctx, asm, flags) == CantCompile
            return CantCompile
          end
          if jit_caller_remove_empty_kw_splat(jit, ctx, asm, flags) == CantCompile
            return CantCompile
          end

          if C.vm_ci_argc(ci) != iseq.body.param.lead_num
            # argument_arity_error
            return CantCompile
          end

          return 0
        else
          # We don't support the remaining `else if`s yet.
          return CantCompile
        end
      end

      # We don't support setup_parameters_complex
      return CantCompile
    end

    # CALLER_SETUP_ARG: Return CantCompile if not supported
    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def jit_caller_setup_arg(jit, ctx, asm, flags)
      if flags & C.VM_CALL_ARGS_SPLAT != 0
        # We don't support vm_caller_setup_arg_splat
        asm.incr_counter(:send_args_splat)
        return CantCompile
      end
      if flags & (C.VM_CALL_KWARG | C.VM_CALL_KW_SPLAT) != 0
        # We don't support keyword args either
        asm.incr_counter(:send_kwarg)
        return CantCompile
      end
    end

    # CALLER_REMOVE_EMPTY_KW_SPLAT: Return CantCompile if not supported
    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def jit_caller_remove_empty_kw_splat(jit, ctx, asm, flags)
      if (flags & C.VM_CALL_KW_SPLAT) > 0
        # We don't support removing the last Hash argument
        asm.incr_counter(:send_kw_splat)
        return CantCompile
      end
    end

    def assert_equal(left, right)
      if left != right
        raise "'#{left.inspect}' was not '#{right.inspect}'"
      end
    end

    def fixnum?(obj)
      flag = C.RUBY_FIXNUM_FLAG
      (C.to_value(obj) & flag) == flag
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def defer_compilation(jit, ctx, asm)
      # Make a stub to compile the current insn
      stub_next_block(jit.iseq, jit.pc, ctx, asm, comment: 'defer_compilation')
    end

    def stub_next_block(iseq, pc, ctx, asm, comment: 'stub_next_block')
      branch_stub = BranchStub.new(
        iseq:,
        shape: Default,
        target0: BranchTarget.new(ctx:, pc:),
      )
      branch_stub.target0.address = Assembler.new.then do |ocb_asm|
        @exit_compiler.compile_branch_stub(ctx, ocb_asm, branch_stub, true)
        @ocb.write(ocb_asm)
      end
      branch_stub.compile = proc do |branch_asm|
        branch_asm.comment(comment)
        branch_asm.stub(branch_stub) do
          case branch_stub.shape
          in Default
            branch_asm.jmp(branch_stub.target0.address)
          in Next0
            # Just write the block without a jump
          end
        end
      end
      branch_stub.compile.call(asm)
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    def side_exit(jit, ctx)
      if side_exit = jit.side_exits[jit.pc]
        return side_exit
      end
      asm = Assembler.new
      @exit_compiler.compile_side_exit(jit, ctx, asm)
      jit.side_exits[jit.pc] = @ocb.write(asm)
    end

    def counted_exit(side_exit, name)
      asm = Assembler.new
      asm.incr_counter(name)
      asm.jmp(side_exit)
      @ocb.write(asm)
    end

    def def_iseq_ptr(cme_def)
      C.rb_iseq_check(cme_def.body.iseq.iseqptr)
    end
  end
end
