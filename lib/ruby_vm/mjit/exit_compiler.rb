module RubyVM::MJIT
  class ExitCompiler
    def initialize
      @gc_refs = [] # TODO: GC offsets?
    end

    # Used for invalidating a block on entry.
    # @param pc [Integer]
    # @param asm [RubyVM::MJIT::Assembler]
    def compile_entry_exit(pc, ctx, asm, cause:)
      # Increment per-insn exit counter
      incr_insn_exit(pc, asm)

      # Fix pc/sp offsets for the interpreter
      save_pc_and_sp(pc, ctx, asm, reset_sp_offset: false)

      # Restore callee-saved registers
      asm.comment("#{cause}: entry exit")
      asm.pop(SP)
      asm.pop(EC)
      asm.pop(CFP)

      asm.mov(C_RET, Qundef)
      asm.ret
    end

    # Set to cfp->jit_return by default for leave insn
    # @param asm [RubyVM::MJIT::Assembler]
    def compile_leave_exit(asm)
      asm.comment('default cfp->jit_return')

      # Restore callee-saved registers
      asm.pop(SP)
      asm.pop(EC)
      asm.pop(CFP)

      # :rax is written by #leave
      asm.ret
    end

    # Fire cfunc events on invalidation by TracePoint
    # @param asm [RubyVM::MJIT::Assembler]
    def compile_full_cfunc_return(asm)
      # This chunk of code expects REG_EC to be filled properly and
      # RAX to contain the return value of the C method.

      asm.comment('full cfunc return')
      asm.mov(C_ARGS[0], EC)
      asm.mov(C_ARGS[1], :rax)
      asm.call(C.rb_full_cfunc_return)

      # TODO: count the exit

      # Restore callee-saved registers
      asm.pop(SP)
      asm.pop(EC)
      asm.pop(CFP)

      asm.mov(C_RET, Qundef)
      asm.ret
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def compile_side_exit(jit, ctx, asm)
      # Increment per-insn exit counter
      incr_insn_exit(jit.pc, asm)

      # Fix pc/sp offsets for the interpreter
      save_pc_and_sp(jit.pc, ctx.dup, asm) # dup to avoid sp_offset update

      # Restore callee-saved registers
      asm.comment("exit to interpreter on #{pc_to_insn(jit.pc).name}")
      asm.pop(SP)
      asm.pop(EC)
      asm.pop(CFP)

      asm.mov(C_RET, Qundef)
      asm.ret
    end

    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    # @param branch_stub [RubyVM::MJIT::BranchStub]
    # @param target0_p [TrueClass,FalseClass]
    def compile_branch_stub(ctx, asm, branch_stub, target0_p)
      # Call rb_mjit_branch_stub_hit
      asm.comment("branch stub hit (#{branch_stub.object_id}): #{branch_stub.iseq.body.location.label}@#{C.rb_iseq_path(branch_stub.iseq)}:#{iseq_lineno(branch_stub.iseq, target0_p ? branch_stub.target0.pc : branch_stub.target1.pc)}")
      asm.mov(:rdi, to_value(branch_stub))
      asm.mov(:esi, ctx.sp_offset)
      asm.mov(:edx, target0_p ? 1 : 0)
      asm.mov(:rcx, branch_stub.object_id)
      asm.call(C.rb_mjit_branch_stub_hit)

      # Jump to the address returned by rb_mjit_stub_hit
      asm.jmp(:rax)
    end

    private

    def pc_to_insn(pc)
      Compiler.decode_insn(C.VALUE.new(pc).*)
    end

    # @param pc [Integer]
    # @param asm [RubyVM::MJIT::Assembler]
    def incr_insn_exit(pc, asm)
      if C.mjit_opts.stats
        insn = Compiler.decode_insn(C.VALUE.new(pc).*)
        asm.comment("increment insn exit: #{insn.name}")
        asm.mov(:rax, (C.mjit_insn_exits + insn.bin).to_i)
        asm.add([:rax], 1) # TODO: lock
      end
    end

    # @param jit [RubyVM::MJIT::JITState]
    # @param ctx [RubyVM::MJIT::Context]
    # @param asm [RubyVM::MJIT::Assembler]
    def save_pc_and_sp(pc, ctx, asm, reset_sp_offset: true)
      # Update pc (TODO: manage PC offset?)
      asm.comment("save PC#{' and SP' if ctx.sp_offset != 0} to CFP")
      asm.mov(:rax, pc) # rax = jit.pc
      asm.mov([CFP, C.rb_control_frame_t.offsetof(:pc)], :rax) # cfp->pc = rax

      # Update sp
      if ctx.sp_offset != 0
        asm.add(SP, C.VALUE.size * ctx.sp_offset) # sp += stack_size
        asm.mov([CFP, C.rb_control_frame_t.offsetof(:sp)], SP) # cfp->sp = sp
        if reset_sp_offset
          ctx.sp_offset = 0
        end
      end
    end

    def to_value(obj)
      @gc_refs << obj
      C.to_value(obj)
    end

    def iseq_lineno(iseq, pc)
      C.rb_iseq_line_no(iseq, (pc - iseq.body.iseq_encoded.to_i) / C.VALUE.size)
    rescue RangeError # bignum too big to convert into `unsigned long long' (RangeError)
      -1
    end
  end
end
