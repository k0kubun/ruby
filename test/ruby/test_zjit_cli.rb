# frozen_string_literal: true
#
# This set of tests can be run with:
#
#     make test/ruby/test_zjit_cli.rb
#
# Test ZJIT through the command-line interface. This relatively heavy-weight
# way to test is only necessary when exercising command-line options, stats,
# environment variables, or process-level behavior. Tests that only exercise
# codegen should be added to the Rust test harness (`codegen_tests.rs`)
# instead: it parallelizes better and allows for easy inspection of VM internal
# states.

require 'test/unit'
require 'envutil'
require_relative '../lib/jit_support'
return unless JITSupport.zjit_supported?

class TestZJITCLI < Test::Unit::TestCase
  def test_enabled
    assert_runs 'false', <<~RUBY, zjit: false
      RubyVM::ZJIT.enabled?
    RUBY
    assert_runs 'true', <<~RUBY, zjit: true
      RubyVM::ZJIT.enabled?
    RUBY
  end

  def test_stats_enabled
    assert_runs 'false', <<~RUBY, stats: false
      RubyVM::ZJIT.stats_enabled?
    RUBY
    assert_runs 'true', <<~RUBY, stats: true
      RubyVM::ZJIT.stats_enabled?
    RUBY
  end

  def test_stats_string_no_zjit
    assert_runs 'nil', <<~RUBY, zjit: false
      RubyVM::ZJIT.stats_string
    RUBY
    assert_runs 'true', <<~RUBY, stats: false
      RubyVM::ZJIT.stats_string.is_a?(String)
    RUBY
    assert_runs 'true', <<~RUBY, stats: true
      RubyVM::ZJIT.stats_string.is_a?(String)
    RUBY
  end

  def test_stats_quiet
    # Test that --zjit-stats-quiet collects stats but doesn't print them
    script = <<~RUBY
      def test = 42
      test
      test
      puts RubyVM::ZJIT.stats_enabled?
    RUBY

    stats_header = "***ZJIT: Printing ZJIT statistics on exit***"

    # With --zjit-stats, stats should be printed to stderr
    out, err, status = eval_with_jit(script, stats: true)
    assert_success(out, err, status)
    assert_include(err, stats_header)
    assert_equal("true\n", out)

    # With --zjit-stats-quiet, stats should NOT be printed but still enabled
    out, err, status = eval_with_jit(script, stats: :quiet)
    assert_success(out, err, status)
    refute_includes(err, stats_header)
    assert_equal("true\n", out)

    # With --zjit-stats=<path>, stats should be printed to the path
    Tempfile.create("zjit-stats-") {|tmp|
      stats_file = tmp.path
      tmp.puts("Lorem ipsum dolor sit amet, consectetur adipiscing elit, ...")
      tmp.close

      out, err, status = eval_with_jit(script, stats: stats_file)
      assert_success(out, err, status)
      refute_includes(err, stats_header)
      assert_equal("true\n", out)
      assert_equal stats_header, File.open(stats_file) {|f| f.gets(chomp: true)}, "should be overwritten"
    }

    # With --zjit-stats=<path> ending in .json, stats should be dumped as JSON
    Tempfile.create(["zjit-stats-", ".json"]) {|tmp|
      stats_file = tmp.path
      tmp.puts("Lorem ipsum dolor sit amet, consectetur adipiscing elit, ...")
      tmp.close

      out, err, status = eval_with_jit(script, stats: stats_file)
      assert_success(out, err, status)
      refute_includes(err, stats_header)
      assert_equal("true\n", out)

      require "json"
      json = JSON.parse(File.read(stats_file))
      assert_kind_of Hash, json, "should be JSON"
      assert json.key?("compiled_iseq_count"), "should contain stats keys"
      refute_includes File.read(stats_file), stats_header, "should not contain the text stats header"
    }
  end

  def test_enable_through_env
    child_env = {'RUBY_YJIT_ENABLE' => nil, 'RUBY_ZJIT_ENABLE' => '1'}
    assert_in_out_err([child_env, '-v'], '') do |stdout, stderr|
      assert_include(stdout.first, '+ZJIT')
      assert_equal([], stderr)
    end
  end

  def test_zjit_enable
    # --disable-all is important in case the build/environment has YJIT enabled by
    # default through e.g. -DYJIT_FORCE_ENABLE. Can't enable ZJIT when YJIT is on.
    assert_separately(["--disable-all"], <<~'RUBY')
      refute_predicate RubyVM::ZJIT, :enabled?
      refute_predicate RubyVM::ZJIT, :stats_enabled?
      refute_includes RUBY_DESCRIPTION, "+ZJIT"

      RubyVM::ZJIT.enable

      assert_predicate RubyVM::ZJIT, :enabled?
      refute_predicate RubyVM::ZJIT, :stats_enabled?
      assert_include RUBY_DESCRIPTION, "+ZJIT"
    RUBY
  end

  def test_zjit_disable
    assert_separately(["--zjit", "--zjit-disable"], <<~'RUBY')
      refute_predicate RubyVM::ZJIT, :enabled?
      refute_includes RUBY_DESCRIPTION, "+ZJIT"

      RubyVM::ZJIT.enable

      assert_predicate RubyVM::ZJIT, :enabled?
      assert_include RUBY_DESCRIPTION, "+ZJIT"
    RUBY
  end

  def test_zjit_prelude_kernel_prepend
    # Simulate what bundler/setup can do: prepend a module to Kernel during
    # the prelude via the BUNDLER_SETUP mechanism in rubygems.rb:
    #   require ENV["BUNDLER_SETUP"] if ENV["BUNDLER_SETUP"] && !defined?(Bundler)
    Tempfile.create(["kernel_prepend", ".rb"]) do |f|
      f.write("Kernel.prepend(Module.new)\n")
      f.flush
      assert_separately([{ "BUNDLER_SETUP" => f.path }, "--enable=gems", "--zjit"], "", ignore_stderr: true)
    end
  end

  def test_zjit_enable_respects_existing_options
    assert_separately(['--zjit-disable', '--zjit-stats-quiet'], <<~RUBY)
      refute_predicate RubyVM::ZJIT, :enabled?
      assert_predicate RubyVM::ZJIT, :stats_enabled?

      RubyVM::ZJIT.enable

      assert_predicate RubyVM::ZJIT, :enabled?
      assert_predicate RubyVM::ZJIT, :stats_enabled?
    RUBY
  end

  def test_toplevel_binding
    # Not using assert_compiles, which doesn't use the toplevel frame for `test_script`.
    out, err, status = eval_with_jit(%q{
      a = 1
      b = 2
      TOPLEVEL_BINDING.local_variable_set(:b, 3)
      c = 4
      print [a, b, c]
    })
    assert_success(out, err, status)
    assert_equal "[1, 3, 4]", out
  end

  def test_send_exit_with_uninitialized_locals
    assert_runs 'nil', %q{
      def entry(init)
        function_stub_exit(init)
      end

      def function_stub_exit(init)
        uninitialized_local = 1 if init
        uninitialized_local
      end

      entry(true) # profile and set 1 to the local slot
      entry(false)
    }, call_threshold: 2, allowed_iseqs: 'entry@-e:2'
  end

  def test_opt_new_with_custom_allocator
    assert_compiles '"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"', %q{
      require "digest"
      def test = Digest::SHA256.new.hexdigest
      test; test
    }, insns: [:opt_new], call_threshold: 2
  end

  def test_opt_new_with_custom_allocator_raises
    assert_compiles '[42, 42]', %q{
      require "digest"
      class C < Digest::Base; end
      def test
        begin
          Digest::Base.new
        rescue NotImplementedError
          42
        end
      end
      [test, test]
    }, insns: [:opt_new], call_threshold: 2
  end

  def test_getconstant_path_autoload
    # A constant-referencing expression can run arbitrary code through Kernel#autoload.
    Dir.mktmpdir('autoload') do |tmpdir|
      autoload_path = File.join(tmpdir, 'test_getconstant_path_autoload.rb')
      File.write(autoload_path, 'X = RUBY_COPYRIGHT')

      assert_compiles RUBY_COPYRIGHT.dump, %Q{
        Object.autoload(:X, #{File.realpath(autoload_path).inspect})
        def test = X
        test
      }, call_threshold: 1, insns: [:opt_getconstant_path]
    end
  end

  # tool/ruby_vm/views/*.erb relies on the zjit instructions a) being contiguous and
  # b) being reliably ordered after all the other instructions.
  def test_instruction_order
    insn_names = RubyVM::INSTRUCTION_NAMES
    zjit, others = insn_names.map.with_index.partition { |name, _| name.start_with?('zjit_') }
    zjit_indexes = zjit.map(&:last)
    other_indexes = others.map(&:last)
    zjit_indexes.product(other_indexes).each do |zjit_index, other_index|
      assert zjit_index > other_index, "'#{insn_names[zjit_index]}' at #{zjit_index} "\
        "must be defined after '#{insn_names[other_index]}' at #{other_index}"
    end
  end

  def test_require_rubygems
    assert_runs 'true', %q{
      require 'rubygems'
    }, call_threshold: 2
  end

  def test_require_rubygems_with_auto_compact
    omit("GC.auto_compact= support is required for this test") unless GC.respond_to?(:auto_compact=)
    assert_runs 'true', %q{
      GC.auto_compact = true
      require 'rubygems'
    }, call_threshold: 2
  end

  def test_stats_availability
    assert_runs '[true, true]', %q{
      def test = 1
      test
      [
        RubyVM::ZJIT.stats[:zjit_insn_count] > 0,
        RubyVM::ZJIT.stats(:zjit_insn_count) > 0,
      ]
    }, stats: true
  end

  def test_stats_consistency
    assert_runs '[]', %q{
      def test = 1
      test # increment some counters

      RubyVM::ZJIT.stats.to_a.filter_map do |key, value|
        # The value may be incremented, but the class should stay the same
        other_value = RubyVM::ZJIT.stats(key)
        if value.class != other_value.class
          [key, value, other_value]
        end
      end
    }, stats: true
  end

  def test_reset_stats
    assert_runs 'true', %q{
      def test = 1
      100.times { test }

      # Get initial stats and verify they're non-zero
      initial_stats = RubyVM::ZJIT.stats

      # Reset the stats
      RubyVM::ZJIT.reset_stats!

      # Get stats after reset
      reset_stats = RubyVM::ZJIT.stats

      [
        # After reset, counters should be zero or at least much smaller
        # (some instructions might execute between reset and reading stats)
        :zjit_insn_count.then { |s| initial_stats[s] > 0 && reset_stats[s] < initial_stats[s] },
        :compiled_iseq_count.then { |s| initial_stats[s] > 0 && reset_stats[s] < initial_stats[s] }
      ].all?
    }, stats: true
  end

  def test_exit_tracing
    # Smoke test: --zjit-trace-exits writes a Fuchsia trace (.fxt) file to /tmp
    assert_compiles('true', <<~RUBY, extra_args: ['--zjit-trace-exits'])
      def test(object) = object.itself

      # induce an exit just for good measure
      array = []
      test(array)
      test(array)
      def array.itself = :not_itself
      test(array)

      fxt_files = Dir.glob("/tmp/perfetto-\#{Process.pid}.fxt")
      result = fxt_files.length == 1 && !File.empty?(fxt_files.first)
      File.unlink(*fxt_files)
      result
    RUBY
  end

  def test_send_forwarded_block_arg_nil_then_non_nil
    # Regression test: when a forwarded &block arg is profiled as nil, the nil
    # block optimization must update the frame state to match the stripped args.
    # Otherwise the saved SP is off by one, causing a stack consistency error
    # when the guard side-exits for a non-nil block.
    assert_runs ':ok', <<~RUBY, call_threshold: 2
      def inner(callable = nil, &block)
        callable || block
      end

      def outer(&block)
        inner(&block)
      end

      100.times { outer }
      result = outer { |x| x }
      result.is_a?(Proc) ? :ok : :fail
    RUBY
  end

  def test_send_forwarded_nil_block_arg_with_polymorphic_receiver
    # Regression test: the nil block optimization strips the block arg from the
    # frame state used to set up the callee frame, but the pre-call guards
    # (receiver GuardType, method-redefinition PatchPoint) must keep using the
    # original frame state that still has the block arg on the stack. Otherwise a
    # guard side-exit re-executes the send with a stack that is missing the block
    # arg slot, corrupting the pushed frame's EP (VM_ENV_FLAGS assertion failure).
    # A polymorphic receiver forces the receiver GuardType to side-exit.
    assert_runs ':ok', <<~RUBY, call_threshold: 2
      class Base
        def self.inner(model, name, &block)
          block ? block.call : model
        end
        def self.outer(model, name, &block)
          inner(model, name, &block)
        end
      end
      class A < Base; end
      class B < Base; end
      class C < Base; end
      class D < Base; end

      1000.times do |i|
        klass = [A, B, C, D][i % 4]
        klass.outer(i, :n)
      end
      :ok
    RUBY
  end

  # Freeing an ISEQ must not leave an `IseqPayload` behind. ZJIT used to create
  # one on the free path for every ISEQ, including the vast majority it had
  # never touched, and never released it: on a Rails app that leak was over half
  # of `zjit_alloc_bytes` and it was invisible to the `mem_*` breakdown, which
  # only walks live ISEQs.
  def test_freed_iseqs_do_not_retain_payloads
    # A high call threshold keeps the churned ISEQs out of the JIT, so nothing
    # pins them and the GC really does free them. Compiled ISEQs stay alive for
    # the process lifetime (their JITFrames mark them), which would make this
    # test vacuous at the default threshold.
    assert_runs 'true', <<~RUBY, stats: false, call_threshold: 1_000_000
      def churn(n)
        n.times do |i|
          eval("def __churn\#{i}(x) = x + 1", TOPLEVEL_BINDING, "churn\#{i}.rb")
          send(:"__churn\#{i}", i)
          Object.send(:remove_method, :"__churn\#{i}")
        end
        3.times { GC.start }
        RubyVM::ZJIT.stats
      end

      small = churn(200)
      large = churn(2000)

      # Each iteration creates and frees several ISEQs. Payload allocations and
      # the unaccounted residual both used to grow with the iteration count;
      # neither may now.
      payload_growth = large[:allocated_iseq_payload_count] - small[:allocated_iseq_payload_count]
      alloc_growth = large[:zjit_alloc_bytes] - small[:zjit_alloc_bytes]
      unaccounted_growth = large[:mem_unaccounted_bytes] - small[:mem_unaccounted_bytes]

      payload_growth < 100 && alloc_growth < 100_000 && unaccounted_growth < 10_000
    RUBY
  end

  #
  # --zjit-background-compile
  #
  # Compilation moves to a dedicated Ruby thread. The risks are all races, so
  # these tests aim at the specific windows the design has to close: an
  # invalidation landing while a compile is in flight, the GC freeing or moving
  # an ISEQ that only the compile queue still references, and losing the compile
  # thread to `fork` or `Thread#kill`.
  #

  # Compile on the background thread and wait for it, so that a run this short
  # deterministically exercises the whole path.
  BG_BLOCK = %w[--zjit-background-compile-block].freeze
  # The real feature: compiles overlap with the requesting thread.
  BG = %w[--zjit-background-compile].freeze

  def test_background_compile_option_compiles_on_the_compile_thread
    assert_runs '[true, true, true, 7]', <<~RUBY, extra_args: BG_BLOCK, stats: :quiet, call_threshold: 2, debug: false
      def add(a, b) = a + b
      10.times { add(1, 2) }
      stats = RubyVM::ZJIT.stats
      [
        stats[:bg_compile_count] > 0,
        stats[:compiled_iseq_count] == stats[:bg_compile_count],
        stats[:bg_compile_overflow_count] == 0,
        add(3, 4),
      ]
    RUBY
  end

  # The default is off: nothing is enqueued and no extra thread appears.
  def test_background_compile_is_off_by_default
    assert_runs '[0, 1]', <<~RUBY, stats: :quiet, call_threshold: 2, debug: false
      def add(a, b) = a + b
      10.times { add(1, 2) }
      [RubyVM::ZJIT.stats[:bg_compile_enqueue_count], Thread.list.size]
    RUBY
  end

  # An ISEQ waiting in the compile queue may be the only thing keeping itself
  # alive: here every method is removed from its class right after being made
  # hot, so nothing but ZJIT's root marking references the ISEQ. Compaction is
  # run in the same window to move what is left.
  def test_background_compile_keeps_queued_iseqs_alive
    assert_runs 'true', <<~RUBY, extra_args: BG, call_threshold: 2, debug: false
      n = 300
      n.times { |i| Object.class_eval "def gone\#{i}(x) = x + \#{i}" }
      o = Object.new
      # Cross the threshold for each, which enqueues it. Neither the calls nor
      # the GCs below release the GVL, so the queue is still full here.
      n.times { |i| 2.times { o.send(:"gone\#{i}", 1) } }
      n.times { |i| Object.send(:remove_method, :"gone\#{i}") }
      5.times { GC.start; GC.compact }
      # Let the compile thread drain what it can still see, then compact again.
      sleep 1
      GC.compact
      true
    RUBY
  end

  # GC.stress runs a collection at every allocation, so the compile thread is
  # marking, moving and freeing ISEQ payloads underneath a queue that is being
  # refilled the whole time.
  def test_background_compile_under_gc_stress
    assert_runs 'true', <<~RUBY, extra_args: BG, call_threshold: 2, debug: false
      n = 40
      n.times { |i| Object.class_eval "def s\#{i}(x) = x + \#{i}" }
      o = Object.new
      GC.stress = true
      n.times { |i| 3.times { o.send(:"s\#{i}", i) } }
      GC.stress = false
      GC.compact
      n.times { |i| raise "wrong result" unless o.send(:"s\#{i}", 1) == 1 + i }
      true
    RUBY
  end

  # Several threads cross compile thresholds while another redefines the very
  # methods and classes those compiles are speculating on. Every invalidation
  # hook runs on a thread holding the GVL, as does the compile, so no
  # invalidation may land between a compile reading the VM and registering the
  # patch point that guards what it read.
  def test_background_compile_invalidation_race
    assert_runs 'true', <<~RUBY, extra_args: BG, call_threshold: 2, debug: false, timeout: 300
      class Target
        VALUE = 1
        def value = VALUE
        def twice = value + value
      end

      stop = false
      workers = 6.times.map do
        Thread.new do
          target = Target.new
          until stop
            1000.times { target.twice }
            # Fresh ISEQs, so there is always new work to enqueue.
            Object.new.singleton_class.class_eval { def only_here = 1 }
          end
        end
      end

      mutator = Thread.new do
        i = 0
        until stop
          i += 1
          # Bust CME assumptions on a method compiled code is speculating on.
          # The string form keeps `VALUE`'s lexical scope inside Target.
          Target.class_eval("def value = VALUE")
          # Bust constant assumptions. Re-setting an existing constant busts the
          # cache without ever leaving it undefined, which the workers are
          # reading concurrently.
          verbose, $VERBOSE = $VERBOSE, nil
          Target.const_set(:VALUE, 1)
          $VERBOSE = verbose
          # Bust NoSingletonClass for a class compiled code has seen.
          Object.new.singleton_class if i.even?
          # Bust "no subclass overrides this method".
          Class.new(Target) { def value = 1 } if i % 8 == 0
          Thread.pass
        end
      end

      sleep 3
      stop = true
      workers.each(&:join)
      mutator.join
      raise "wrong result" unless Target.new.twice == 2
      true
    RUBY
  end

  # Redefining a basic operator invalidates code from underneath in-flight
  # compiles, and does so through a different invalidation hook than method
  # redefinition.
  def test_background_compile_bop_redefinition_race
    assert_runs 'true', <<~RUBY, extra_args: BG, call_threshold: 2, debug: false, timeout: 300
      class Num
        def initialize(v) = @v = v
        def +(other) = Num.new(@v + other.v)
        def v = @v
      end
      def sum(a, b) = a + b

      stop = false
      workers = 4.times.map do
        Thread.new do
          until stop
            500.times { sum(Num.new(1), Num.new(2)) }
            Object.new.singleton_class.class_eval { def fresh = 1 }
          end
        end
      end
      mutator = Thread.new do
        until stop
          Num.class_eval { def +(other) = Num.new(@v + other.v) }
          Thread.pass
        end
      end
      sleep 2
      stop = true
      workers.each(&:join)
      mutator.join
      sum(Num.new(2), Num.new(3)).v == 5
    RUBY
  end

  # More distinct hot ISEQs than the queue can hold. Overflow must drop the
  # request, re-arm the ISEQ's threshold so a later call offers it again, and
  # never block or lose correctness.
  def test_background_compile_queue_overflow
    assert_runs '[true, true]', <<~RUBY, extra_args: BG, stats: :quiet, call_threshold: 2, debug: false, timeout: 300
      n = 3000
      n.times { |i| Object.class_eval "def q\#{i}(x) = x + \#{i}" }
      o = Object.new
      n.times { |i| 2.times { o.send(:"q\#{i}", 1) } }
      n.times { |i| raise "wrong result" unless o.send(:"q\#{i}", 1) == 1 + i }
      # Re-armed requests get another chance, so keep calling and let the
      # compile thread catch up.
      3.times do
        sleep 0.5
        n.times { |i| raise "wrong result" unless o.send(:"q\#{i}", 2) == 2 + i }
      end
      stats = RubyVM::ZJIT.stats
      # Overflow must not be mistaken for a broken compile thread: an enqueuing
      # loop that never yields the GVL legitimately outruns it.
      [stats[:bg_compile_overflow_count] > 0, stats[:bg_compile_disabled_count] == 0]
    RUBY
  end

  # `fork` can only be issued by a thread holding the GVL, and a compile holds
  # the GVL throughout, so a fork can never interrupt one. What it does do is
  # leave the child with no compile thread; the child must notice and start a
  # new one, which drains whatever was still queued at the fork.
  def test_background_compile_fork
    omit 'fork not supported' unless Process.respond_to?(:fork)
    assert_runs 'true', <<~RUBY, extra_args: BG, call_threshold: 2, debug: false, timeout: 300
      n = 200
      n.times { |i| Object.class_eval "def f\#{i}(x) = x + \#{i}" }
      o = Object.new
      # Queue is non-empty from here on.
      n.times { |i| 2.times { o.send(:"f\#{i}", 1) } }

      pid = fork do
        n.times { |i| 50.times { raise "wrong result" unless o.send(:"f\#{i}", 1) == 1 + i } }
        # A fresh method after the fork forces an enqueue, which is where the
        # dead compile thread is noticed and replaced.
        Object.class_eval "def after_fork(x) = x * 3"
        50.times { raise "wrong result" unless o.after_fork(2) == 6 }
        sleep 0.5
        exit!(Thread.list.size == 2 ? 0 : 3)
      end
      _, status = Process.waitpid2(pid)
      # The parent's compile thread is untouched.
      n.times { |i| raise "wrong result" unless o.send(:"f\#{i}", 3) == 3 + i }
      raise "child failed: \#{status.inspect}" unless status.success?
      true
    RUBY
  end

  # Killing the compile thread must degrade to "ZJIT keeps working", not to
  # "ZJIT silently stops compiling".
  def test_background_compile_survives_killed_compile_thread
    assert_runs '[true, true]', <<~RUBY, extra_args: BG, stats: :quiet, call_threshold: 2, debug: false, timeout: 300
      def warm(x) = x + 1
      5.times { warm(1) }
      sleep 0.3
      (Thread.list - [Thread.current]).each { |t| t.kill; t.join }

      n = 100
      n.times { |i| Object.class_eval "def k\#{i}(x) = x + \#{i}" }
      o = Object.new
      n.times { |i| 3.times { o.send(:"k\#{i}", 1) } }
      sleep 1
      n.times { |i| raise "wrong result" unless o.send(:"k\#{i}", 1) == 1 + i }
      stats = RubyVM::ZJIT.stats
      [stats[:bg_compile_thread_restart_count] > 0, stats[:bg_compile_count] > 0]
    RUBY
  end

  # A non-main ractor cannot use the compile thread (it lives in the main
  # ractor), so its compiles stay synchronous. Check both kinds coexist.
  def test_background_compile_ractor
    assert_runs 'true', <<~RUBY, extra_args: BG, call_threshold: 2, debug: false, timeout: 300
      Warning[:experimental] = false
      def shared(x) = x + 1
      100.times { shared(1) }
      ractors = 3.times.map do
        Ractor.new do
          acc = 0
          20_000.times { acc += shared(1) }
          acc
        end
      end
      ractors.map { |r| r.value } == [40_000] * 3
    RUBY
  end

  # Exiting with work still queued must not hang or crash: Ruby kills the
  # compile thread as part of ordinary shutdown.
  def test_background_compile_exit_with_pending_work
    assert_runs 'true', <<~RUBY, extra_args: BG, call_threshold: 2, debug: false
      n = 500
      n.times { |i| Object.class_eval "def x\#{i}(y) = y + \#{i}" }
      o = Object.new
      n.times { |i| 2.times { o.send(:"x\#{i}", 1) } }
      at_exit { }
      true
    RUBY
  end

  # A background compile must never break deadlock detection: the compile thread
  # parks as a deadlockable sleeper precisely so that a program whose only other
  # thread sleeps forever still reports a deadlock.
  def test_background_compile_preserves_deadlock_detection
    script = <<~RUBY
      def warm(x) = x + 1
      5.times { warm(1) }
      sleep 0.2
      Thread.stop
    RUBY
    _out, err, status = EnvUtil.invoke_ruby(
      ['--disable-gems', '--zjit-call-threshold=2', '--zjit-background-compile', '-e', script],
      '', true, true
    )
    refute_predicate status, :success?
    assert_match(/No live threads left. Deadlock/, err)
  end

  private

  # Assert that every method call in `test_script` can be compiled by ZJIT
  # at a given call_threshold
  def assert_compiles(expected, test_script, insns: [], **opts)
    assert_runs(expected, test_script, insns:, assert_compiles: true, **opts)
  end

  # Assert that `test_script` runs successfully with ZJIT enabled.
  # Unlike `assert_compiles`, `assert_runs(assert_compiles: false)`
  # allows ZJIT to skip compiling methods.
  def assert_runs(expected, test_script, insns: [], assert_compiles: false, **opts)
    pipe_fd = 3
    disasm_method = :test

    script = <<~RUBY
      ret_val = (_test_proc = -> { #{('RubyVM::ZJIT.assert_compiles; ' if assert_compiles)}#{test_script.lstrip} }).call
      result = {
        ret_val:,
        #{ unless insns.empty?
           "insns: RubyVM::InstructionSequence.of(method(#{disasm_method.inspect})).to_a"
        end}
      }
      IO.open(#{pipe_fd}).write(Marshal.dump(result))
    RUBY

    out, err, status, result = eval_with_jit(script, pipe_fd:, **opts)
    assert_success(out, err, status)

    result = Marshal.load(result)
    assert_equal(expected, result.fetch(:ret_val).inspect)

    unless insns.empty?
      iseq = result.fetch(:insns)
      assert_equal(
        "YARVInstructionSequence/SimpleDataFormat",
        iseq.first,
        "Failed to get ISEQ disassembly. " \
        "Make sure to put code directly under the '#{disasm_method}' method."
      )
      iseq_insns = iseq.last

      expected_insns = Set.new(insns)
      iseq_insns.each do
        next unless it.is_a?(Array)
        expected_insns.delete(it.first)
      end
      assert(expected_insns.empty?, -> { "Not present in ISeq: #{expected_insns.to_a}" })
    end
  end

  # Run a Ruby process with ZJIT options and a pipe for writing test results
  def eval_with_jit(
    script,
    call_threshold: 1,
    num_profiles: 1,
    zjit: true,
    stats: false,
    debug: true,
    allowed_iseqs: nil,
    extra_args: nil,
    timeout: 1000,
    pipe_fd: nil
  )
    args = ["--disable-gems", *extra_args]
    if zjit
      args << "--zjit-call-threshold=#{call_threshold}"
      args << "--zjit-num-profiles=#{num_profiles}"
      case stats
      when true
        args << "--zjit-stats"
      when :quiet
        args << "--zjit-stats-quiet"
      else
        args << "--zjit-stats=#{stats}" if stats
      end
      args << "--zjit-debug" if debug
      if allowed_iseqs
        jitlist = Tempfile.new("jitlist")
        jitlist.write(allowed_iseqs)
        jitlist.close
        args << "--zjit-allowed-iseqs=#{jitlist.path}"
      end
    end
    args << "-e" << script_shell_encode(script)
    ios = {}
    if pipe_fd
      pipe_r, pipe_w = IO.pipe
      # Separate thread so we don't deadlock when
      # the child ruby blocks writing the output to pipe_fd
      pipe_out = nil
      pipe_reader = Thread.new do
        pipe_out = pipe_r.read
        pipe_r.close
      end
      ios[pipe_fd] = pipe_w
    end
    result = EnvUtil.invoke_ruby(args, '', true, true, rubybin: RbConfig.ruby, timeout: timeout, ios:)
    if pipe_fd
      pipe_w.close
      pipe_reader.join(timeout)
      result << pipe_out
    end
    result
  ensure
    pipe_reader&.kill
    pipe_reader&.join(timeout)
    pipe_r&.close
    pipe_w&.close
    jitlist&.unlink
  end

  def assert_success(out, err, status)
    message = "exited with status #{status.to_i}"
    message << "\nstdout:\n```\n#{out}```\n" unless out.empty?
    message << "\nstderr:\n```\n#{err}```\n" unless err.empty?
    assert status.success?, message
  end

  def script_shell_encode(s)
    # We can't pass utf-8-encoded characters directly in a shell arg. But we can use Ruby \u constants.
    s.chars.map { |c| c.ascii_only? ? c : "\\u%x" % c.codepoints[0] }.join
  end
end
