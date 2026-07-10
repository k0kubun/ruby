#!/usr/bin/env ruby

class JITPerf
  INTERPRETER_SYMBOLS = [
    # rb_* entry points
    "rb_call0",
    "rb_funcallv_scope",
    "rb_yield",

    # VM helpers without rb_vm_/vm_ prefixes
    "callable_method_entry_or_negative",
    "invoke_block_from_c_bh",
    "setup_parameters_complex",
  ].freeze

  def initialize
    @total_cycles = 0
    @category_cycles = Hash.new(0)
    @detailed_category_cycles = Hash.new { |hash, category| hash[category] = Hash.new(0) }
    @categories = {}
    @perf_maps = {}
  end

  def read(path)
    File.foreach(path).with_index(1) do |line, lineno|
      next if line.strip.empty?

      process_event(parse_line(line))
    rescue ArgumentError => error
      abort "#{path}:#{lineno}: #{error.message}"
    end
  rescue SystemCallError => error
    abort "#{path}: #{error.message}"
  end

  def print_report
    return if @total_cycles == 0

    puts "Aggregated Event Data:"
    puts format("%-20s %-50s %20s %15s", "[dso]", "[symbol or category]", "[top-most cycle ratio]", "[num cycles]")

    most_common(@category_cycles).each do |category, cycles|
      ratio = cycles.to_f / @total_cycles * 100
      dsos = @detailed_category_cycles[category].each_key.map(&:first).uniq
      dso_display = dsos.length == 1 ? dsos.first : "Multiple DSOs"
      puts format("%-20s %-50s %20.2f%% %15d", dso_display, truncate_symbol(category), ratio, cycles)
    end

    most_common(@category_cycles).each do |category, _cycles|
      next unless @categories.key?(category)

      symbols = @detailed_category_cycles[category]
      category_total = symbols.values.sum
      category_ratio = category_total.to_f / @total_cycles * 100

      puts
      puts format("Category: %s (%.2f%%)", category, category_ratio)
      puts format("%-20s %-50s %20s %15s", "[dso]", "[symbol]", "[top-most cycle ratio]", "[num cycles]")

      most_common(symbols).each do |(dso, symbol), cycles|
        symbol_ratio = cycles.to_f / category_total * 100
        puts format("%-20s %-50s %20.2f%% %15d", dso, truncate_symbol(symbol), symbol_ratio, cycles)
      end
    end
  end

  private

  def parse_line(line)
    # Example:
    # ruby 78207 3482.848465: 1212775 cpu_core/cycles:P/: 5c0333f682e1 [JIT] getlocal_WC_0+0x0 (/tmp/perf-78207.map)
    #
    # Split into command, pid, timestamp, period, event, ip, and the remaining
    # "symbol (dso)" text. The final field is kept intact because symbols can
    # contain spaces.
    fields = line.split(nil, 7)
    raise ArgumentError, "unexpected perf script line: #{line.chomp}" if fields.length < 7

    begin
      period = Integer(fields[3])
    rescue ArgumentError, TypeError
      raise ArgumentError, "unexpected sample period in perf script line: #{line.chomp}"
    end

    # Parse the trailing "symbol (dso)" text from the right, then drop the
    # instruction offset after "+" from the symbol name.
    dso_start = fields[6].rindex(" (")
    raise ArgumentError, "missing dso in perf script line: #{line.chomp}" unless dso_start

    dso_with_suffix = fields[6][(dso_start + 2)..-1]
    dso_end = dso_with_suffix.index(")")
    raise ArgumentError, "missing dso terminator in perf script line: #{line.chomp}" unless dso_end

    symbol = fields[6][0...dso_start].split("+", 2).first
    dso = dso_with_suffix[0...dso_end]

    begin
      pid = Integer(fields[1].split("/", 2).first)
      ip = Integer(fields[5], 16)
    rescue ArgumentError, TypeError
      raise ArgumentError, "unexpected pid or IP in perf script line: #{line.chomp}"
    end

    [pid, ip, dso, symbol, period]
  end

  def process_event(event)
    pid, ip, full_dso, symbol, cycles = event
    full_dso, symbol = resolve_jit_symbol(pid, ip, full_dso, symbol)
    dso = File.basename(full_dso || "Unknown_dso")
    symbol ||= "[unknown]"

    @total_cycles += cycles

    category = categorize_symbol(dso, symbol)
    @category_cycles[category] += cycles
    @detailed_category_cycles[category][[dso, symbol]] += cycles

    @categories[category] = true if category.start_with?("[") && category.end_with?("]")
  end

  def resolve_jit_symbol(pid, ip, dso, symbol)
    return [dso, symbol] unless symbol == "[unknown]"
    return [dso, symbol] unless dso == "[anon:Ruby:rb_jit_reserve_addr_space]" || (dso && File.basename(dso).start_with?("perf-"))

    if (perf_symbol = lookup_perf_symbol(pid, ip))
      ["/tmp/perf-#{pid}.map", perf_symbol]
    else
      [dso, symbol]
    end
  end

  def lookup_perf_symbol(pid, ip)
    entries = perf_map_entries(pid)
    return nil if entries.empty?

    idx = entries.bsearch_index { |entry| entry[0] > ip } || entries.length
    idx -= 1
    while idx >= 0 && entries[idx][4] > ip
      entry = entries[idx]
      return entry[3] if entry[1] > ip

      idx -= 1
    end
    nil
  end

  def perf_map_entries(pid)
    @perf_maps.fetch(pid) do
      path = "/tmp/perf-#{pid}.map"
      entries = []
      File.foreach(path).with_index do |line, index|
        start, size, *symbol = line.split
        next unless start && size

        start_addr = Integer(start, 16)
        code_size = Integer(size, 16)
        next if code_size == 0

        entries << [start_addr, start_addr + code_size, index, symbol.join(" ")]
      rescue ArgumentError
        next
      end
      entries.sort_by! { |start_addr, _end_addr, index, _symbol| [start_addr, index] }
      max_end_addr = 0
      entries.each do |entry|
        max_end_addr = [max_end_addr, entry[1]].max
        entry << max_end_addr
      end
      @perf_maps[pid] = entries
    rescue SystemCallError
      @perf_maps[pid] = []
    end
  end

  def truncate_symbol(symbol, max_length = 50)
    symbol.length <= max_length ? symbol : "#{symbol[0...(max_length - 3)]}..."
  end

  def categorize_symbol(dso, symbol)
    if dso == "sqlite3_native.so"
      "[sqlite3]"
    elsif symbol.include?("SHA256")
      "[sha256]"
    elsif symbol.start_with?("[JIT] gen_send")
      "[JIT send]"
    elsif symbol.start_with?("[JIT]") || symbol.start_with?("ZJIT: ") || dso.start_with?("perf-")
      "[JIT code]"
    elsif symbol.include?("::") || symbol.start_with?("_ZN4yjit") || symbol.start_with?("_ZN4zjit")
      "[JIT compile]"
    elsif symbol.start_with?("rb_vm_") || symbol.start_with?("vm_") || INTERPRETER_SYMBOLS.include?(symbol)
      "[interpreter]"
    elsif symbol.start_with?("rb_hash_") || symbol.start_with?("hash_")
      "[rb_hash_*]"
    elsif symbol.start_with?("rb_ary_") || symbol.start_with?("ary_")
      "[rb_ary_*]"
    elsif symbol.start_with?("rb_str_") || symbol.start_with?("str_")
      "[rb_str_*]"
    elsif symbol.start_with?("rb_sym") || symbol.start_with?("sym_")
      "[rb_sym_*]"
    elsif symbol.start_with?("rb_st_") || symbol.start_with?("st_")
      "[rb_st_*]"
    elsif symbol.start_with?("rb_ivar_") || symbol.include?("shape")
      "[ivars]"
    elsif symbol.include?("match") || symbol.start_with?("rb_reg") || symbol.start_with?("onig")
      "[regexp]"
    elsif symbol.include?("alloc") || symbol.include?("free") || symbol.include?("gc")
      "[GC]"
    elsif symbol.include?("pthread") && symbol.include?("lock")
      "[pthread lock]"
    else
      symbol
    end
  end

  def most_common(counter)
    counter.each.with_index
      .sort_by { |((_key, cycles), index)| [-cycles, index] }
      .map(&:first)
  end
end

if ARGV.length != 1
  abort "Usage: #{File.basename($PROGRAM_NAME)} <perf-script-output>"
end

jit_perf = JITPerf.new
jit_perf.read(ARGV[0])
jit_perf.print_report
