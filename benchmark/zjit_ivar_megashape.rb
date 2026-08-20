# Shape-polymorphic instance variable access.
#
# Models the Shopify Storefront Renderer pattern that motivates ZJIT's ivar
# shape table (see zjit/src/ivar_cache.rs): a single class whose instances stop
# at many different points in one long ivar chain, so every ivar site sees
# hundreds of distinct shape ids and no bounded inline guard chain can cover the
# traffic.
#
# Two access distributions are measured, because they stress the table in
# opposite ways:
#
#   cyclic  every shape is touched exactly once per pass, in a stride order. The
#           worst case for a direct-mapped table: any slot holding two live
#           shapes misses on every access.
#   skewed  shape rank r is picked with weight 1/(r+1), which is what a real
#           application's shape distribution at one site looks like.
#
# The population is DEPTHS instances at distinct chain depths, doubled by
# freezing half of them (the frozen bit is part of shape_id, so it doubles the
# shape count without adding shape *variations*, which past SHAPE_MAX_VARIATIONS
# would demote the class to hash-backed complex shapes).
#
#   ruby --zjit benchmark/zjit_ivar_megashape.rb
#   ruby --zjit --zjit-stats benchmark/zjit_ivar_megashape.rb   # table counters
#   ruby --zjit --zjit-disable-ivar-cache benchmark/zjit_ivar_megashape.rb
#
# Wall and CPU time both swing by 2x on a loaded machine; for A/B work prefer
# `perf stat -e instructions` over two iteration counts and compare the slope.
#
# Env knobs: ITERS, REPS, DEPTHS.

DEPTHS = Integer(ENV.fetch("DEPTHS", 110))
ITERS  = Integer(ENV.fetch("ITERS", 2000))
REPS   = Integer(ENV.fetch("REPS", 5))

class Mega
  body = +"def initialize(depth)\n"
  DEPTHS.times do |i|
    body << "  @i#{i} = #{i}\n"
    body << "  return if depth == #{i + 1}\n"
  end
  body << "end\n"
  eval(body) # rubocop:disable Security/Eval

  attr_reader :i0
  attr_writer :i0

  def read_first = @i0
  def read_absent = @never_assigned
  def write_first(v) = (@i0 = v)
end

class Mono
  def initialize
    @a = 1
    @b = 2
  end
  attr_reader :a
  attr_writer :a
  def read_a = @a
end

Mega.new(DEPTHS) # prime RCLASS_MAX_IV_COUNT so every instance is embedded

READ_POP = ((1..DEPTHS).map { |d| Mega.new(d) } +
            (1..DEPTHS).map { |d| Mega.new(d).freeze })
WRITE_POP = (1..DEPTHS).map { |d| Mega.new(d) }

def cyclic(pop, n) = Array.new(n) { |i| pop[(i * 97) % pop.size] }

def skewed(pop, n)
  cdf = []
  acc = 0.0
  total = (0...pop.size).sum { |r| 1.0 / (r + 1) }
  pop.each_index { |r| acc += 1.0 / (r + 1); cdf << acc / total }
  rng = Random.new(1234)
  Array.new(n) { x = rng.rand; pop[cdf.index { |c| c >= x } || pop.size - 1] }
end

POPS = {
  "cyclic" => [cyclic(READ_POP, 2 * DEPTHS).freeze, cyclic(WRITE_POP, 2 * DEPTHS).freeze],
  "skewed" => [skewed(READ_POP, 2 * DEPTHS).freeze, skewed(WRITE_POP, 2 * DEPTHS).freeze],
}
MONO = Array.new(2 * DEPTHS) { Mono.new }.freeze

def bench_attr_reader(pop) = pop.each { |o| o.i0 }
def bench_plain_read(pop) = pop.each { |o| o.read_first }
def bench_absent_read(pop) = pop.each { |o| o.read_absent }
def bench_attr_writer(pop) = pop.each { |o| o.i0 = 7 }
def bench_plain_write(pop) = pop.each { |o| o.write_first(9) }
def bench_mono_reader(pop) = pop.each { |o| o.a }
def bench_mono_plain(pop) = pop.each { |o| o.read_a }
def bench_mono_writer(pop) = pop.each { |o| o.a = 3 }

benches = []
POPS.each do |kind, (read_pop, write_pop)|
  benches << ["attr_reader/#{kind}", method(:bench_attr_reader), read_pop]
  benches << ["plain read/#{kind}", method(:bench_plain_read), read_pop]
  benches << ["absent read/#{kind}", method(:bench_absent_read), read_pop]
  benches << ["attr_writer/#{kind}", method(:bench_attr_writer), write_pop]
  benches << ["plain write/#{kind}", method(:bench_plain_write), write_pop]
end
benches << ["attr_reader/mono", method(:bench_mono_reader), MONO]
benches << ["plain read/mono", method(:bench_mono_plain), MONO]
benches << ["attr_writer/mono", method(:bench_mono_writer), MONO]

# Warm every site up so ZJIT compiles it (and finishes respecializing) first.
benches.each { |_, m, pop| 60.times { m.call(pop) } }

width = benches.map { |name, _, _| name.length }.max
totals = Hash.new(0.0)
benches.each do |name, m, pop|
  best = Float::INFINITY
  REPS.times do
    t0 = Process.clock_gettime(Process::CLOCK_PROCESS_CPUTIME_ID)
    ITERS.times { m.call(pop) }
    t1 = Process.clock_gettime(Process::CLOCK_PROCESS_CPUTIME_ID)
    best = [best, t1 - t0].min
  end
  totals[name.split("/").last] += best
  puts format("%-#{width}s  %8.4f s  %7.2f ns/op", name, best, best / (ITERS * pop.size) * 1e9)
end
totals.each { |kind, secs| puts format("%-#{width}s  %8.4f s", "TOTAL #{kind}", secs) }
