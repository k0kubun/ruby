# Class-polymorphic method dispatch.
#
# Models the Shopify Storefront Renderer pattern that motivates ZJIT's send
# callcache table (see zjit/src/send_cache.rs): one call site whose receiver is
# any of hundreds of unrelated classes, so no bounded inline class-guard chain
# can cover the traffic and every call falls through to a full dynamic dispatch.
# On SFR that is ~14M megamorphic sends per benchmark run, 42% of all dynamic
# sends.
#
# Three receiver distributions are measured, because they stress the table in
# opposite ways:
#
#   cyclic  every class is touched exactly once per pass, in a stride order. The
#           worst case for a direct-mapped table: any slot holding two live
#           classes misses on every access.
#   skewed  class rank r is picked with weight 1/(r+1), which is what a real
#           application's receiver distribution at one site looks like.
#   random  uniform over the population with a fixed seed.
#
# The mono/poly cases are controls: those sites are covered by ZJIT's inline
# class-guard chain and must not regress.
#
#   ruby --zjit benchmark/zjit_megamorphic_send.rb
#   ruby --zjit --zjit-stats benchmark/zjit_megamorphic_send.rb   # table counters
#   ruby --zjit --zjit-disable-send-cache benchmark/zjit_megamorphic_send.rb
#
# Wall and CPU time both swing on a loaded machine; for A/B work prefer
# `perf stat -e instructions` over one iteration count and compare the totals.
#
# Env knobs: CLASSES, ITERS, REPS.

CLASSES = Integer(ENV.fetch("CLASSES", 200))
ITERS   = Integer(ENV.fetch("ITERS", 2000))
REPS    = Integer(ENV.fetch("REPS", 5))

# Every class defines the same three methods, so the *name* is shared by all
# CLASSES receivers and only the class differs: exactly the shape of a
# megamorphic site, and the shape a per-name table has to cope with.
KLASSES = Array.new(CLASSES) do |i|
  Class.new do
    define_method(:value) { i }
    define_method(:value_with_args) { |a, b| a + b + i }
    define_method(:value_with_block) { |&blk| blk ? blk.call(i) : i }
  end
end

# A deep-ish hierarchy so the lookup the table memoizes is not trivially at
# depth 0. Ruby's own lookup walks the ancestors on a cc miss.
MIXIN = Module.new { def mixed_in = 42 }
KLASSES.each { |k| k.include(MIXIN) }

class Mono
  def value = 1
  def value_with_args(a, b) = a + b
  def value_with_block(&blk) = blk ? blk.call(1) : 1
  def mixed_in = 42
end

# Four classes: within ZJIT's inline class-guard chain, so these sites stay
# polymorphic-specialized and must be unaffected by the table.
POLY_KLASSES = Array.new(4) do |i|
  Class.new do
    define_method(:value) { i }
    define_method(:value_with_args) { |a, b| a + b + i }
    define_method(:value_with_block) { |&blk| blk ? blk.call(i) : i }
    define_method(:mixed_in) { 42 }
  end
end

POP = KLASSES.map(&:new).freeze

def cyclic(pop, n) = Array.new(n) { |i| pop[(i * 97) % pop.size] }

def skewed(pop, n)
  cdf = []
  acc = 0.0
  total = (0...pop.size).sum { |r| 1.0 / (r + 1) }
  pop.each_index { |r| acc += 1.0 / (r + 1); cdf << acc / total }
  rng = Random.new(1234)
  Array.new(n) { x = rng.rand; pop[cdf.index { |c| c >= x } || pop.size - 1] }
end

def uniform(pop, n)
  rng = Random.new(4321)
  Array.new(n) { pop[rng.rand(pop.size)] }
end

N = 2 * CLASSES
POPS = {
  "cyclic" => cyclic(POP, N).freeze,
  "skewed" => skewed(POP, N).freeze,
  "random" => uniform(POP, N).freeze,
}
MONO = Array.new(N) { Mono.new }.freeze
POLY = uniform(POLY_KLASSES.map(&:new), N).freeze

# Each of these is a distinct call site, so each gets its own profile and its
# own dispatch decision.
def bench_value(pop) = pop.each { |o| o.value }
def bench_args(pop) = pop.each { |o| o.value_with_args(1, 2) }
def bench_block(pop) = pop.each { |o| o.value_with_block { |x| x } }
def bench_mixin(pop) = pop.each { |o| o.mixed_in }

# `respond_to?`-free two-site variant: two sites calling the same name, which a
# per-name table lets warm each other.
def bench_two_sites(pop)
  pop.each { |o| o.value }
  pop.each { |o| o.value }
end

benches = []
POPS.each do |kind, pop|
  benches << ["value/#{kind}", method(:bench_value), pop]
  benches << ["args/#{kind}", method(:bench_args), pop]
  benches << ["block/#{kind}", method(:bench_block), pop]
  benches << ["mixin/#{kind}", method(:bench_mixin), pop]
  benches << ["2sites/#{kind}", method(:bench_two_sites), pop]
end
benches << ["value/mono", method(:bench_value), MONO]
benches << ["args/mono", method(:bench_args), MONO]
benches << ["block/mono", method(:bench_block), MONO]
benches << ["value/poly", method(:bench_value), POLY]
benches << ["args/poly", method(:bench_args), POLY]

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
