# Preparation:
#   $ uname -r # check kernel version
#   $ sudo apt remove linux-tools-6.2.0-33-generic # uninstall perf
#
# Install perf with Python support:
#   # optional: libelf-dev libunwind-dev libaudit-dev libslang2-dev libdw-dev
#   $ sudo apt-get install libpython3-dev python3-pip flex libtraceevent-dev
#   $ git clone --branch=v6.2 https://github.com/torvalds/linux
#   $ cd linux/tools/perf
#   $ make
#   $ make install
#
# Usage:
#   # perf record -F max -e cycles -p $pid
#   $ ruby --yjit-perf -Iharness-perf benchmarks/lobsters/benchmark.rb
#   $ perf script -s ../../ruby/ruby/misc/yjit_perf.py

import os
import sys
import math
import struct
from collections import Counter

sys.path.append(os.environ['PERF_EXEC_PATH'] + \
        '/scripts/python/Perf-Trace-Util/lib/Perf/Trace')

from perf_trace_context import *
from EventClass import *

total_cycles = 0
jited_cycles = 0
symbol_cycles = Counter()

def process_event(event):
    global total_cycles, jited_cycles, symbol_cycles

    sample = event["sample"]
    # Symbol and dso info are not always resolved
    dso = event.get("dso", "Unknown_dso")

    # It looks like "period" is cycle count of the sample
    cycles = sample["period"]
    total_cycles += cycles

    if dso.endswith("map"):
        symbol = event.get("symbol", "[unknown]")
        symbol_cycles[symbol] += cycles
        jited_cycles += cycles

def trace_end():
    print("total cycles: {}".format(total_cycles))
    if total_cycles:
        print("JITed cycles: {} ({:.1f}%)".format(jited_cycles, jited_cycles / total_cycles * 100))
        max_symbol_len = max([len(i) for i in symbol_cycles.keys()])
        for symbol, cycles in symbol_cycles.most_common():
            print("{} {: 5.1f}% {}".format(symbol.ljust(max_symbol_len), cycles / jited_cycles * 100, cycles))
