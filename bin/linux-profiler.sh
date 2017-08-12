#!/bin/sh

#install:
#sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
#git clone https://github.com/brendangregg/FlameGraph
OUTPUTFILE=cpu_profile
EXEFILE="./lighting"
FLAMEGRAPHDIR=/home/dari/Programming/FlameGraph

perf record -F 2000 -g $EXEFILE
perf script | $FLAMEGRAPHDIR/stackcollapse-perf.pl > out.perf-folded
$FLAMEGRAPHDIR/flamegraph.pl out.perf-folded > perf-kernel.svg
firefox perf-kernel.svg

