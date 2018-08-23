#!/bin/sh

#install:
#sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
#git clone https://github.com/brendangregg/FlameGraph
OUTPUTFILE=perf.data
EXEFILE="$1"

#set your own flamegraph directory here
FLAMEGRAPHDIR=/home/dari/Programming/FlameGraph


#sudo sh -c 'echo 0 >/proc/sys/kernel/perf_event_paranoid'
#sudo sh -c 'echo kernel.perf_event_paranoid=0 > /etc/sysctl.d/local.conf'

perf record -g --call-graph dwarf -a -e instructions:u -F 1000 -o $OUTPUTFILE -- $EXEFILE

perf script | $FLAMEGRAPHDIR/stackcollapse-perf.pl > out.perf-folded
$FLAMEGRAPHDIR/flamegraph.pl out.perf-folded > perf-kernel.svg
firefox perf-kernel.svg

