#!/bin/sh
#sudo apt-get install libgoogle-perftools-de

OUTPUTFILE=cpu_profile
EXEFILE="./lighting"

export LD_PRELOAD=/usr/lib/libprofiler.so 
export CPUPROFILE=$OUTPUTFILE
export CPUPROFILE_FREQUENCY=250

$EXEFILE

export LD_PRELOAD=""
google-pprof --gv $EXEFILE $OUTPUTFILE
