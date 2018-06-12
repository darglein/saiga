#!/bin/sh

for d in */ ; do
#    echo "$d"
	cp CMakeLists_samples.txt "$d/CMakeLists.txt"
done
