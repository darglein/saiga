#!/bin/bash

INPUT=Module_CMakeLists.txt
TARGET=CMakeLists.txt


for f in *; do
    if [[ -d $f ]]; then
        echo "cp $INPUT $f/$TARGET"
        cp $INPUT $f/$TARGET
    fi
done
