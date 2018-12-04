#!/bin/sh

# Copy this file to the parent directory before executing (TODO)

find -iname *.h -o -iname *.cpp -o -iname *.inl -o -iname *.cc -o -iname *.cu | xargs clang-format -i
