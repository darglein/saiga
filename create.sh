#!/bin/sh

rm -rf build
mkdir build
cd build
cmake -DSAIGA_MODULE_CUDA=OFF -DCMAKE_PREFIX_PATH=~/Programming/EigenRecursive -DSAIGA_NO_INSTALL=ON ..
cd ..

