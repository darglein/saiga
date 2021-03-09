#!/bin/bash

# Stop processing on any error.
set -e

SRC_DIR="/home/travis/eigen"
INSTALL_DIR="/home/travis/install"

mkdir $INSTALL_DIR

git clone https://gitlab.com/libeigen/eigen.git $SRC_DIR
cd $SRC_DIR
git checkout 28aef8e816faadc0e51afbfe3fa91f10f477535d
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ..
cmake --build . -j4
cmake --install .
