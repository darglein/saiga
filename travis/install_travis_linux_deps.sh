#!/bin/bash

# Stop processing on any error.
set -e

sudo apt-get install -y cmake;
sudo apt-get install -y g++-8;
sudo apt-get install -y clang-8;
sudo apt-get install -y libfreetype6-dev libglm-dev;
sudo apt-get install -y libegl1-mesa-dev;
sudo apt-get install -y libsdl2-dev libglfw3-dev;
sudo apt-get install -y libpng-dev libfreeimage-dev libfreeimageplus-dev;
sudo apt-get install -y libopenal-dev libopus-dev libopusfile-dev;
sudo apt-get install -y libavutil-dev libavcodec-dev libavresample-dev libswscale-dev libavformat-dev;
sudo apt-get install -y libassimp-dev libeigen3-dev libsuitesparse-dev;
sudo apt-get install -y libgoogle-glog-dev libgflags-dev
