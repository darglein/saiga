SRC_DIR="C:/Users/travis/eigen"
INSTALL_DIR="C:/Users/travis/install"

mkdir $INSTALL_DIR

git clone https://gitlab.com/libeigen/eigen.git SRC_DIR
cd SRC_DIR
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -G "Visual Studio 15 2017 Win64" ..
cmake --build . -j4
cmake --install .
