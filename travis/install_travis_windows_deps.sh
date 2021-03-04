VCPKGDIR="C:/Users/travis/vcpkg"
mkdir $VCPKGDIR
cd "C:/Users/travis/"


export VCPKG_DEFAULT_TRIPLET=x64-windows
git clone https://github.com/Microsoft/vcpkg.git $VCPKGDIR
cd $VCPKGDIR
./bootstrap-vcpkg.bat
./vcpkg integrate install
./vcpkg install zlib
./vcpkg install libpng
./vcpkg install sdl2
./vcpkg install freetype
./vcpkg install glfw3
./vcpkg install gtest

#ls -a installed/x64-windows
cd ..

SRC_DIR="C:/Users/travis/eigen"
INSTALL_DIR="C:/Users/travis/install"

mkdir $INSTALL_DIR

git clone https://gitlab.com/libeigen/eigen.git $SRC_DIR
cd $SRC_DIR
git checkout 28aef8e816faadc0e51afbfe3fa91f10f477535d
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -G "Visual Studio 15 2017 Win64" ..
cmake --build . -j4
cmake --install .
