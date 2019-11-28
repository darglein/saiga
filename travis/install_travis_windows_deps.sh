mkdir "C:/Users/travis/install"

git clone https://github.com/eigenteam/eigen-git-mirror.git "C:/Users/travis/eigen"
cd "C:/Users/travis/eigen"
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="C:/Users/travis/install" -G "Visual Studio 15 2017 Win64" ..
cmake --build . -j4
cmake --install .
