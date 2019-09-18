


BASEDIR=$(dirname "$0")

cd ${BASEDIR}

BASEDIR=$PWD

ROOT_DIR=$BASEDIR/..
INSTALL_DIR=$BASEDIR/install


###################################################################
# Download Sophus
cd ${BASEDIR}
git clone https://github.com/strasdat/Sophus.git Sophus
cd Sophus
git checkout 96d109d3079df93cf1561dbcbb551f12e374e149
mkdir build
cd build
cmake -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON ..
make -j4 install
 
# Copy source to vision/sophus
cd ${ROOT_DIR}
mkdir src/saiga/vision/sophus/External
cp -r ${INSTALL_DIR}/include/sophus/* src/saiga/vision/sophus/External




###################################################################
# Download EigenRecursive
cd ${BASEDIR}
git clone https://github.com/darglein/EigenRecursive.git EigenRecursive
cd EigenRecursive
git checkout 800cdea9b30daba62f8566657ea248da6761d7fb
mkdir build
cd build
cmake -DBUILD_SAMPLES=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON ..
make -j4 install

# Copy source to vision/sophus
cd ${ROOT_DIR}
mkdir src/saiga/vision/recursive/External
cp -r ${INSTALL_DIR}/include/EigenRecursive/* src/saiga/vision/recursive/External
