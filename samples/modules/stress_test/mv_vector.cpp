/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

//#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 10

#include "saiga/core/time/performanceMeasure.h"
#include "saiga/core/util/crash.h"
#include "saiga/core/math/random.h"

#include "Eigen/Core"

#include <random>

using namespace Saiga;


// View Assembly code
// 1. Go to the directory of the .o file.
//        cd saiga/build/samples/modules/eigen/CMakeFiles/modules_eigen.dir
// 2. Dumb assembly to file
//        objdump -d -M intel -S mv_vector.cpp.o > vector_asm.txt


void testMatrixVector2()
{
    const int size = 8;
    using T        = double;
    using Mat      = Eigen::Matrix<T, size, size>;
    using Vec      = Eigen::Matrix<T, size, 1>;



    Mat A;
    Vec y, x;



    //    Eigen::internal::product_type
    //    Eigen::internal::general_matrix_vector_product
    // asm("# Start y=A*x");
    y = (A * x).eval();
    // asm("# End y=A*x");

    std::cout << y << std::endl;
}

void testMatrixVector()
{
    std::cout << "testMatrixVector" << std::endl;

    testMatrixVector2();

    std::cout << "testMatrixVector done." << std::endl;
}
