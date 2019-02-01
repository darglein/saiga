/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/time/performanceMeasure.h"
#include "saiga/core/util/crash.h"
#include "saiga/core/util/random.h"

#include "Eigen/Core"

#include <random>

using namespace Saiga;


// objdump -d -M intel -S test.o

void testMatrixVector()
{
    cout << "testMatrixVector" << endl;


    const int size = 8;
    using T        = double;
    using Mat      = Eigen::Matrix<T, size, size>;
    using Vec      = Eigen::Matrix<T, size, 1>;

    Mat A = Mat::Random();
    Vec y, x;

    x = Vec::Random();

    asm("# Start y=A*x");
    y = A * x;
    asm("# End y=A*x");

    cout << "testMatrixVector done." << endl;
}
