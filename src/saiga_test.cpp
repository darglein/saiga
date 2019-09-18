/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"

using namespace Saiga;

int main()
{
    initSaigaSampleNoWindow();


    auto saigaFlags = EigenHelper::getSaigaEigenCompileFlags();
    std::cout << saigaFlags << std::endl;

    return 0;
}
