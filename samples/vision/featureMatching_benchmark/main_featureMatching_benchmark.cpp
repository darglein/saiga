/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/framework/framework.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/random.h"
#include "saiga/vision/Eigen_Compile_Checker.h"
#include "saiga/vision/VisionIncludes.h"


int main(int, char**)
{
    Saiga::SaigaParameters saigaParameters;
    Saiga::initSample(saigaParameters);
    Saiga::initSaiga(saigaParameters);

    Saiga::EigenHelper::checkEigenCompabitilty<15357>();
    Saiga::Random::setSeed(93865023985);

    return 0;
}
