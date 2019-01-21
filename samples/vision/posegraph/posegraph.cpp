/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/vision/scene/PoseGraph.h"

#include "saiga/time/timer.h"
#include "saiga/util/random.h"
#include "saiga/vision/Eigen_Compile_Checker.h"
#include "saiga/vision/g2o/g2oPoseGraph.h"
using namespace Saiga;

int main(int, char**)
{
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Saiga::Random::setSeed(93865023985);


    PoseGraph pg;
    pg.load("test.posegraph");
    pg.addNoise(0.1);
    cout << "chi2 " << pg.chi2() << endl;


    g2oPoseGraph opg;
    opg.solve(pg);

    return 0;
}
