/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/vision/scene/PoseGraph.h"

#include "saiga/framework/framework.h"
#include "saiga/time/timer.h"
#include "saiga/util/fileChecker.h"
#include "saiga/util/random.h"
#include "saiga/vision/Eigen_Compile_Checker.h"
#include "saiga/vision/g2o/g2oPoseGraph.h"
#include "saiga/vision/pgo/PGORecursive.h"
using namespace Saiga;

int main(int, char**)
{
    Saiga::SaigaParameters saigaParameters;
    Saiga::initSample(saigaParameters);
    Saiga::initSaiga(saigaParameters);

    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Saiga::Random::setSeed(93865023985);


    PoseGraph pg;
    //    pg.load(SearchPathes::data("vision/slam_30_431.posegraph"));
    pg.load(SearchPathes::data("vision/slam_125_3495.posegraph"));
    pg.addNoise(0.1);
    cout << endl;


    PGOOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.maxIterations          = 10;
    baoptions.maxIterativeIterations = 20;
    baoptions.iterativeTolerance     = 1e-10;
    //    baoptions.solverType             = PGOOptions::SolverType::Direct;
    baoptions.solverType = PGOOptions::SolverType::Iterative;

    std::vector<std::shared_ptr<PGOBase>> solvers;

    solvers.push_back(std::make_shared<PGORec>());
    solvers.push_back(std::make_shared<g2oPGO>());

    for (auto& s : solvers)
    {
        cout << "[Solver] " << s->name << endl;
        auto cpy       = pg;
        auto rmsbefore = cpy.chi2();
        {
            SAIGA_BLOCK_TIMER(s->name);
            s->solve(cpy, baoptions);
        }
        cout << "Error " << rmsbefore << " -> " << cpy.chi2() << endl << endl;
    }

    return 0;
}
