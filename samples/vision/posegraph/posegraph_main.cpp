/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/framework/framework.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/random.h"
#include "saiga/vision/BALDataset.h"
#include "saiga/vision/Eigen_Compile_Checker.h"
#include "saiga/vision/ceres/CeresPGO.h"
#include "saiga/vision/g2o/g2oPoseGraph.h"
#include "saiga/vision/pgo/PGORecursive.h"
#include "saiga/vision/scene/PoseGraph.h"
using namespace Saiga;

int main(int, char**)
{
    Saiga::SaigaParameters saigaParameters;
    Saiga::initSample(saigaParameters);
    Saiga::initSaiga(saigaParameters);

    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Saiga::Random::setSeed(93865023985);


    //    std::string path = "vision/problem-00257-65132-pre.txt";
    std::string path = "vision/problem-00356-226730-pre.txt";
    //    std::string path = "vision/problem-00257-65132-pre.txt";
    //    std::string path = "vision/problem-00257-65132-pre.txt";

    Saiga::BALDataset bald(SearchPathes::data(path));
    Scene scene = bald.makeScene();

    PoseGraph pg(scene);
    //    pg.load(SearchPathes::data("vision/slam_30_431.posegraph"));
    //    pg.load(SearchPathes::data("vision/slam_125_3495.posegraph"));
    //    pg.load(SearchPathes::data("vision/loop.posegraph"));
    pg.addNoise(0.05);
    cout << endl;


    PGOOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.maxIterations          = 3;
    baoptions.maxIterativeIterations = 10;
    baoptions.iterativeTolerance     = 1e-50;
    //    baoptions.solverType             = PGOOptions::SolverType::Direct;
    baoptions.solverType = PGOOptions::SolverType::Iterative;

    std::vector<std::shared_ptr<PGOBase>> solvers;

    solvers.push_back(std::make_shared<PGORec>());
    solvers.push_back(std::make_shared<g2oPGO>());
    solvers.push_back(std::make_shared<CeresPGO>());

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
