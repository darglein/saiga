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
#include "saiga/vision/scene/SynteticScene.h"
using namespace Saiga;

int main(int, char**)
{
    Saiga::SaigaParameters saigaParameters;
    Saiga::initSample(saigaParameters);
    Saiga::initSaiga(saigaParameters);

    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Saiga::Random::setSeed(93865023985);


    //    std::string path = "vision/problem-00257-65132-pre.txt";
    //    std::string path = "vision/problem-00356-226730-pre.txt";
    //    std::string path = "vision/problem-00257-65132-pre.txt";
    //    std::string path = "vision/problem-00257-65132-pre.txt";

    //    Saiga::BALDataset bald(SearchPathes::data(path));
    //    Scene scene = bald.makeScene();

    Scene scene;
    //    scene.load(SearchPathes::data("vision/tum_office.scene"));
    scene.load(SearchPathes::data("vision/tum_large.scene"));

    //    SynteticScene sscene;
    //    sscene.numCameras     = 2;
    //    sscene.numImagePoints = 2;
    //    sscene.numWorldPoints = 7;
    //    scene                 = sscene.circleSphere();
    //    scene.addWorldPointNoise(0.01);
    //    scene.addImagePointNoise(1.0);
    //    scene.addExtrinsicNoise(0.01);


    cout << "Density: " << scene.getSchurDensity() << endl;
    PoseGraph pg(scene);
    //    pg.load(SearchPathes::data("vision/slam_30_431.posegraph"));
    //    pg.load(SearchPathes::data("vision/slam_125_3495.posegraph"));
    //    pg.load(SearchPathes::data("vision/loop.posegraph"));
    pg.addNoise(1.05);
    cout << endl;


    OptimizationOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.maxIterations          = 10;
    baoptions.maxIterativeIterations = 15;
    baoptions.iterativeTolerance     = 1e-50;
    //    baoptions.initialLambda          = 1e3;
    baoptions.solverType = OptimizationOptions::SolverType::Direct;
    //    baoptions.solverType = OptimizationOptions::SolverType::Iterative;
    cout << baoptions << endl;


    std::vector<std::unique_ptr<PGOBase>> solvers;

    solvers.push_back(std::make_unique<PGORec>());
    solvers.push_back(std::make_unique<g2oPGO>());
    solvers.push_back(std::make_unique<CeresPGO>());

    for (auto& s : solvers)
    {
        cout << "[Solver] " << s->name << endl;
        auto cpy = pg;
        //        auto rmsbefore = cpy.chi2();
        cout << "chi2 " << cpy.chi2() << endl;
        //        {
        //            SAIGA_BLOCK_TIMER(s->name);
        s->create(cpy);
        auto opt                 = dynamic_cast<Optimizer*>(s.get());
        opt->optimizationOptions = baoptions;
        SAIGA_ASSERT(opt);
        auto result = opt->solve();
        //        }
        cout << "Error " << result.cost_initial << " -> " << result.cost_final << endl;
        cout << "Time LinearSolver/Total: " << result.linear_solver_time << "/" << result.total_time << endl;
        cout << endl;
    }

    return 0;
}
