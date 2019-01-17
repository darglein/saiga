/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/time/timer.h"
#include "saiga/util/random.h"
#include "saiga/vision/BALDataset.h"
#include "saiga/vision/Eigen_Compile_Checker.h"
#include "saiga/vision/ba/BAPoseOnly.h"
#include "saiga/vision/ba/BARecursive.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/g2o/g2oBA2.h"
#include "saiga/vision/scene/SynteticScene.h"
using namespace Saiga;

void buildScene(Scene& scene)
{
    SynteticScene sscene;
    sscene.numCameras     = 3;
    sscene.numImagePoints = 2;
    sscene.numWorldPoints = 5;
    scene                 = sscene.circleSphere();
    scene.addWorldPointNoise(0.01);
    scene.addImagePointNoise(1.0);
    scene.addExtrinsicNoise(0.01);
}

void buildSceneBAL(Scene& scene)
{
    Saiga::BALDataset bald("problem-00021-11315-pre.txt");
    //    Saiga::BALDataset bald("problem-00257-65132-pre.txt");
    //    Saiga::BALDataset bald("problem-01778-993923-pre.txt");
    scene = bald.makeScene();


    //    scene.removeOutliers(2);


    SAIGA_ASSERT(scene);

#if 0
    scene.bf = 1000;
    for (auto& ip : scene.images.front().stereoPoints)
    {
        ip.depth = scene.depth(scene.images.front(), ip) + std::abs(Random::gaussRand(0, 10));
    }

    scene.worldPoints.resize(scene.worldPoints.size() * 50);
#endif
    //    for (int i = 0; i < (int)scene.images.size() / 2; ++i) scene.removeCamera(i);
    //    for (int i = 0; i < scene.worldPoints.size() / 2; ++i) scene.removeWorldPoint(i);
    //        {
    //    }
    SAIGA_ASSERT(scene);
    //    scene.compress();
    //    scene.sortByWorldPointId();
    cout << endl;
}


int main(int, char**)
{
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Saiga::Random::setSeed(93865023985);

    Scene scene;
    scene.load("test.scene");
    //        buildScene(scene);
    //    buildSceneBAL(scene);

    BAOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.maxIterations          = 3;
    baoptions.maxIterativeIterations = 10;
    baoptions.solverType             = BAOptions::SolverType::Iterative;

    std::vector<std::shared_ptr<BABase>> solvers;

    solvers.push_back(std::make_shared<BARec>());
    solvers.push_back(std::make_shared<BAPoseOnly>());
    solvers.push_back(std::make_shared<CeresBA>());
    solvers.push_back(std::make_shared<g2oBA2>());

    for (auto& s : solvers)
    {
        cout << "[Solver] " << s->name << endl;
        Scene cpy      = scene;
        auto rmsbefore = cpy.rms();
        {
            SAIGA_BLOCK_TIMER(s->name);
            s->solve(cpy, baoptions);
        }
        cout << "Error " << rmsbefore << " -> " << cpy.rms() << endl << endl;
    }
    return 0;
}
