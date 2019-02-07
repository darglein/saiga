/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/framework/framework.h"
#include "saiga/core/time/Time"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/random.h"
#include "saiga/vision/BALDataset.h"
#include "saiga/vision/Eigen_Compile_Checker.h"
#include "saiga/vision/ba/BAPoseOnly.h"
#include "saiga/vision/ba/BARecursive.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/g2o/g2oBA2.h"
#include "saiga/vision/scene/SynteticScene.h"

#include <fstream>
#ifdef SAIGA_USE_MKL
#    include "saiga/vision/mkl/MKLBA.h"
#endif

using namespace Saiga;


void buildScene(Scene& scene)
{
    SynteticScene sscene;
    sscene.numCameras     = 2;
    sscene.numImagePoints = 2;
    sscene.numWorldPoints = 2;
    scene                 = sscene.circleSphere();
    scene.addWorldPointNoise(0.01);
    scene.addImagePointNoise(1.0);
    scene.addExtrinsicNoise(0.01);
}

void buildSceneBAL(Scene& scene, const std::string& path)
{
    Saiga::BALDataset bald(SearchPathes::data(path));
    //  Saiga::BALDataset bald(SearchPathes::data("vision/problem-00021-11315-pre.txt"));
    //    Saiga::BALDataset bald(SearchPathes::data("vision/problem-00257-65132-pre.txt"));
    //    Saiga::BALDataset bald(SearchPathes::data("vision/problem-01778-993923-pre.txt"));

    scene = bald.makeScene();


    scene.addImagePointNoise(0.1);
    scene.addExtrinsicNoise(0.001);

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
    cout << endl;
}

#define WRITE_TO_FILE


void test_to_file()
{
    cout << "Running long performance test to file..." << endl;
    std::vector<std::string> files = {"vision/problem-00257-65132-pre.txt"};

    BAOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.maxIterations          = 5;
    baoptions.maxIterativeIterations = 20;
    baoptions.iterativeTolerance     = 1e-50;
    baoptions.solverType             = BAOptions::SolverType::Iterative;
    cout << baoptions << endl;


    int its = 9;
    std::ofstream strm("ba_perf.csv");
    strm << "file,solver,time" << endl;

    for (auto file : files)
    {
        Scene scene;
        buildSceneBAL(scene, file);

        std::vector<std::shared_ptr<BABase>> solvers;
        solvers.push_back(std::make_shared<BARec>());
        solvers.push_back(std::make_shared<g2oBA2>());
        solvers.push_back(std::make_shared<CeresBA>());

        for (auto& s : solvers)
        {
            auto stat = Saiga::measureObject(its, [&]() {
                Scene cpy = scene;
                s->solve(cpy, baoptions);
            });

            auto t = stat.median;
            cout << s->name << ": " << t << "ms" << endl;
            strm << file << "," << s->name << "," << t << endl;
        }
    }
}


int main(int, char**)
{
    Saiga::SaigaParameters saigaParameters;
    Saiga::initSample(saigaParameters);
    Saiga::initSaiga(saigaParameters);

    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Saiga::Random::setSeed(93865023985);

    Scene scene;
    //    scene.load(SearchPathes::data("vision/slam_30_2656.scene"));
    //    scene.load(SearchPathes::data("vision/slam_125_8658.scene"));
    //    scene.load(SearchPathes::data("vision/slam.scene"));
    buildScene(scene);



    //    test_to_file();
    //    return 0;

    //    buildSceneBAL(scene, "vision/problem-00257-65132-pre.txt");

    cout << scene << endl;

    BAOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.maxIterations          = 5;
    baoptions.maxIterativeIterations = 20;
    baoptions.iterativeTolerance     = 1e-50;

    //    baoptions.huberMono   = 5.99;
    //    baoptions.huberStereo = 7.815;

    //    baoptions.solverType = BAOptions::SolverType::Direct;
    baoptions.solverType = BAOptions::SolverType::Iterative;
    cout << baoptions << endl;

    std::vector<std::shared_ptr<BABase>> solvers;

    //    solvers.push_back(std::make_shared<BARec>());
    //    solvers.push_back(std::make_shared<BAPoseOnly>());
    //    solvers.push_back(std::make_shared<g2oBA2>());
    solvers.push_back(std::make_shared<CeresBA>());

#if 1
    {
        Scene cpy      = scene;
        auto rmsbefore = cpy.rms();
        {
            BAPoseOnly test;
            test.posePointDense(cpy, 1);
        }
        cout << "Error " << rmsbefore << " -> " << cpy.rms() << endl << endl;
    }
#endif


#ifdef SAIGA_USE_MKL
//    solvers.push_back(std::make_shared<MKLBA>());
#endif
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
