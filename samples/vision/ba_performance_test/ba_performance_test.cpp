/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/framework/framework.h"
#include "saiga/core/time/Time"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/random.h"
#include "saiga/core/util/table.h"
#include "saiga/vision/BALDataset.h"
#include "saiga/vision/Eigen_Compile_Checker.h"
#include "saiga/vision/ba/BAPoseOnly.h"
#include "saiga/vision/ba/BARecursive.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/g2o/g2oBA2.h"
#include "saiga/vision/scene/SynteticScene.h"

#include <fstream>

const std::string balPrefix = "vision/bal/";

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
    scene = bald.makeScene();
    //  Saiga::BALDataset bald(SearchPathes::data("vision/problem-00021-11315-pre.txt"));
    //    Saiga::BALDataset bald(SearchPathes::data("vision/problem-00257-65132-pre.txt"));
    //    Saiga::BALDataset bald(SearchPathes::data("vision/problem-01778-993923-pre.txt"));


    Saiga::Random::setSeed(926703466);

    scene.addImagePointNoise(0.001);
    scene.addExtrinsicNoise(0.0001);
    scene.addWorldPointNoise(0.001);

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
}

#define WRITE_TO_FILE


void test_to_file()
{
    cout << "Running long performance test to file..." << endl;
#if 1
    std::vector<std::string> files = {
        "problem-16-22106-pre.txt",     "problem-21-11315-pre.txt",    "problem-52-64053-pre.txt",
        "problem-93-61203-pre.txt",     "problem-138-19878-pre.txt",   "problem-138-44033-pre.txt",
        "problem-174-50489-pre.txt",    "problem-202-132796-pre.txt",  "problem-257-65132-pre.txt",
        "problem-356-226730-pre.txt",   "problem-931-102699-pre.txt",  "problem-1102-780462-pre.txt",
        "problem-1723-156502-pre.txt",  "problem-1778-993923-pre.txt", "problem-3068-310854-pre.txt",
        "problem-13682-4456117-pre.txt"};
#else
    std::vector<std::string> files = {"vision/problem-1723-156502-pre.txt"};
#endif

    OptimizationOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.maxIterations          = 1;
    baoptions.maxIterativeIterations = 15;
    baoptions.iterativeTolerance     = 1e-50;
    baoptions.solverType             = OptimizationOptions::SolverType::Direct;
    cout << baoptions << endl;


    int its = 1;
    std::ofstream strm("ba_benchmark.csv");
    strm << "file,solver,solver_type,iterationis,time,timeLS,rms" << endl;


    Saiga::Table table({20, 20, 10, 10});

    for (auto file : files)
    {
        Scene scene;
        buildSceneBAL(scene, balPrefix + file);

        std::vector<std::shared_ptr<BABase>> solvers;
        solvers.push_back(std::make_shared<BARec>());
        solvers.push_back(std::make_shared<g2oBA2>());
        solvers.push_back(std::make_shared<CeresBA>());


        table << "Name"
              << "Error"
              << "Time_LS"
              << "Time_Total";

        for (auto& s : solvers)
        {
            std::vector<double> times;
            std::vector<double> timesl;
            double chi2;
            for (int i = 0; i < its; ++i)
            {
                Scene cpy = scene;
                s->create(cpy);
                auto opt                 = dynamic_cast<Optimizer*>(s.get());
                opt->optimizationOptions = baoptions;
                SAIGA_ASSERT(opt);
                auto result = opt->solve();
                chi2        = result.cost_final;
                times.push_back(result.total_time);
                timesl.push_back(result.linear_solver_time);
            }


            auto t  = make_statistics(times).median;
            auto tl = make_statistics(timesl).median;
            table << s->name << chi2 << tl << t;
            strm << file << "," << s->name << "," << (int)baoptions.solverType << "," << (int)baoptions.maxIterations
                 << "," << t << "," << tl << "," << chi2 << endl;
        }
        cout << endl;
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
    //        scene.load(SearchPathes::data("vision/slam_30_2656.scene"));
    //    scene.load(SearchPathes::data("vision/slam_125_8658.scene"));
    //    scene.load(SearchPathes::data("vision/slam.scene"));
    //    buildScene(scene);



    test_to_file();
    return 0;

    //    buildSceneBAL(scene, balPrefix + "problem-00021-11315-pre.txt");
    buildSceneBAL(scene, balPrefix + "problem-257-65132-pre.txt");
    //    buildSceneBAL(scene, balPrefix + "problem-00356-226730-pre.txt");


    cout << scene << endl;

    OptimizationOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.maxIterations          = 5;
    baoptions.maxIterativeIterations = 10;
    baoptions.iterativeTolerance     = 1e-50;

    //    baoptions.huberMono   = 5.99;
    //    baoptions.huberStereo = 7.815;

    baoptions.solverType = OptimizationOptions::SolverType::Direct;
    //    baoptions.solverType = BAOptions::SolverType::Iterative;
    cout << baoptions << endl;

    std::vector<std::shared_ptr<BABase>> solvers;

    solvers.push_back(std::make_shared<BARec>());
    //    solvers.push_back(std::make_shared<BAPoseOnly>());
    solvers.push_back(std::make_shared<g2oBA2>());
    solvers.push_back(std::make_shared<CeresBA>());

    for (auto& s : solvers)
    {
        cout << "[Solver] " << s->name << endl;
        Scene cpy = scene;
        s->create(cpy);
        auto opt                 = dynamic_cast<Optimizer*>(s.get());
        opt->optimizationOptions = baoptions;
        SAIGA_ASSERT(opt);
        auto result = opt->solve();

        cout << "Error " << result.cost_initial << " -> " << result.cost_final << endl;
        cout << "Time LinearSolver/Total: " << result.linear_solver_time << "/" << result.total_time << endl;
        cout << endl;
    }
    return 0;
}
