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



void test_to_file(const OptimizationOptions& baoptions, const std::string& file, int its)
{
    cout << "Running long performance test to file..." << endl;
#if 1
    std::vector<std::string> files = {
        "problem-16-22106-pre.txt",    "problem-21-11315-pre.txt",    "problem-52-64053-pre.txt",
        "problem-93-61203-pre.txt",    "problem-138-19878-pre.txt",   "problem-138-44033-pre.txt",
        "problem-174-50489-pre.txt",   "problem-202-132796-pre.txt",  "problem-257-65132-pre.txt",
        "problem-356-226730-pre.txt",  "problem-931-102699-pre.txt",  "problem-1102-780462-pre.txt",
        "problem-1723-156502-pre.txt", "problem-1778-993923-pre.txt",
    };  // "problem-3068-310854-pre.txt" "problem-13682-4456117-pre.txt"
#else
    std::vector<std::string> files = {"vision/problem-1723-156502-pre.txt"};
#endif

    std::ofstream strm(file);
    strm << "file,solver,images,points,solver_type,iterations,time secnds,timeLS seconds,rms" << endl;


    Saiga::Table table({20, 20, 10, 10});

    for (auto file : files)
    {
        Scene scene;
        buildSceneBAL(scene, balPrefix + file);

        std::vector<std::shared_ptr<BABase>> solvers;
        solvers.push_back(std::make_shared<BARec>());
        solvers.push_back(std::make_shared<g2oBA2>());
        solvers.push_back(std::make_shared<CeresBA>());


        cout << "> Initial Error: " << scene.chi2() << endl;
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


            auto t  = make_statistics(times).median / 1000.0;
            auto tl = make_statistics(timesl).median / 1000.0;
            table << s->name << chi2 << tl << t;
            strm << file << "," << s->name << "," << scene.images.size() << "," << scene.worldPoints.size() << ","
                 << (int)baoptions.solverType << "," << (int)baoptions.maxIterations << "," << t << "," << tl << ","
                 << chi2 << endl;
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



    if (1)
    {
        OptimizationOptions baoptions;
        baoptions.debugOutput            = false;
        baoptions.maxIterations          = 3;
        baoptions.maxIterativeIterations = 25;
        baoptions.iterativeTolerance     = 1e-50;
        baoptions.initialLambda = 1e10;  // use a high lambda for the benchmark so it converges slowly, but surely
        baoptions.solverType    = OptimizationOptions::SolverType::Iterative;
        cout << baoptions << endl;

        test_to_file(baoptions, "ba_benchmark_cg.csv", 11);
    }
    {
        OptimizationOptions baoptions;
        baoptions.debugOutput   = false;
        baoptions.maxIterations = 3;
        baoptions.solverType    = OptimizationOptions::SolverType::Direct;
        baoptions.initialLambda = 1e10;  // use a high lambda for the benchmark so it converges slowly, but surely
        cout << baoptions << endl;

        test_to_file(baoptions, "ba_benchmark_chol.csv", 11);
    }
    return 0;


    Scene scene;
    //        scene.load(SearchPathes::data("vision/slam_30_2656.scene"));
    //    scene.load(SearchPathes::data("vision/slam_125_8658.scene"));
    //    scene.load(SearchPathes::data("vision/slam.scene"));
    //    buildScene(scene);

    //        buildSceneBAL(scene, balPrefix + "problem-21-11315-pre.txt");
    //    buildSceneBAL(scene, balPrefix + "problem-257-65132-pre.txt");
    buildSceneBAL(scene, balPrefix + "problem-356-226730-pre.txt");


    cout << scene << endl;

    OptimizationOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.maxIterations          = 3;
    baoptions.maxIterativeIterations = 15;
    baoptions.iterativeTolerance     = 1e-50;
    baoptions.solverType             = OptimizationOptions::SolverType::Iterative;
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
