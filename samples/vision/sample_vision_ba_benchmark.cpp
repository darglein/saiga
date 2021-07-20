/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/framework/framework.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/table.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/g2o/g2oBA2.h"
#include "saiga/vision/recursive/BAPointOnly.h"
#include "saiga/vision/recursive/BARecursive.h"
#include "saiga/vision/scene/BALDataset.h"
#include "saiga/vision/scene/SynteticScene.h"

#include <fstream>

const std::string balPrefix = "vision/bal/";

using namespace Saiga;


void buildScene(Scene& scene)
{
    scene = SynteticScene::CircleSphere(3, 3, 3);
    scene.addWorldPointNoise(0.01);
    //    scene.addImagePointNoise(1.0);
    //    scene.addExtrinsicNoise(0.01);
}

std::vector<std::string> getBALFiles()
{
    std::vector<std::string> files;


    files.insert(files.end(), {"vision/tum_office.scene"});
    files.insert(files.end(), {"vision/tum_large.scene"});
//    files.insert(files.end(), {"dubrovnik-00161-103832.txt"});
//    files.insert(files.end(), {"dubrovnik-00356-226730.txt"});
#if 1
    files.insert(files.end(), {"dubrovnik-00016-22106.txt", "dubrovnik-00161-103832.txt"});
    files.insert(files.end(), {"dubrovnik-00262-169354.txt", "dubrovnik-00356-226730.txt"});

    files.insert(files.end(), {"final-00093-61203.txt", "final-00394-100368.txt"});
    files.insert(files.end(), {"final-00961-187103.txt"});
    //    files.insert(files.end(), {"final-04585-1324582.txt", "final-13682-4456117.txt"});

    files.insert(files.end(), {"ladybug-00049-7776.txt", "ladybug-00539-65220.txt"});
    files.insert(files.end(), {"ladybug-00969-105826.txt", "ladybug-01723-156502.txt"});

    files.insert(files.end(), {"trafalgar-00138-44033.txt", "trafalgar-00021-11315.txt"});
    files.insert(files.end(), {"trafalgar-00201-54427.txt", "trafalgar-00257-65132.txt"});

    files.insert(files.end(), {"venice-00052-64053.txt", "venice-01184-816583.txt"});
    files.insert(files.end(), {"venice-01666-983911.txt", "venice-01778-993923.txt"});
#endif

    return files;
}

void buildSceneBAL(Scene& scene, const std::string& path)
{
    Saiga::BALDataset bald(SearchPathes::data(path));
    scene = bald.makeScene();

    Saiga::Random::setSeed(926703466);

    scene.applyErrorToImagePoints();
    scene.addImagePointNoise(0.001);
    //    scene.addExtrinsicNoise(0.0001);
    scene.addWorldPointNoise(0.001);

    auto medianError  = scene.statistics().median;
    scene.globalScale = 1.0 / medianError;
    scene.removeOutliers(10);
    scene.compress();
    std::cout << "> Scene Preprocessing done." << std::endl;

    SAIGA_ASSERT(scene);
}

#define WRITE_TO_FILE


void test_to_file(const OptimizationOptions& baoptions, const std::string& file, int its)
{
    std::cout << baoptions << std::endl;

    std::cout << "Running long performance test to file..." << std::endl;

    auto files = getBALFiles();


    std::ofstream strm(file);
    strm << "file,images,points,schur density,solver_type,iterations,time_recursive,time_g2o,time_ceres" << std::endl;


    Saiga::Table table({20, 20, 15, 15});

    for (auto file : files)
    {
        Scene scene;

        if (hasEnding(file, ".scene"))
        {
            auto fullFile = file;
            scene.load(fullFile);
            scene.normalize();
            scene.addImagePointNoise(0.001);
            scene.addWorldPointNoise(0.001);
        }
        else
        {
            auto fullFile = SearchPathes::data(balPrefix + file);
            buildSceneBAL(scene, fullFile);
        }

        std::vector<std::shared_ptr<BABase>> solvers;
        solvers.push_back(std::make_shared<BARec>());
        solvers.push_back(std::make_shared<g2oBA2>());
        solvers.push_back(std::make_shared<CeresBA>());



        std::cout << "> Initial Error: " << scene.chi2() << " - " << scene.rms() << std::endl;
        table << "Name"
              << "Final Error"
              << "Time_LS"
              << "Time_Total";

        strm << file << "," << scene.images.size() << "," << scene.worldPoints.size() << "," << scene.getSchurDensity()
             << "," << (int)baoptions.solverType << "," << (int)baoptions.maxIterations;

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
                auto result = opt->initAndSolve();
                chi2        = result.cost_final;
                times.push_back(result.total_time);
                timesl.push_back(result.linear_solver_time);
            }


            auto t  = Statistics(times).median / 1000.0 / baoptions.maxIterations;
            auto tl = Statistics(timesl).median / 1000.0 / baoptions.maxIterations;
            table << s->name << chi2 << tl << t;
            strm << "," << t;
        }
        strm << std::endl;
        std::cout << std::endl;
    }
}


int main(int, char**)
{
    initSaigaSampleNoWindow();

    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Saiga::Random::setSeed(93865023985);


#if 0

    {
        OptimizationOptions baoptions;
        baoptions.debugOutput   = false;
        baoptions.maxIterations = 3;
        baoptions.initialLambda = 1000;  // use a high lambda for the benchmark so it converges slowly, but surely
        int testIts             = 11;
        if (1)
        {
            baoptions.maxIterativeIterations = 25;
            baoptions.iterativeTolerance     = 1e-50;
            baoptions.solverType             = OptimizationOptions::SolverType::Iterative;
            test_to_file(baoptions, "ba_benchmark_cg.csv", testIts);
        }
        if (1)
        {
            baoptions.solverType = OptimizationOptions::SolverType::Direct;
            test_to_file(baoptions, "ba_benchmark_chol.csv", testIts);
        }
        return 0;
    }
#endif

    Scene scene;
    //    scene.load(SearchPathes::data("vision/slam_30_2656.scene"));
    //    scene.load(SearchPathes::data("vision/slam_125_8658.scene"));
    //    scene.load(SearchPathes::data("vision/scene_gba.scene"));
    scene.load(SearchPathes::data("vision/broken.scene"));

//    scene.addExtrinsicNoise(0.01);
#if 0
    std::cout << scene << std::endl;

    int maxcams = 20;
    if (maxcams < (int)scene.images.size())
    {
        for (int i = maxcams; i < (int)scene.images.size(); ++i)
        {
            scene.removeCamera(i);
        }
    }
    scene.compress();

    int maxwps = 80;
    if (maxwps < (int)scene.worldPoints.size())
    {
        for (int i = maxwps; i < (int)scene.worldPoints.size(); ++i)
        {
            scene.removeWorldPoint(i);
        }
    }
    scene.compress();

#endif
    //    buildScene(scene);
    //    scene.addExtrinsicNoise(0.01);

    //        buildSceneBAL(scene, balPrefix + "problem-21-11315-pre.txt");
    // buildSceneBAL(scene, balPrefix + "trafalgar-00201-54427.txt");
    //    buildSceneBAL(scene, balPrefix + "ladybug-00049-7776.txt");
    //    buildSceneBAL(scene, balPrefix + "venice-01778-993923.txt");


    std::cout << scene << std::endl;

    OptimizationOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.debug                  = false;
    baoptions.maxIterations          = 3;
    baoptions.maxIterativeIterations = 20;
    baoptions.iterativeTolerance     = 1e-50;
    baoptions.numThreads             = 1;
    baoptions.buildExplizitSchur     = true;



    baoptions.solverType = OptimizationOptions::SolverType::Iterative;
    std::cout << baoptions << std::endl;

    for (int i = 0; i < scene.images.size() / 2; ++i)
    {
        scene.images[i].constant = true;
    }

    std::vector<std::shared_ptr<BABase>> solvers;

    solvers.push_back(std::make_shared<BARec>());
    solvers.push_back(std::make_shared<CeresBA>());
    //    solvers.push_back(std::make_shared<BAPoseOnly>());
    solvers.push_back(std::make_shared<g2oBA2>());

    scene.globalScale = 1;
    for (auto& s : solvers)
    {
        std::cout << "[Solver] " << s->name << std::endl;
        Scene cpy = scene;
        s->create(cpy);
        auto opt                 = dynamic_cast<Optimizer*>(s.get());
        opt->optimizationOptions = baoptions;
        SAIGA_ASSERT(opt);


        OptimizationResults result;
        auto lmopt = dynamic_cast<LMOptimizer*>(opt);
        if (false && lmopt)
        {
            lmopt->initOMP();
            result = lmopt->solveOMP();
        }
        else
        {
            result = opt->initAndSolve();
        }

        std::cout << scene.chi2() << " -> " << cpy.chi2() << std::endl;
        //        auto result = opt->initAndSolve();

        std::cout << "Error " << result.cost_initial << " -> " << result.cost_final << std::endl;
        std::cout << "Time LinearSolver/Total: " << result.linear_solver_time << "/" << result.total_time << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
