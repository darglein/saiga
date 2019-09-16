/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/framework/framework.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/table.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/ceres/CeresPGO.h"
#include "saiga/vision/g2o/g2oPoseGraph.h"
#include "saiga/vision/recursive/PGORecursive.h"
#include "saiga/vision/scene/BALDataset.h"
#include "saiga/vision/scene/PoseGraph.h"
#include "saiga/vision/scene/SynteticScene.h"

#include <fstream>
using namespace Saiga;


std::vector<std::string> getBALFiles()
{
    std::vector<std::string> files;

    files.insert(files.end(), {"vision/tum_office.scene"});
    files.insert(files.end(), {"vision/tum_large.scene"});
//    files.insert(files.end(), {"dubrovnik-00356-226730.txt"});
#if 0
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

const std::string balPrefix = "vision/bal/";

void test_to_file(const OptimizationOptions& baoptions, const std::string& file, int its)
{
    std::cout << baoptions << std::endl;
    std::cout << "Running long performance test to file..." << std::endl;

    auto files = getBALFiles();


    std::ofstream strm(file);
    strm << "file,edges,poses,density,solver_type,iterations,time_recursive,time_g2o,time_ceres" << std::endl;


    Saiga::Table table({20, 20, 15, 15});

    for (auto file : files)
    {
        PoseGraph scene;

        if (hasEnding(file, ".scene"))
        {
            auto fullFile = file;
            Scene s;
            s.load(fullFile);
            s.normalize();
            scene = PoseGraph(s);
        }
        else
        {
            auto fullFile = SearchPathes::data(balPrefix + file);
            //            buildSceneBAL(scene, fullFile);
            Saiga::BALDataset bald(SearchPathes::data(fullFile));
            scene = PoseGraph(bald.makeScene());
        }

        scene.addNoise(0.55);

        std::vector<std::shared_ptr<PGOBase>> solvers;
        solvers.push_back(std::make_shared<PGORec>());
        solvers.push_back(std::make_shared<g2oPGO>());
        solvers.push_back(std::make_shared<CeresPGO>());



        std::cout << "> Initial Error: " << scene.chi2() << " - " << scene.rms() << std::endl;
        table << "Name"
              << "Final Error"
              << "Time_LS"
              << "Time_Total";

        strm << file << "," << scene.edges.size() << "," << scene.poses.size() << "," << scene.density() << ","
             << (int)baoptions.solverType << "," << (int)baoptions.maxIterations;

        for (auto& s : solvers)
        {
            std::vector<double> times;
            std::vector<double> timesl;
            double chi2;
            for (int i = 0; i < its; ++i)
            {
                auto cpy = scene;
                s->create(cpy);
                auto opt                 = dynamic_cast<Optimizer*>(s.get());
                opt->optimizationOptions = baoptions;
                SAIGA_ASSERT(opt);
                auto result = opt->initAndSolve();
                chi2        = result.cost_final;
                times.push_back(result.total_time);
                timesl.push_back(result.linear_solver_time);
            }


            auto t  = make_statistics(times).median / 1000.0 / baoptions.maxIterations;
            auto tl = make_statistics(timesl).median / 1000.0 / baoptions.maxIterations;
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

    {
        OptimizationOptions baoptions;
        baoptions.debugOutput   = false;
        baoptions.maxIterations = 3;
        //        baoptions.initialLambda = 1;  // use a high lambda for the benchmark so it converges slowly, but
        //        surely
        int testIts = 1;
        if (0)
        {
            baoptions.maxIterativeIterations = 25;
            baoptions.iterativeTolerance     = 1e-50;
            baoptions.solverType             = OptimizationOptions::SolverType::Direct;
            test_to_file(baoptions, "pgo_benchmark_cg.csv", testIts);
        }

        if (0)
        {
            baoptions.solverType = OptimizationOptions::SolverType::Direct;
            test_to_file(baoptions, "pgo_benchmark_chol.csv", testIts);
        }
    }
    //    return 0;

    //    std::string path = "vision/problem-00257-65132-pre.txt";
    //    std::string path = "vision/problem-00356-226730-pre.txt";
    //    std::string path = "vision/problem-00257-65132-pre.txt";
    //    std::string path = "vision/problem-00257-65132-pre.txt";

    //    Saiga::BALDataset bald(SearchPathes::data(path));
    //    Scene scene = bald.makeScene();

    Scene scene;
    //    scene.load(SearchPathes::data("vision/tum_large.scene"));

#if 0
    SynteticScene sscene;
    sscene.numCameras     = 5;
    sscene.numImagePoints = 5;
    sscene.numWorldPoints = 7;
    scene                 = sscene.circleSphere();
#else

    scene.load(SearchPathes::data("vision/tum_office.scene"));
#endif
    //    scene.addWorldPointNoise(0.01);
    //    scene.addImagePointNoise(1.0);
    //    scene.addExtrinsicNoise(0.01);

    std::cout << scene << std::endl;

    //    std::cout << "Density: " << scene.getSchurDensity() << std::endl;
    PoseGraph pg(scene);

    //    pg = PoseGraph();
    //    pg.load(SearchPathes::data("vision/loop.posegraph"));
    std::cout << pg << std::endl;
    //    pg.load(SearchPathes::data("vision/slam_30_431.posegraph"));
    //    pg.load(SearchPathes::data("vision/slam_125_3495.posegraph"));
    //    pg.load(SearchPathes::data("vision/loop.posegraph"));
    //    std::cout << pg.chi2() << std::endl;
    pg.addNoise(1.05);
    pg.fixScale = false;
    //    std::cout << pg.chi2() << std::endl;
    //    std::cout << std::endl;


    OptimizationOptions baoptions;
    baoptions.debugOutput            = false;
    baoptions.maxIterations          = 1;
    baoptions.maxIterativeIterations = 15;
    baoptions.iterativeTolerance     = 1e-50;
    //    baoptions.initialLambda          = 1e3;
    //    baoptions.solverType = OptimizationOptions::SolverType::Direct;
    baoptions.solverType = OptimizationOptions::SolverType::Iterative;
    std::cout << baoptions << std::endl;


    std::vector<std::unique_ptr<PGOBase>> solvers;

    solvers.push_back(std::make_unique<PGORec>());
    //    solvers.push_back(std::make_unique<g2oPGO>());
    solvers.push_back(std::make_unique<CeresPGO>());

    for (auto& s : solvers)
    {
        std::cout << "[Solver] " << s->name << std::endl;
        auto cpy = pg;
        //        auto rmsbefore = cpy.chi2();
        //        std::cout << "chi2 " << cpy.chi2() << std::endl;
        //        {
        //            SAIGA_BLOCK_TIMER(s->name);
        s->create(cpy);
        auto opt                 = dynamic_cast<Optimizer*>(s.get());
        opt->optimizationOptions = baoptions;
        SAIGA_ASSERT(opt);
        auto result = opt->initAndSolve();
        //        }
        std::cout << "Error " << result.cost_initial << " -> " << result.cost_final << std::endl;
        std::cout << "Time LinearSolver/Total: " << result.linear_solver_time << "/" << result.total_time << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
