/**
 * Copyright (c) 2021 Darius Rückert
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
#include "saiga/vision/recursive/PGOSim3Recursive.h"
#include "saiga/vision/scene/BALDataset.h"
#include "saiga/vision/scene/PoseGraph.h"
#include "saiga/vision/scene/SynteticPoseGraph.h"
#include "saiga/vision/scene/SynteticScene.h"

#include <fstream>
using namespace Saiga;


int main(int, char**)
{
    EigenHelper::EigenCompileFlags flags;
    flags.create<938476>();
    std::cout << flags << std::endl;

    Saiga::Random::setSeed(93865023985);



    PoseGraph pg = SyntheticPoseGraph::CircleWithDrift(5, 250, 6, 0.01, 0.005);
    std::cout << pg << std::endl;


    OptimizationOptions baoptions;
    baoptions.debugOutput        = false;
    baoptions.maxIterations      = 5;
    baoptions.iterativeTolerance = 1e-50;
    baoptions.minChi2Delta       = 0;
    baoptions.solverType         = OptimizationOptions::SolverType::Direct;
    std::cout << baoptions << std::endl;


    std::vector<std::unique_ptr<PGOBase>> solvers;

    solvers.push_back(std::make_unique<PGORec>());
    solvers.push_back(std::make_unique<PGOSim3Rec>());
    solvers.push_back(std::make_unique<CeresPGO>());

    for (auto& s : solvers)
    {
        std::cout << "[Solver] " << s->name << std::endl;
        auto cpy = pg;
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
