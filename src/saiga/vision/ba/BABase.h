/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/vision/util/Optimizer.h"

namespace Saiga
{
class Scene;

struct SAIGA_VISION_API BAOptions
{
    // Use Huber Cost function if these values are > 0
    float huberMono   = -1;
    float huberStereo = -1;

    int helper_threads = 1;
    int solver_threads = 1;

    void imgui();
};


/**
 * @brief The BABase class
 *
 * Base class and interface for all BA implementations.
 */
class SAIGA_VISION_API BABase
{
   public:
    BABase(const std::string& name) : name(name) {}
    virtual ~BABase() {}
    virtual void create(Scene& scene) = 0;

    std::string name;
    BAOptions baOptions;
};


constexpr OptimizationOptions defaultBAOptimizationOptions()
{
    OptimizationOptions options;
    options.debugOutput            = false;
    options.maxIterations          = 5;
    options.maxIterativeIterations = 20;
    options.iterativeTolerance     = 1e-10;
    options.solverType             = OptimizationOptions::SolverType::Iterative;
    options.numThreads             = 1;
    options.buildExplizitSchur     = true;
    return options;
}

}  // namespace Saiga
