/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/scene/Scene.h"

namespace Saiga
{
struct SAIGA_VISION_API BAOptions
{
    int maxIterations = 10;

    enum class SolverType : int
    {
        Iterative = 0,
        Direct    = 1
    };
    SolverType solverType = SolverType::Iterative;

    int maxIterativeIterations = 50;
    double iterativeTolerance  = 1e-5;

    // Use Huber Cost function if these values are > 0
    float huberMono   = -1;
    float huberStereo = -1;

    bool debugOutput = false;

    void imgui();
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, BAOptions& op);


/**
 * @brief The BABase class
 *
 * Base class and interface for all BA implementations.
 */
class SAIGA_VISION_API BABase
{
   public:
    BABase(const std::string& name) : name(name) {}
    virtual void solve(Scene& scene, const BAOptions& options) = 0;

    std::string name;
};


}  // namespace Saiga
