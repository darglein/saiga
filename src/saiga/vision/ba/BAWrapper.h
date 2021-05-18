/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/ba/BABase.h"

#include <memory>
namespace Saiga
{
class SAIGA_VISION_API BAWrapper
{
   public:
    enum class Framework
    {
        Best,
        Recursive,
        Ceres,
        G2O
    };


    BAWrapper(const Framework& fw = Framework::Best);

    void create(Scene& scene);
    OptimizationResults solve(const OptimizationOptions& optimizationOptions, const BAOptions& baOptions);
    OptimizationResults initAndSolve(const OptimizationOptions& optimizationOptions, const BAOptions& baOptions);

   private:
    std::unique_ptr<BABase> ba;
    Optimizer* opt();
    Framework fw;
};



}  // namespace Saiga
