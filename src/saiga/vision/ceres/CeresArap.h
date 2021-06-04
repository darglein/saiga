/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/vision/util/Optimizer.h"
#include "saiga/vision/arap/ArapBase.h"
#include "saiga/vision/arap/ArapProblem.h"

namespace Saiga
{
class SAIGA_VISION_API CeresArap : public ArapBase, public Optimizer
{
   public:
    CeresArap() : ArapBase("Ceres") {}
    void optimizeAutodiff(ArapProblem& arap, int its);


    virtual OptimizationResults initAndSolve() override;

    void create(ArapProblem& scene) override { _scene = &scene; }


   private:
    ArapProblem* _scene;
};

}  // namespace Saiga
