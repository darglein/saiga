/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/vision/Optimizer.h"

#include "ArapBase.h"
#include "ArapProblem.h"

namespace Saiga
{
class CeresArap : public ArapBase, public Optimizer
{
   public:
    CeresArap() : ArapBase("Ceres") {}
    void optimizeAutodiff(ArapProblem& arap, int its);


    virtual OptimizationResults solve() override;

    void create(ArapProblem& scene) override { _scene = &scene; }


   private:
    ArapProblem* _scene;
};

}  // namespace Saiga
