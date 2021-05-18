/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/ba/BABase.h"

namespace Saiga
{
class SAIGA_VISION_API CeresBARS : public BABase, public Optimizer
{
   public:
    CeresBARS() : BABase("Ceres BA") {}
    virtual ~CeresBARS() {}
    virtual OptimizationResults initAndSolve() override;
    virtual void create(Scene& scene) override { _scene = &scene; }

   private:
    Scene* _scene;
};

}  // namespace Saiga
