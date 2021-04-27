/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/ba/BABase.h"

namespace Saiga
{
class SAIGA_VISION_API CeresBA : public BABase, public Optimizer
{
   public:
    CeresBA() : BABase("Ceres BA") {}
    virtual ~CeresBA() {}
    virtual OptimizationResults initAndSolve() override;
    virtual void create(Scene& scene) override { _scene = &scene; }

   private:
    Scene* _scene;
};

}  // namespace Saiga
