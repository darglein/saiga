/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/pgo/PGOBase.h"

namespace Saiga
{
class SAIGA_VISION_API g2oPGO : public PGOBase, public Optimizer
{
   public:
    g2oPGO() : PGOBase("g2oPGO") {}
    virtual ~g2oPGO() {}

    virtual OptimizationResults initAndSolve() override;
    virtual void create(PoseGraph& scene) override { _scene = &scene; }

   private:
    PoseGraph* _scene;
};

}  // namespace Saiga
