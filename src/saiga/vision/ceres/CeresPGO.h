/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/pgo/PGOBase.h"

namespace Saiga
{
class SAIGA_VISION_API CeresPGO : public PGOBase, public Optimizer
{
   public:
    CeresPGO() : PGOBase("CeresPGO") {}
    virtual ~CeresPGO() {}
    virtual OptimizationResults solve() override;
    virtual void create(PoseGraph& scene) override { _scene = &scene; }

   private:
    PoseGraph* _scene;
};

}  // namespace Saiga
