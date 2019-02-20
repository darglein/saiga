/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/pgo/PGOBase.h"

namespace Saiga
{
class SAIGA_VISION_API CeresPGO : public PGOBase
{
   public:
    CeresPGO() : PGOBase("CeresPGO") {}
    virtual ~CeresPGO() {}
    virtual void solve(PoseGraph& scene, const PGOOptions& options) override;
};

}  // namespace Saiga
