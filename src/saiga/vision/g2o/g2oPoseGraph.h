/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/pgo/PGOBase.h"

namespace Saiga
{
class SAIGA_GLOBAL g2oPGO : public PGOBase
{
   public:
    g2oPGO() : PGOBase("g2oPGO") {}
    virtual ~g2oPGO() {}
    virtual void solve(PoseGraph& scene, const PGOOptions& options) override;
};

}  // namespace Saiga
