/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/scene/PoseGraph.h"

namespace Saiga
{
class SAIGA_GLOBAL g2oPoseGraph
{
   public:
    void solve(PoseGraph& scene);
};

}  // namespace Saiga
