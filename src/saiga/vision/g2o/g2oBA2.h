/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/ba/BABase.h"

namespace Saiga
{
class SAIGA_VISION_API g2oBA2 : public BABase
{
   public:
    g2oBA2() : BABase("g2o BA") {}
    virtual ~g2oBA2() {}
    virtual void solve(Scene& scene, const BAOptions& options) override;
};

}  // namespace Saiga
