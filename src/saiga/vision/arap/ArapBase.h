/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/vision/util/Optimizer.h"

#include "ArapProblem.h"

namespace Saiga
{
class SAIGA_VISION_API ArapBase
{
   public:
    ArapBase(const std::string& name) : name(name) {}
    virtual ~ArapBase() {}
    virtual void create(ArapProblem& scene) = 0;

    std::string name;
};

}  // namespace Saiga
