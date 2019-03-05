/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "ArapProblem.h"

namespace Saiga
{
class CeresArap
{
   public:
    void optimize(ArapProblem& arap, int its);
};

}  // namespace Saiga
