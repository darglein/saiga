/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/Scene.h"

namespace Saiga
{
class SAIGA_GLOBAL CeresBA
{
   public:
    void optimize(Scene& scene, int its);
};

}  // namespace Saiga
