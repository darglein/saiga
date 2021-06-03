/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"

#include "Distortion.h"
#include "Intrinsics4.h"

namespace Saiga
{
struct PinholeCamera
{
    IntrinsicsPinholed K;
    Distortion dis;
};

}  // namespace Saiga
