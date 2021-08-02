#pragma once
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/math/math.h"
#include "saiga/vision/VisionTypes.h"

#include <iostream>
namespace Saiga
{
class SAIGA_VISION_API MetashapeCameraReader
{
   public:
    struct Intrinsics
    {
        std::string name;
        int w, h;
        IntrinsicsPinholed K;
        Distortion dis;
    };

    struct Extrinsics
    {
        std::string name;
        int sensor_id;
        SE3 pose;
        int orientation = 0;
    };

    std::vector<Intrinsics> sensors;
    std::vector<Extrinsics> cameras;


    MetashapeCameraReader(const std::string& file, bool verbose = false);
};


}  // namespace Saiga