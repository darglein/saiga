/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/Optimizer.h"
#include "saiga/vision/scene/Scene.h"

namespace Saiga
{
struct SAIGA_VISION_API BAOptions
{
    // Use Huber Cost function if these values are > 0
    float huberMono   = -1;
    float huberStereo = -1;


    void imgui();
};


/**
 * @brief The BABase class
 *
 * Base class and interface for all BA implementations.
 */
class SAIGA_VISION_API BABase
{
   public:
    BABase(const std::string& name) : name(name) {}
    virtual void create(Scene& scene) = 0;

    std::string name;
    BAOptions baOptions;
};


}  // namespace Saiga
