/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/util/Optimizer.h"
#include "saiga/vision/scene/PoseGraph.h"


namespace Saiga
{
/**
 * @brief The BABase class
 *
 * Base class and interface for all BA implementations.
 */
class SAIGA_VISION_API PGOBase
{
   public:
    PGOBase(const std::string& name) : name(name) {}
    virtual ~PGOBase() {}
    virtual void create(PoseGraph& scene) = 0;

    std::string name;
};


}  // namespace Saiga
