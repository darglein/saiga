/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL Instance
{
public:
    void destroy();
    void create(std::vector<const char *> instanceExtensions, bool enableValidation);

    operator VkInstance() const { return instance; }

private:
    // Vulkan instance, stores all per-application states
    VkInstance instance;
};


}
}
