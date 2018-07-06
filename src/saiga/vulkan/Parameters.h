/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"

namespace Saiga {
namespace Vulkan {




struct SAIGA_GLOBAL VulkanParameters
{

    bool   enableValidationLayer    = true;
    bool enableImgui = true;


    void fromConfigFile(const std::string& file);
};


}
}
