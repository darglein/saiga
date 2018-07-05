/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/svulkan.h"

namespace Saiga {
namespace Vulkan {

/**
 * Helper objects to syncronize rendering with multiple buffers in the swap chain.
 *
 *
 */
class SAIGA_GLOBAL FrameSync
{
public:
    VkSemaphore imageVailable;
    VkSemaphore renderComplete;
    VkFence frameFence;

    void create(VkDevice device);
    void destroy(VkDevice device);

    void wait(VkDevice device);
};

}
}
