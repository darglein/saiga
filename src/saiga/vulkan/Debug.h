#pragma once

#include "saiga/vulkan/svulkan.h"

namespace Saiga {
namespace Vulkan {

// Load debug function pointers and set debug callback
// if callBack is NULL, default message callback will be used
void setupDebugging(
        VkInstance instance,
        VkDebugReportFlagsEXT flags,
        VkDebugReportCallbackEXT callBack
        );

// Clear debug callback
void freeDebugCallback(VkInstance instance);


std::vector<const char*> getDebugValidationLayers();


}
}
