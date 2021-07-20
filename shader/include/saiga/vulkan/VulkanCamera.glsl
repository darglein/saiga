/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "VulkanCamera.h"

layout(set = 5, binding = 7) uniform CameraUniformBufferObject
{
    VulkanCameraData data;
}
camera;
