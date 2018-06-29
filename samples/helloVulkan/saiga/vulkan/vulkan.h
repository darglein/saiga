/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/config.h"
#include "saiga/util/glm.h"
#include "saiga/util/assert.h"

#define VK_USE_PLATFORM_XCB_KHR
#include "vulkan/vulkan.hpp"
#include "xcb/xcb.h"


#ifdef SAIGA_ASSERTS
#define CHECK_VK(_f) SAIGA_ASSERT((_f) == vk::Result::eSuccess)
#else
#define CHECK_VK(_f) (_f)
#endif
