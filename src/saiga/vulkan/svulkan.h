/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"

#include "vulkan/vulkan.hpp"

#ifdef SAIGA_ASSERTS
#    define CHECK_VK(_f) SAIGA_ASSERT((_f) == vk::Result::eSuccess)
#else
#    define CHECK_VK(_f) (_f)
#endif

#ifndef SAIGA_USE_VULKAN
#    error Saiga was build without Vulkan.
#endif

#define SAIGA_VULKAN_INCLUDED
