//
// Created by Peter Eichinger on 06.11.18.
//

#pragma once

#include "saiga/export.h"
#include "saiga/util/assert.h"

#include "vulkan/vulkan.hpp"

struct SAIGA_GLOBAL AsyncCommand
{
    vk::CommandBuffer cmd;
    vk::Fence fence;

    AsyncCommand(vk::CommandBuffer _cmd, vk::Fence _fence) : cmd(_cmd), fence(_fence)
    {
        SAIGA_ASSERT(cmd, "Invalid command for async");
        SAIGA_ASSERT(fence, "Invalid fence for async");
    }
};