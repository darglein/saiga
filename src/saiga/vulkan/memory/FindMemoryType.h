//
// Created by Peter Eichinger on 25.10.18.
//

#pragma once

#include <vulkan/vulkan.hpp>

inline uint32_t findMemoryType(vk::PhysicalDevice _pDev, uint32_t typeFilter, const vk::MemoryPropertyFlags& properties)
{
    // TODO: Move this to the appropriate classes and store the value
    vk::PhysicalDeviceMemoryProperties memProperties = _pDev.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}