//
// Created by Peter Eichinger on 2019-03-04.
//

#pragma once
#include "saiga/core/util/easylogging++.h"
#include "saiga/export.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/buffer/UniformBuffer.h"
#include "saiga/vulkan/memory/BufferMemoryLocation.h"
#include "saiga/vulkan/memory/ImageMemoryLocation.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/texture/Texture.h"

#include <map>
#include <utility>

namespace Saiga::Vulkan
{
class DescriptorSetLayout;
struct VulkanBase;

class SAIGA_VULKAN_API DescriptorSet
{
    using Clock     = Memory::ImageMemoryLocation::Clock;
    using TimePoint = Memory::ImageMemoryLocation::TimePoint;

   protected:
    VulkanBase* base;
    // Points into the PipelineBase that created the set: the set must not be used after the pipeline object is gone.
    // Note: assign() writes the set in place, so it must not be called while a frame in flight uses the set (use a
    // DynamicDescriptorSet for that).
    DescriptorSetLayout* layout;
    vk::DescriptorSet descriptorSet;
    std::map<uint32_t, std::pair<Texture*, TimePoint>> assigned_textures;
    DescriptorSet() : base(nullptr), layout(nullptr), descriptorSet(nullptr), assigned_textures() {}
    DescriptorSet(VulkanBase* _base, DescriptorSetLayout* _layout, vk::DescriptorSet _descriptorSet)
        : base(_base), layout(_layout), descriptorSet(_descriptorSet), assigned_textures()
    {
    }

    // Frees the set once all frames submitted so far (including the one being recorded) have completed.
    void release();

   public:
    virtual ~DescriptorSet();

    DescriptorSet(const DescriptorSet& other)            = delete;
    DescriptorSet& operator=(const DescriptorSet& other) = delete;

    DescriptorSet(DescriptorSet&& other) noexcept;
    DescriptorSet& operator=(DescriptorSet&& other) noexcept;


    inline operator vk::ArrayProxy<const vk::DescriptorSet>() { return descriptorSet; }
    inline operator vk::DescriptorSet() { return descriptorSet; }
    inline operator bool() { return descriptorSet; }
    virtual void update() = 0;

    void assign(uint32_t index, UniformBuffer* buffer);
    void assign(uint32_t index, Texture* texture);
};
class SAIGA_VULKAN_API StaticDescriptorSet final : public DescriptorSet
{
   public:
    ~StaticDescriptorSet() final = default;
    StaticDescriptorSet() : DescriptorSet() {}
    StaticDescriptorSet(VulkanBase* base, DescriptorSetLayout* layout, vk::DescriptorSet descriptor)
        : DescriptorSet(base, layout, descriptor)
    {
    }
    StaticDescriptorSet(const StaticDescriptorSet& other)            = delete;
    StaticDescriptorSet& operator=(const StaticDescriptorSet& other) = delete;

    StaticDescriptorSet(StaticDescriptorSet&& other)            = default;
    StaticDescriptorSet& operator=(StaticDescriptorSet&& other) = default;

   public:
    inline void update() override {}
};

/**
 * A descriptor set that follows textures moved by the defragmenter: update() (call it before the set is bound in a
 * frame) creates a new set with the new image views if an assigned texture was moved, and frees the old set once
 * the frames that may use it have completed.
 */
class SAIGA_VULKAN_API DynamicDescriptorSet final : public DescriptorSet
{
   public:
    DynamicDescriptorSet() : DescriptorSet() {}
    DynamicDescriptorSet(VulkanBase* base, DescriptorSetLayout* layout, vk::DescriptorSet descriptor)
        : DescriptorSet(base, layout, descriptor)
    {
    }

    DynamicDescriptorSet(const DynamicDescriptorSet& other)            = delete;
    DynamicDescriptorSet& operator=(const DynamicDescriptorSet& other) = delete;

    DynamicDescriptorSet(DynamicDescriptorSet&& other) noexcept            = default;
    DynamicDescriptorSet& operator=(DynamicDescriptorSet&& other) noexcept = default;

    ~DynamicDescriptorSet() final = default;

    void update() override;
};
}  // namespace Saiga::Vulkan
