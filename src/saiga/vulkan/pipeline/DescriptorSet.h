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
    struct AssignedTexture
    {
        Texture* texture = nullptr;
        // The sampler used instead of the texture's default sampler (not owned), or null.
        vk::Sampler sampler;
        TimePoint time;
    };
    // The assigned textures and buffers with the time they were written (see BaseMemoryLocation::modified_time).
    std::map<uint32_t, AssignedTexture> assigned_textures;
    std::map<uint32_t, std::pair<Buffer*, TimePoint>> assigned_buffers;
    DescriptorSet() : base(nullptr), layout(nullptr), descriptorSet(nullptr) {}
    DescriptorSet(VulkanBase* _base, DescriptorSetLayout* _layout, vk::DescriptorSet _descriptorSet)
        : base(_base), layout(_layout), descriptorSet(_descriptorSet)
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

    // A uniform or storage buffer (the whole buffer).
    void assign(uint32_t index, Buffer* buffer);
    void assign(uint32_t index, Texture* texture);
    // Samples the texture with the given sampler instead of its default sampler. The sampler must outlive the set.
    void assign(uint32_t index, Texture* texture, const Sampler& sampler);
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
 * A descriptor set that follows textures and buffers moved by the defragmentation (see mark_dynamic()): update()
 * (called by PipelineBase::bindDescriptorSet) creates a new set with the new handles if an assigned texture or buffer
 * was moved, and frees the old set once the frames that may use it have completed.
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
