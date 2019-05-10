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
    DescriptorSetLayout* layout;
    vk::DescriptorSet descriptorSet;
    std::map<uint32_t, std::pair<Texture*, TimePoint>> assigned_textures;
    DescriptorSet() : base(nullptr), layout(nullptr), descriptorSet(nullptr), assigned_textures() {}
    DescriptorSet(VulkanBase* _base, DescriptorSetLayout* _layout, vk::DescriptorSet _descriptorSet)
        : base(_base), layout(_layout), descriptorSet(_descriptorSet), assigned_textures()
    {
    }

   public:
    virtual ~DescriptorSet();

    DescriptorSet(const DescriptorSet& other) = delete;
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
    StaticDescriptorSet(const StaticDescriptorSet& other) = delete;
    StaticDescriptorSet& operator=(const StaticDescriptorSet& other) = delete;

    StaticDescriptorSet(StaticDescriptorSet&& other) = default;
    StaticDescriptorSet& operator=(StaticDescriptorSet&& other) = default;

   public:
    inline void update() override {}
};

class SAIGA_VULKAN_API DynamicDescriptorSet final : public DescriptorSet
{
   private:
    std::pair<vk::DescriptorSet, uint32_t> m_old_set;

   public:
    DynamicDescriptorSet() : DescriptorSet() {}
    DynamicDescriptorSet(VulkanBase* base, DescriptorSetLayout* layout, vk::DescriptorSet descriptor)
        : DescriptorSet(base, layout, descriptor), m_old_set(nullptr, 0xFFFFFFFF)
    {
    }

    DynamicDescriptorSet(const DynamicDescriptorSet& other) = delete;
    DynamicDescriptorSet& operator=(const DynamicDescriptorSet& other) = delete;

    DynamicDescriptorSet(DynamicDescriptorSet&& other) noexcept;
    DynamicDescriptorSet& operator=(DynamicDescriptorSet&& other) noexcept;

    ~DynamicDescriptorSet() final;

    inline void update() override;
};
}  // namespace Saiga::Vulkan
