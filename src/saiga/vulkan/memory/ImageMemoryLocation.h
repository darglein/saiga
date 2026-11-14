//
// Created by Peter Eichinger on 2019-02-25.
//

#pragma once
#include "MemoryLocation.h"


namespace Saiga::Vulkan::Memory
{
struct SAIGA_VULKAN_API ImageData
{
    vk::ImageLayout layout;
    vk::Image image;
    vk::ImageCreateInfo image_create_info;
    vk::ImageView view;
    vk::ImageViewCreateInfo view_create_info;
    vk::MemoryRequirements image_requirements;
    std::optional<vk::SamplerCreateInfo> sampler_create_info;
    vk::Sampler sampler;

    ImageData(vk::ImageCreateInfo _image_create_info, vk::ImageViewCreateInfo _view_create_info,
              vk::ImageLayout _layout)
        : layout(_layout),
          image(nullptr),
          image_create_info(std::move(_image_create_info)),
          view(nullptr),
          view_create_info(std::move(_view_create_info)),
          image_requirements(),
          sampler_create_info(),
          sampler(nullptr)
    {
    }

    ImageData(vk::ImageCreateInfo _image_create_info, vk::ImageViewCreateInfo _view_create_info,
              vk::ImageLayout _layout, vk::SamplerCreateInfo _sampler_create_info)
        : layout(_layout),
          image(nullptr),
          image_create_info(std::move(_image_create_info)),
          view(nullptr),
          view_create_info(std::move(_view_create_info)),
          image_requirements(),
          sampler_create_info(std::move(_sampler_create_info)),
          sampler(nullptr)
    {
    }

    ImageData(std::nullptr_t)
        : layout(vk::ImageLayout::eUndefined),
          image(nullptr),
          image_create_info(),
          view(nullptr),
          view_create_info(),
          image_requirements(),
          sampler_create_info(),
          sampler(nullptr)
    {
    }

    ImageData(const ImageData& other) = default;

    ImageData(ImageData&& other) noexcept
        : layout(other.layout),
          image(other.image),
          image_create_info(other.image_create_info),
          view(other.view),
          view_create_info(other.view_create_info),
          image_requirements(other.image_requirements),
          sampler_create_info(other.sampler_create_info),
          sampler(other.sampler)
    {
        other.sampler = nullptr;
        other.image   = nullptr;
        other.view    = nullptr;
    }

    ImageData& operator=(const ImageData& other) = default;

    ImageData& operator=(ImageData&& other) noexcept
    {
        if (this != &other)
        {
            this->image               = other.image;
            this->image_create_info   = other.image_create_info;
            this->view                = other.view;
            this->view_create_info    = other.view_create_info;
            this->sampler             = other.sampler;
            this->sampler_create_info = other.sampler_create_info;
            this->image_requirements  = other.image_requirements;
            this->layout              = other.layout;

            other.image   = nullptr;
            other.view    = nullptr;
            other.sampler = nullptr;
        }

        return *this;
    }

    explicit operator bool() const { return image && view; }

    // Compares the handles (the create infos are not compared).
    bool operator==(const ImageData& other) const
    {
        return layout == other.layout && image == other.image && view == other.view && sampler == other.sampler;
    }
    bool operator!=(const ImageData& other) const { return !(*this == other); }

    void copy_create_info_from(ImageData const& other) { set_info(other.image_create_info, other.view_create_info); }

    void set_info(vk::ImageCreateInfo const& _image_create_info, vk::ImageViewCreateInfo const& _view_create_info)
    {
        image_create_info = _image_create_info;
        view_create_info  = _view_create_info;
    }

    void create_image(vk::Device device)
    {
        image                  = device.createImage(image_create_info);
        image_requirements     = device.getImageMemoryRequirements(image);
        view_create_info.image = image;
        layout                 = image_create_info.initialLayout;
    }

    void create_view(vk::Device device) { view = device.createImageView(view_create_info); }

    void create_sampler(vk::Device device)
    {
        if (sampler_create_info)
        {
            sampler = device.createSampler(sampler_create_info.value());
        }
    }
    void destroy_owned_data(vk::Device device)
    {
        if (sampler)
        {
            device.destroy(sampler);
            sampler = nullptr;
        }
        if (view)
        {
            device.destroy(view);
            view = nullptr;
        }

        if (image)
        {
            device.destroy(image);
            image = nullptr;
        }
        layout = vk::ImageLayout::eUndefined;
    }

    friend std::ostream& operator<<(std::ostream& os, const ImageData& data)
    {
        std::stringstream ss;
        ss << std::hex << "L" << vk::to_string(data.layout) << " I" << data.image << " V" << data.view << " S"
           << data.sampler;
        os << ss.str();
        return os;
    }

    inline vk::DescriptorImageInfo get_descriptor_info() const
    {
        SAIGA_ASSERT(view && sampler, "Can't create descriptors for images without view or sampler");
        vk::DescriptorImageInfo descriptorInfo;
        descriptorInfo.imageLayout = layout;
        descriptorInfo.imageView   = view;
        descriptorInfo.sampler     = sampler;
        return descriptorInfo;
    }

    static vk::ImageAspectFlags aspectFlags(vk::Format format)
    {
        switch (format)
        {
            case vk::Format::eD16Unorm:
            case vk::Format::eX8D24UnormPack32:
            case vk::Format::eD32Sfloat:
                return vk::ImageAspectFlagBits::eDepth;
            case vk::Format::eS8Uint:
                return vk::ImageAspectFlagBits::eStencil;
            case vk::Format::eD16UnormS8Uint:
            case vk::Format::eD24UnormS8Uint:
            case vk::Format::eD32SfloatS8Uint:
                return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
            default:
                return vk::ImageAspectFlagBits::eColor;
        }
    }

    /**
     * Records a layout transition of all mip levels and layers from the current layout to newLayout (synchronization2).
     * The stage/access masks are chosen for the layout pair: uploads (undefined/shader read -> transfer dst ->
     * shader read) with a conservative full barrier for all other layouts (e.g. general for storage images).
     */
    void transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout)
    {
        using Stage  = vk::PipelineStageFlagBits2;
        using Access = vk::AccessFlagBits2;

        const auto oldLayout = layout;

        vk::ImageMemoryBarrier2 barrier;
        barrier.oldLayout           = oldLayout;
        barrier.newLayout           = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = image;
        // The layout is tracked per image, so transition all mip levels and layers.
        barrier.subresourceRange = vk::ImageSubresourceRange(aspectFlags(image_create_info.format), 0,
                                                             VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);

        // Accesses before the transition (source scope).
        switch (oldLayout)
        {
            case vk::ImageLayout::eUndefined:
            case vk::ImageLayout::ePreinitialized:
                // Nothing to wait for, the content is discarded (preinitialized: host writes are visible at submit).
                barrier.srcStageMask  = Stage::eNone;
                barrier.srcAccessMask = Access::eNone;
                break;
            case vk::ImageLayout::eTransferDstOptimal:
                barrier.srcStageMask  = Stage::eAllTransfer;
                barrier.srcAccessMask = Access::eTransferWrite;
                break;
            case vk::ImageLayout::eTransferSrcOptimal:
                // Write after read: an execution dependency is enough.
                barrier.srcStageMask  = Stage::eAllTransfer;
                barrier.srcAccessMask = Access::eNone;
                break;
            case vk::ImageLayout::eShaderReadOnlyOptimal:
                barrier.srcStageMask  = Stage::eAllCommands;
                barrier.srcAccessMask = Access::eNone;
                break;
            default:
                // E.g. general (storage image writes, transfers, ...): wait for all writes.
                barrier.srcStageMask  = Stage::eAllCommands;
                barrier.srcAccessMask = Access::eMemoryWrite;
                break;
        }

        // Accesses after the transition (destination scope).
        switch (newLayout)
        {
            case vk::ImageLayout::eTransferDstOptimal:
                barrier.dstStageMask  = Stage::eAllTransfer;
                barrier.dstAccessMask = Access::eTransferWrite;
                break;
            case vk::ImageLayout::eTransferSrcOptimal:
                barrier.dstStageMask  = Stage::eAllTransfer;
                barrier.dstAccessMask = Access::eTransferRead;
                break;
            case vk::ImageLayout::eShaderReadOnlyOptimal:
                barrier.dstStageMask  = Stage::eAllCommands;
                barrier.dstAccessMask = Access::eShaderRead;
                break;
            default:
                // E.g. general (storage image reads/writes, transfers, ...).
                barrier.dstStageMask  = Stage::eAllCommands;
                barrier.dstAccessMask = Access::eMemoryRead | Access::eMemoryWrite;
                break;
        }

        vk::DependencyInfo dependency;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers    = &barrier;
        cmd.pipelineBarrier2(dependency);

        layout = newLayout;
    }
};
using ImageMemoryLocation = BaseMemoryLocation<ImageData>;

inline void bind_image_data(vk::Device device, ImageMemoryLocation* location, ImageData&& data)
{
    SafeAccessor safe(*location);
    location->data = data;
    device.bindImageMemory(location->data.image, location->memory, location->offset);
}

}  // namespace Saiga::Vulkan::Memory
