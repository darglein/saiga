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

    void transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout)
    {
        auto& imageLayout                       = layout;
        vk::ImageMemoryBarrier barrier          = {};
        barrier.oldLayout                       = imageLayout;
        barrier.newLayout                       = newLayout;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = image;
        barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        //        barrier.srcAccessMask = 0; // TODO
        //        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite; // TODO


        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (imageLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
        {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage      = vk::PipelineStageFlagBits::eHost;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (imageLayout == vk::ImageLayout::eTransferDstOptimal &&
                 newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage      = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eAllCommands;
        }
        else
        {
            //            throw std::invalid_argument("unsupported layout transition!");
            barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage      = vk::PipelineStageFlagBits::eAllCommands;
            destinationStage = vk::PipelineStageFlagBits::eAllCommands;
        }

        cmd.pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier);

        imageLayout = newLayout;
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
