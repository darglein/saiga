/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VulkanAsset.h"

#include "saiga/core/image/imageTransformations.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"

namespace Saiga
{
namespace Vulkan
{
void VulkanVertexColoredAsset::init(Saiga::Vulkan::VulkanBase& base)
{
    auto indices = mesh.getIndexList();

    vertexBuffer.init(base, mesh.vertices.size(), vk::MemoryPropertyFlagBits::eDeviceLocal);
    indexBuffer.init(base, indices.size(), vk::MemoryPropertyFlagBits::eDeviceLocal);

    vertexBuffer.stagedUpload(base, mesh.vertices.size() * sizeof(VertexType), mesh.vertices.data());
    indexBuffer.stagedUpload(base, indices.size() * sizeof(uint32_t), indices.data());
}


void VulkanVertexColoredAsset::render(vk::CommandBuffer cmd)
{
    //    if(!vertexBuffer.m_memoryLocation.buffer) return;
    vertexBuffer.bind(cmd);
    indexBuffer.bind(cmd);
    indexBuffer.draw(cmd);
}

void VulkanLineVertexColoredAsset::init(VulkanBase& base)
{
    auto lines   = mesh.toLineList();
    auto newSize = lines.size();
    auto size    = newSize * sizeof(VertexType);
    vertexBuffer.init(base, newSize, vk::MemoryPropertyFlagBits::eDeviceLocal);


    vertexBuffer.stagedUpload(base, size, lines.data());
}

void VulkanLineVertexColoredAsset::render(vk::CommandBuffer cmd)
{
    vertexBuffer.bind(cmd);
    vertexBuffer.draw(cmd);
}



void VulkanPointCloudAsset::init(VulkanBase& base, int _capacity)
{
    capacity = _capacity;
    vertexBuffer.init(base, capacity, vk::MemoryPropertyFlagBits::eDeviceLocal);
    stagingBuffer.init(base, capacity * sizeof(VertexType));
    pointCloud = ArrayView<VertexType>((VertexType*)stagingBuffer.getMappedPointer(), capacity);
}

void VulkanPointCloudAsset::render(vk::CommandBuffer cmd, int start, int count)
{
    vertexBuffer.bind(cmd);
    vertexBuffer.draw(cmd, count < 0 ? size : count, start);
}

void VulkanPointCloudAsset::updateBuffer(vk::CommandBuffer cmd, int start, int count)
{
    stagingBuffer.copyTo(cmd, vertexBuffer, start * sizeof(VertexType), start * sizeof(VertexType),
                         (count < 0 ? size : count) * sizeof(VertexType));
}



void VulkanTexturedAsset::init(VulkanBase& base)
{
    //    vertexBuffer.destroy();
    //    vertexBuffer.init(base,mesh.vertices);

    //    indexBuffer.destroy();
    //    indexBuffer.init(base,mesh.getIndexList());

    auto indices = mesh.getIndexList();

    //    std::cout << mesh.vertices.size() << std::endl;
    vertexBuffer.init(base, mesh.vertices.size(), vk::MemoryPropertyFlagBits::eDeviceLocal);
    indexBuffer.init(base, indices.size(), vk::MemoryPropertyFlagBits::eDeviceLocal);


    vertexBuffer.stagedUpload(base, mesh.vertices.size() * sizeof(VertexType), mesh.vertices.data());
    indexBuffer.stagedUpload(base, indices.size() * sizeof(IndexType), indices.data());

    textures.clear();

    // load textures
    for (auto& tg : groups)
    {
        auto tex = std::make_shared<Texture2D>();

        Saiga::Image img(tg.material.diffuse);

        if (img.type == UC3)
        {
            Saiga::TemplatedImage<ucvec4> img2(img.height, img.width);
            Saiga::ImageTransformation::addAlphaChannel(img.getConstImageView<ucvec3>(), img2.getImageView());
            //            std::cout << img2 << std::endl;
            tex->fromImage(base, img2);
        }
        else
        {
            tex->fromImage(base, img);
        }



        textures.push_back(tex);
    }
}
void VulkanTexturedAsset::render(vk::CommandBuffer cmd)
{
    //    if(!vertexBuffer.buffer) return;
    vertexBuffer.bind(cmd);
    indexBuffer.bind(cmd);
    indexBuffer.draw(cmd);
}


}  // namespace Vulkan
}  // namespace Saiga
