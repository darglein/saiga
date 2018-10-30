/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VulkanAsset.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"
#include "saiga/image/imageTransformations.h"

namespace Saiga {
namespace Vulkan {

void VulkanVertexColoredAsset::init(Saiga::Vulkan::VulkanBase &base)
{
    auto indices = mesh.getIndexList();
#if 0
//    vertexBuffer.init(base,mesh.vertices.size(),vk::MemoryPropertyFlagBits::eDeviceLocal);
    vertexBuffer.init(base,mesh.vertices.size());

    auto indices = mesh.getIndexList();
//    indexBuffer.init(base,indices.size(),vk::MemoryPropertyFlagBits::eDeviceLocal);
    indexBuffer.init(base,indices.size());


    vertexBuffer.mappedUpload(0,mesh.vertices.size()*sizeof(VertexType),mesh.vertices.data());
    indexBuffer.mappedUpload(0,indices.size()*sizeof(uint32_t),indices.data());

//    vertexBuffer.stagedUpload(base,0,mesh.vertices.size()*sizeof(VertexType),mesh.vertices.data());
//    indexBuffer.stagedUpload(base,0,indices.size()*sizeof(uint32_t),indices.data());
#else
        vertexBuffer.init(base,mesh.vertices.size(),vk::MemoryPropertyFlagBits::eDeviceLocal);
        indexBuffer.init(base,indices.size(),vk::MemoryPropertyFlagBits::eDeviceLocal);

    vertexBuffer.stagedUpload(base, mesh.vertices.size() * sizeof(VertexType), mesh.vertices.data());
    indexBuffer.stagedUpload(base, indices.size() * sizeof(uint32_t), indices.data());
#endif
}


void VulkanVertexColoredAsset::render(vk::CommandBuffer cmd)
{
//    if(!vertexBuffer.m_memoryLocation.buffer) return;
    vertexBuffer.bind(cmd);
    indexBuffer.bind(cmd);
    indexBuffer.draw(cmd);
}

void VulkanLineVertexColoredAsset::init(VulkanBase &base, VulkanMemory &memory)
{
    auto lines = mesh.toLineList();
    auto newSize = lines.size();
    auto size = newSize*sizeof(VertexType);
    vertexBuffer.init(base, newSize,vk::MemoryPropertyFlagBits::eDeviceLocal);


    vertexBuffer.stagedUpload(base, size, lines.data());
}

void VulkanLineVertexColoredAsset::render(vk::CommandBuffer cmd)
{
//    if(!vertexBuffer.buffer) return;
    vertexBuffer.bind(cmd);
    vertexBuffer.draw(cmd);
}



void VulkanPointCloudAsset::init(VulkanBase &base, int _capacity)
{
    capacity = _capacity;
    vertexBuffer.init(base,capacity,vk::MemoryPropertyFlagBits::eDeviceLocal);
    stagingBuffer.init(base,capacity * sizeof(VertexType));
    pointCloud = ArrayView<VertexType>( (VertexType*)stagingBuffer.m_memoryLocation.map(base.device),capacity);
}

void VulkanPointCloudAsset::render(vk::CommandBuffer cmd, int start, int count)
{
//    if(!vertexBuffer.buffer || count == 0) return;
    vertexBuffer.bind(cmd);
    vertexBuffer.draw(cmd,count,start);
}

void VulkanPointCloudAsset::updateBuffer(vk::CommandBuffer cmd, int start, int count)
{

    size_t offset = vertexBuffer.m_memoryLocation.offset + start * sizeof(VertexType);
    size_t size = count * sizeof(VertexType);
    vk::BufferCopy bc(offset,offset,size);
    cmd.copyBuffer(stagingBuffer.m_memoryLocation.buffer,vertexBuffer.m_memoryLocation.buffer,bc);
}




void VulkanTexturedAsset::init(VulkanBase &base)
{
//    vertexBuffer.destroy();
//    vertexBuffer.init(base,mesh.vertices);

//    indexBuffer.destroy();
//    indexBuffer.init(base,mesh.getIndexList());

    auto indices = mesh.getIndexList();

    vertexBuffer.init(base,mesh.vertices.size(),vk::MemoryPropertyFlagBits::eDeviceLocal);
    indexBuffer.init(base,indices.size(),vk::MemoryPropertyFlagBits::eDeviceLocal);


    vertexBuffer.stagedUpload(base, mesh.vertices.size() * sizeof(VertexType), mesh.vertices.data());
    indexBuffer.stagedUpload(base, indices.size() * sizeof(IndexType), indices.data());

    textures.clear();

    //load textures
    for(auto& tg : groups)
    {

        auto tex = std::make_shared<Texture2D>();

        Saiga::Image img(tg.material.diffuse);

        if(img.type == UC3)
        {
            Saiga::TemplatedImage<ucvec4> img2(img.height,img.width);
            Saiga::ImageTransformation::addAlphaChannel(img.getConstImageView<ucvec3>(),img2.getImageView());
//            cout << img2 << endl;
            tex->fromImage(base,img2);
        }else{
            tex->fromImage(base,img);
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


}
}
