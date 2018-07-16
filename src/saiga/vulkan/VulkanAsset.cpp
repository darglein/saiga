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



void VulkanVertexColoredAsset::render(vk::CommandBuffer cmd)
{
    if(!vertexBuffer.buffer) return;
    vertexBuffer.bind(cmd);
    indexBuffer.bind(cmd);
    indexBuffer.draw(cmd);
}

void VulkanVertexColoredAsset::updateBuffer(VulkanBase &base)
{
    vertexBuffer.destroy();
    vertexBuffer.init(base,mesh.vertices);

    indexBuffer.destroy();
    indexBuffer.init(base,mesh.getIndexList());
}

void VulkanVertexColoredAsset::destroy()
{
    vertexBuffer.destroy();
    indexBuffer.destroy();
}

void VulkanLineVertexColoredAsset::render(vk::CommandBuffer cmd)
{
    if(!vertexBuffer.buffer) return;
    vertexBuffer.bind(cmd);
    vertexBuffer.draw(cmd,size);
}

void VulkanLineVertexColoredAsset::updateBuffer(VulkanBase &base)
{
//    vertexBuffer.destroy();
//    vertexBuffer.init(base,mesh.toLineList());

    auto lines = mesh.toLineList();
    auto newSize = lines.size();

    if(newSize > (size_t)capacity)
    {
        capacity = mesh.numLines() * 2;
        vertexBuffer.destroy();
        vertexBuffer.init(base,newSize);
    }

    vertexBuffer.mappedUpload(0,lines.size()*sizeof(VertexType),lines.data());
    size = newSize;
}

void VulkanLineVertexColoredAsset::destroy()
{
    vertexBuffer.destroy();
}



void VulkanPointCloudAsset::render(vk::CommandBuffer cmd)
{
    if(!vertexBuffer.buffer) return;
    vertexBuffer.bind(cmd);
    vertexBuffer.draw(cmd,size);
}

void VulkanPointCloudAsset::updateBuffer(VulkanBase &base)
{
    if(mesh.points.size() > (size_t)capacity)
    {
        capacity = mesh.points.size() * 2;
        vertexBuffer.destroy();
        vertexBuffer.init(base,capacity);
    }
    vertexBuffer.mappedUpload(0,mesh.points.size()*sizeof(VertexType),mesh.points.data());
    size = mesh.points.size();
}

void VulkanPointCloudAsset::destroy()
{
    vertexBuffer.destroy();
}




void VulkanTexturedAsset::render(vk::CommandBuffer cmd)
{
    if(!vertexBuffer.buffer) return;
    vertexBuffer.bind(cmd);
    indexBuffer.bind(cmd);
    indexBuffer.draw(cmd);
}

void VulkanTexturedAsset::updateBuffer(VulkanBase &base)
{
    vertexBuffer.destroy();
    vertexBuffer.init(base,mesh.vertices);

    indexBuffer.destroy();
    indexBuffer.init(base,mesh.getIndexList());

    textures.clear();

    //load textures
    for(auto& tg : groups)
    {

        auto tex = std::make_shared<Texture2D>();

        Saiga::Image img(tg.material.diffuse);

        if(img.type == UC3)
        {
            Saiga::TemplatedImage<ucvec4> img2(img.height,img.width);
            Saiga::ImageTransformation::addAlphaChannel(img.getImageView<ucvec3>(),img2.getImageView());
            cout << img2 << endl;
            tex->fromImage(base,img2);
        }else{
            tex->fromImage(base,img);
        }



        textures.push_back(tex);
    }
}

void VulkanTexturedAsset::destroy()
{
    vertexBuffer.destroy();
    indexBuffer.destroy();
    textures.clear();
}


}
}
