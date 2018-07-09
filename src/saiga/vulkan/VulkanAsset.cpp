/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VulkanAsset.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"

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
    vertexBuffer.draw(cmd);
}

void VulkanLineVertexColoredAsset::updateBuffer(VulkanBase &base)
{
    vertexBuffer.destroy();
    vertexBuffer.init(base,mesh.toLineList());
}

void VulkanLineVertexColoredAsset::destroy()
{
    vertexBuffer.destroy();
}



void VulkanPointCloudAsset::render(vk::CommandBuffer cmd)
{
    if(!vertexBuffer.buffer) return;
    vertexBuffer.bind(cmd);
    vertexBuffer.draw(cmd);
}

void VulkanPointCloudAsset::updateBuffer(VulkanBase &base)
{
    vertexBuffer.destroy();
    vertexBuffer.init(base,mesh.points);
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
        Saiga::Image img;
        auto res = img.load(tg.material.diffuse);
        SAIGA_ASSERT(res);
        cout << img << endl;
        Texture2D tex;
        tex.fromImage(base,img);
        textures.push_back(tex);
    }
}

void VulkanTexturedAsset::destroy()
{
    vertexBuffer.destroy();
    indexBuffer.destroy();

    for(auto& t : textures)
        t.destroy();
}


}
}
