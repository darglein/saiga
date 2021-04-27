/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once
#include "saiga/core/geometry/LineMesh.h"
#include "saiga/core/geometry/PointCloud.h"
#include "saiga/core/model/all.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/VulkanBuffer.hpp"
#include "saiga/vulkan/buffer/IndexBuffer.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"
#include "saiga/vulkan/buffer/VertexBuffer.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/texture/Texture.h"

#include "pipeline/DescriptorSet.h"
namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API VulkanVertexColoredAsset : public TriangleMesh<VertexNC, uint32_t>
{
   public:
    VertexBuffer<VertexType> vertexBuffer;
    IndexBuffer<uint32_t> indexBuffer;


    void init(Saiga::Vulkan::VulkanBase& base);
    void render(vk::CommandBuffer cmd);

    //    void updateBuffer(Saiga::Vulkan::VulkanBase& base);
};


class SAIGA_VULKAN_API VulkanLineVertexColoredAsset
{
   public:
    using IndexType  = uint32_t;
    using VertexType = VertexNC;
    LineMesh<VertexType, IndexType> mesh;

    VertexBuffer<VertexType> vertexBuffer;

    void init(VulkanBase& base);
    void render(vk::CommandBuffer cmd);
};



class SAIGA_VULKAN_API VulkanPointCloudAsset
{
   public:
    using VertexType = VertexNC;

    ArrayView<VertexType> pointCloud;

    VertexBuffer<VertexType> vertexBuffer;
    StagingBuffer stagingBuffer;

    int size     = 0;
    int capacity = 0;

    // Creates the buffers with max number of points
    void init(VulkanBase& base, int capacity);

    void render(vk::CommandBuffer cmd, int start = 0, int count = -1);
    void updateBuffer(vk::CommandBuffer cmd, int start = 0, int count = -1);
};



}  // namespace Vulkan
}  // namespace Saiga
