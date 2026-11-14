/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "PointCloudRenderer.h"

#include "saiga/core/model/model_loader_obj.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

#include <locale>
#include <sstream>

namespace Saiga
{
namespace Vulkan
{
void PointCloudRenderer::destroy()
{
    Pipeline::destroy();
    shaderPipeline.destroy();
    uniformBufferVS.destroy();
}
bool PointCloudRenderer::bind(vk::CommandBuffer cmd)
{
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
    return Pipeline::bind(cmd);
}

void PointCloudRenderer::pushModel(VkCommandBuffer cmd, mat4 model)
{
    pushConstant(cmd, vk::ShaderStageFlagBits::eVertex, sizeof(mat4), model.data());
}



void PointCloudRenderer::updateUniformBuffers(vk::CommandBuffer cmd, mat4 view, mat4 proj)
{
    uboVS.projection = proj;
    uboVS.modelview  = view;
    uboVS.lightPos   = vec4(5, 5, 5, 0);
    uniformBufferVS.update(cmd, sizeof(UBOVS), &uboVS);
}

void PointCloudRenderer::init(VulkanBase& vulkanDevice, const RenderingFormats& formats, float pointSize)
{
    PipelineBase::init(vulkanDevice, 1);
    addDescriptorSetLayout({{0, {7, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex}}});
    addPushConstantRange({vk::ShaderStageFlagBits::eVertex, 0, sizeof(mat4)});
    // Not std::to_string, which uses the C locale's decimal separator (e.g. "5,000000").
    std::ostringstream point_size_define;
    point_size_define.imbue(std::locale::classic());
    point_size_define << "#define POINT_SIZE " << pointSize;
    if (!base->enabledFeatures.largePoints)
    {
        // Without the largePoints feature only a point size of 1 is supported.
        point_size_define << "\n#define SAIGA_NO_LARGE_POINTS";
        if (pointSize != 1.0f)
        {
            LOG(WARNING) << "Point size " << pointSize << " requested, large points is not enabled or supported";
        }
    }
    shaderPipeline.loadGLSL(device, {{"vulkan/point.vert", vk::ShaderStageFlagBits::eVertex, point_size_define.str()},
                                     {"vulkan/point.frag", vk::ShaderStageFlagBits::eFragment, ""}});
    PipelineInfo info;
    info.inputAssemblyState.topology = vk::PrimitiveTopology::ePointList;
    info.addVertexInfo<VertexType>();
    create(formats, info);



    uniformBufferVS.init(vulkanDevice, &uboVS, sizeof(UBOVS));
    descriptorSet                           = createDescriptorSet();
    vk::DescriptorBufferInfo descriptorInfo = uniformBufferVS.getDescriptorInfo();
    device.updateDescriptorSets(
        {
            vk::WriteDescriptorSet(descriptorSet, 7, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &descriptorInfo,
                                   nullptr),
        },
        nullptr);
}



}  // namespace Vulkan
}  // namespace Saiga
