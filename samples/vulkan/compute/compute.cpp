/*
* Vulkan Example - imGui (https://github.com/ocornut/imgui)
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "compute.h"
#include <saiga/imgui/imgui.h>
#include "saiga/util/color.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#error OpenGL was included somewhere.
#endif

Compute::Compute(Saiga::Vulkan::VulkanWindow &window, Saiga::Vulkan::VulkanForwardRenderer &renderer)
    :  Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), renderer(renderer)
{
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f,aspect,0.1f,50.0f,true);
    camera.setView(vec3(0,5,10),vec3(0,0,0),vec3(0,1,0));
    camera.rotationPoint = vec3(0);

    window.setCamera(&camera);
}

Compute::~Compute()
{
    teapot.destroy();
    plane.destroy();
    assetRenderer.destroy();
    compute.storageBuffer.destroy();
    computePipeline.destroy();
    compute.commandPool.destroy();

    vkDestroyFence(device,compute.fence,nullptr);
}

void Compute::init(Saiga::Vulkan::VulkanBase &base)
{
    vulkanDevice = &renderer.base;
    device = vulkanDevice->device;



    // create storage buffer
    compute.data.resize(10,1);
    compute.storageBuffer.createBuffer(renderer.base,sizeof(int)*compute.data.size(),vk::BufferUsageFlagBits::eStorageBuffer);
    compute.storageBuffer.allocateMemoryBuffer(renderer.base,vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);

    compute.storageBuffer.mappedUpload(0,sizeof(int)*compute.data.size(),compute.data.data());

    compute.storageBuffer.mappedDownload(0,sizeof(int)*compute.data.size(),compute.data.data());

//    for(int i : compute.data)
//        cout << i << endl;

//    vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);
    compute.queue = device.getQueue(vulkanDevice->queueFamilyIndices.compute,0);


    computePipeline.init(base,1);
    computePipeline.setDescriptorSetLayout({{ 0,vk::DescriptorType::eStorageBuffer,1,vk::ShaderStageFlagBits::eCompute }});


//    computePipeline.createPipelineLayout({
                                             //        vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex,0,sizeof(mat4))
//                                         });



    descriptorSet = computePipeline.createDescriptorSet();

    vk::DescriptorBufferInfo descriptorInfo = compute.storageBuffer.createInfo();
    device.updateDescriptorSets({
                                    vk::WriteDescriptorSet(descriptorSet,0,0,1,vk::DescriptorType::eStorageBuffer,nullptr,&descriptorInfo,nullptr),
                                },nullptr);

    // Load all shaders.
    // Note: The shader type is deduced from the ending.
    computePipeline.shader.load(device,"vulkan/test.comp");



    // We use the default pipeline with "VertexNC" input vertices.
    Saiga::Vulkan::ComputePipelineInfo info;
    computePipeline.preparePipelines(info,base.pipelineCache);


    compute.commandPool.create(device,vulkanDevice->queueFamilyIndices.compute);
    compute.commandBuffer = compute.commandPool.allocateCommandBuffer();


    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
      VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));

      cout << "computing..." << endl;

//      vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline.pipeline);
//      vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);
        vk::CommandBuffer cmd = compute.commandBuffer;

      cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,computePipeline.pipelineLayout,0,descriptorSet,nullptr);
      cmd.bindPipeline(vk::PipelineBindPoint::eCompute,computePipeline.pipeline);

      // Dispatch the compute job
      vkCmdDispatch(compute.commandBuffer,  1, 1, 1);

      vkEndCommandBuffer(compute.commandBuffer);

      VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(0);
      VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &compute.fence));


      vk::SubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
      computeSubmitInfo.commandBufferCount = 1;
      computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
//      VK_CHECK_RESULT( vkQueueSubmit( compute.queue, 1, &computeSubmitInfo, compute.fence ) );
      compute.queue.submit(computeSubmitInfo,compute.fence);
      vkWaitForFences(device, 1, &compute.fence, VK_TRUE, UINT64_MAX);
      vkResetFences(device, 1, &compute.fence);


      compute.storageBuffer.mappedDownload(0,sizeof(int)*compute.data.size(),compute.data.data());

      for(int i : compute.data)
          cout << i << endl;
}



void Compute::update(float dt)
{
    camera.update(dt);
    camera.interpolate(dt,0);
}


void Compute::render(VkCommandBuffer cmd)
{

}

void Compute::renderGUI()
{

}


void Compute::keyPressed(SDL_Keysym key)
{
    switch(key.scancode){
    case SDL_SCANCODE_ESCAPE:
        parentWindow.close();
        break;
    default:
        break;
    }
}

void Compute::keyReleased(SDL_Keysym key)
{
}


