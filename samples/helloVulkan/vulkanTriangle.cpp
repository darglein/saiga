/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkanTriangle.h"
#include "saiga/util/assert.h"
#include <chrono>
#include <thread>
namespace Saiga {


VulkanWindow::VulkanWindow()
    : Application(500,500)
{


    // ======================= Layers =======================



    vk::Result res;










    uniformBuffer.init(*this);






    // vertex buffer

    //    init_vertex_buffer();
    vertexBuffer.init(*this);
    indexBuffer.init(*this);


    // pipeline layout

    {
        vk::DescriptorSetLayoutBinding layout_binding = {};
        layout_binding.binding = 0;
        layout_binding.descriptorType = vk::DescriptorType::eUniformBuffer;
        layout_binding.descriptorCount = 1;
        layout_binding.stageFlags = vk::ShaderStageFlagBits::eVertex;
        layout_binding.pImmutableSamplers = NULL;

        /* Next take layout bindings and use them to create a descriptor set layout
         */
        vk::DescriptorSetLayoutCreateInfo descriptor_layout = {};
        //        descriptor_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        //        descriptor_layout.pNext = NULL;
        descriptor_layout.bindingCount = 1;
        descriptor_layout.pBindings = &layout_binding;

        /* Number of descriptor sets needs to be the same at alloc,       */
        /* pipeline layout creation, and descriptor set layout creation   */
#define NUM_DESCRIPTOR_SETS 1


        desc_layout.resize(NUM_DESCRIPTOR_SETS);
        //        res = vkCreateDescriptorSetLayout(info.device, &descriptor_layout, NULL, info.desc_layout.data());
        res = device.createDescriptorSetLayout(&descriptor_layout,nullptr,desc_layout.data());
        SAIGA_ASSERT(res == vk::Result::eSuccess);


        /* Now use the descriptor layout to create a pipeline layout */
        vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo = {};
        //        pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        //        pPipelineLayoutCreateInfo.pNext = NULL;
        pPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
        pPipelineLayoutCreateInfo.pPushConstantRanges = NULL;
        pPipelineLayoutCreateInfo.setLayoutCount = NUM_DESCRIPTOR_SETS;
        pPipelineLayoutCreateInfo.pSetLayouts = desc_layout.data();

        res = device.createPipelineLayout(&pPipelineLayoutCreateInfo, NULL, &pipeline_layout);
        SAIGA_ASSERT(res == vk::Result::eSuccess);
        //        assert(res == VK_SUCCESS);
    }



    // descriptor set

    {
        vk::DescriptorPoolSize type_count[1];
        type_count[0].type = vk::DescriptorType::eUniformBuffer;
        type_count[0].descriptorCount = 1;

        vk::DescriptorPoolCreateInfo descriptor_pool = {};
        //        descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO vk::;
        //        descriptor_pool.pNext = NULL;
        descriptor_pool.maxSets = 1;
        descriptor_pool.poolSizeCount = 1;
        descriptor_pool.pPoolSizes = type_count;

        //        res = vkCreateDescriptorPool(info.device, &descriptor_pool, NULL, &info.desc_pool);
        res = device.createDescriptorPool(&descriptor_pool, NULL, &desc_pool);
        SAIGA_ASSERT(res == vk::Result::eSuccess);


        vk::DescriptorSetAllocateInfo alloc_info[1];
        //           alloc_info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        //           alloc_info[0].pNext = NULL;
        alloc_info[0].descriptorPool = desc_pool;
        alloc_info[0].descriptorSetCount = NUM_DESCRIPTOR_SETS;
        alloc_info[0].pSetLayouts = desc_layout.data();

        desc_set.resize(NUM_DESCRIPTOR_SETS);
        //           res = vkAllocateDescriptorSets(info.device, alloc_info, info.desc_set.data());
        res = device.allocateDescriptorSets(alloc_info, desc_set.data());
        SAIGA_ASSERT(res == vk::Result::eSuccess);


        vk::WriteDescriptorSet writes[1];

        //           writes[0] = {};
        //           writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        //           writes[0].pNext = NULL;
        writes[0].dstSet = desc_set[0];
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        writes[0].pBufferInfo = &uniformBuffer.info;
        writes[0].dstArrayElement = 0;
        writes[0].dstBinding = 0;

        //           vkUpdateDescriptorSets(info.device, 1, writes, 0, NULL);
        device.updateDescriptorSets(1, writes, 0, NULL);

    }



    shader.init(*this);



    //pipeline
    {
        vk::DynamicState dynamicStateEnables[VK_DYNAMIC_STATE_RANGE_SIZE ];
        vk::PipelineDynamicStateCreateInfo dynamicState = {};
        memset(dynamicStateEnables, 0, sizeof dynamicStateEnables);
        //    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        //    dynamicState.pNext = NULL;
        dynamicState.pDynamicStates = dynamicStateEnables;
        dynamicState.dynamicStateCount = 0;

        vk::PipelineVertexInputStateCreateInfo vi;
        //    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        //    vi.pNext = NULL;
        //    vi.flags = 0;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &vertexBuffer.vi_binding;
        vi.vertexAttributeDescriptionCount = vertexBuffer.vi_attribs.size();
        vi.pVertexAttributeDescriptions = vertexBuffer.vi_attribs.data();

        vk::PipelineInputAssemblyStateCreateInfo ia;
        //    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        //    ia.pNext = NULL;
        //    ia.flags = 0;
        ia.primitiveRestartEnable = VK_FALSE;
        ia.topology = vk::PrimitiveTopology::eTriangleList;

        vk::PipelineRasterizationStateCreateInfo rs;
        //    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        //    rs.pNext = NULL;
        //    rs.flags = 0;
        rs.polygonMode = vk::PolygonMode::eFill;
        rs.cullMode = vk::CullModeFlagBits::eBack;
        rs.frontFace = vk::FrontFace::eClockwise;
        rs.depthClampEnable = VK_FALSE;
        rs.rasterizerDiscardEnable = VK_FALSE;
        rs.depthBiasEnable = VK_FALSE;
        rs.depthBiasConstantFactor = 0;
        rs.depthBiasClamp = 0;
        rs.depthBiasSlopeFactor = 0;
        rs.lineWidth = 1.0f;


        vk::PipelineColorBlendStateCreateInfo cb;
        //    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        //    cb.pNext = NULL;
        //    cb.flags = 0;
        vk::PipelineColorBlendAttachmentState att_state[1];
        att_state[0].colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB| vk::ColorComponentFlagBits::eA;
        att_state[0].blendEnable = VK_FALSE;
        att_state[0].alphaBlendOp = vk::BlendOp::eAdd;
        att_state[0].colorBlendOp = vk::BlendOp::eAdd;
        att_state[0].srcColorBlendFactor = vk::BlendFactor::eZero;
        att_state[0].dstColorBlendFactor = vk::BlendFactor::eZero;
        att_state[0].srcAlphaBlendFactor = vk::BlendFactor::eZero;
        att_state[0].dstAlphaBlendFactor = vk::BlendFactor::eZero;
        cb.attachmentCount = 1;
        cb.pAttachments = att_state;
        cb.logicOpEnable = VK_FALSE;
        cb.logicOp = vk::LogicOp::eNoOp;
        cb.blendConstants[0] = 1.0f;
        cb.blendConstants[1] = 1.0f;
        cb.blendConstants[2] = 1.0f;
        cb.blendConstants[3] = 1.0f;

        vk::PipelineViewportStateCreateInfo vp = {};
        //    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        //    vp.pNext = NULL;
        //    vp.flags = 0;
        vp.viewportCount = 1;
        dynamicStateEnables[dynamicState.dynamicStateCount++] = vk::DynamicState::eViewport;
        vp.scissorCount = 1;
        dynamicStateEnables[dynamicState.dynamicStateCount++] = vk::DynamicState::eScissor;
        vp.pScissors = NULL;
        vp.pViewports = NULL;

        vk::PipelineDepthStencilStateCreateInfo ds;
        //    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        //    ds.pNext = NULL;
        //    ds.flags = 0;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = vk::CompareOp::eLessOrEqual;
        ds.depthBoundsTestEnable = VK_FALSE;
        ds.minDepthBounds = 0;
        ds.maxDepthBounds = 0;
        ds.stencilTestEnable = VK_FALSE;
        ds.back.failOp = vk::StencilOp::eKeep;
        ds.back.passOp = vk::StencilOp::eKeep;
        ds.back.compareOp = vk::CompareOp::eAlways;
        ds.back.compareMask = 0;
        ds.back.reference = 0;
        ds.back.depthFailOp = vk::StencilOp::eKeep;
        ds.back.writeMask = 0;
        ds.front = ds.back;


        vk::PipelineMultisampleStateCreateInfo ms;
        //    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        //    ms.pNext = NULL;
        //    ms.flags = 0;
        ms.pSampleMask = NULL;
        ms.rasterizationSamples = vk::SampleCountFlagBits::e1;
        ms.sampleShadingEnable = VK_FALSE;
        ms.alphaToCoverageEnable = VK_FALSE;
        ms.alphaToOneEnable = VK_FALSE;
        ms.minSampleShading = 0.0;

        vk::GraphicsPipelineCreateInfo pipeline_info;
        //    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        //    pipeline.pNext = NULL;
        pipeline_info.layout = pipeline_layout;
        //    pipeli_infone.basePipelineHandle = VK_NULL_HANDLE;
        pipeline_info.basePipelineIndex = 0;
        //    pipeli_infone.flags = 0;
        pipeline_info.pVertexInputState = &vi;
        pipeline_info.pInputAssemblyState = &ia;
        pipeline_info.pRasterizationState = &rs;
        pipeline_info.pColorBlendState = &cb;
        pipeline_info.pTessellationState = NULL;
        pipeline_info.pMultisampleState = &ms;
        pipeline_info.pDynamicState = &dynamicState;
        pipeline_info.pViewportState = &vp;
        pipeline_info.pDepthStencilState = &ds;
        pipeline_info.pStages = shader.shaderStages;
        pipeline_info.stageCount = 2;
        pipeline_info.renderPass = forwardRenderer.render_pass;
        pipeline_info.subpass = 0;

        vk::PipelineCacheCreateInfo pipelineCacheInfo;
        //    pipelineCache.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        //    pipelineCache.pNext = NULL;
        pipelineCacheInfo.initialDataSize = 0;
        pipelineCacheInfo.pInitialData = NULL;
        //    pipelineCache.flags = 0;
        res = device.createPipelineCache(&pipelineCacheInfo, NULL, &pipelineCache);


        res = device.createGraphicsPipelines(pipelineCache, 1, &pipeline_info, NULL, &pipeline);
        SAIGA_ASSERT(res == vk::Result::eSuccess);
    }


}


VulkanWindow::~VulkanWindow()
{


    device.destroy();
    inst.destroy();
}

void VulkanWindow::update()
{

}

void VulkanWindow::render(vk::CommandBuffer &cmd)
{


    vk::ClearValue clear_values[2];
//    float c = float(i) / count;
    float c = 0.5f;
    clear_values[0].color.float32[0] = c;
    clear_values[0].color.float32[1] = c;
    clear_values[0].color.float32[2] = c;
    clear_values[0].color.float32[3] = c;
    clear_values[1].depthStencil.depth = 1.0f;
    clear_values[1].depthStencil.stencil = 0;

    vk::Semaphore imageAcquiredSemaphore;
    vk::SemaphoreCreateInfo imageAcquiredSemaphoreCreateInfo;
    //        imageAcquiredSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    //        imageAcquiredSemaphoreCreateInfo.pNext = NULL;
    //        imageAcquiredSemaphoreCreateInfo.flags = 0;

    CHECK_VK(device.createSemaphore(&imageAcquiredSemaphoreCreateInfo, NULL, &imageAcquiredSemaphore));


    // Get the index of the next available swapchain image:
    //        current_buffer = device.acquireNextImageKHR(swap_chain, UINT64_MAX, imageAcquiredSemaphore, vk::Fence()).value;
    swapChain->acquireNextImage(imageAcquiredSemaphore,current_buffer);


    vk::RenderPassBeginInfo rp_begin;
    //        rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    //        rp_begin.pNext = NULL;
    rp_begin.renderPass = forwardRenderer.render_pass;
    rp_begin.framebuffer = forwardRenderer.framebuffers[current_buffer];
    rp_begin.renderArea.offset.x = 0;
    rp_begin.renderArea.offset.y = 0;
    rp_begin.renderArea.extent.width = window.width;
    rp_begin.renderArea.extent.height = window.height;
    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clear_values;

    cmd.beginRenderPass(&rp_begin, vk::SubpassContents::eInline);



    cmd.bindPipeline( vk::PipelineBindPoint::eGraphics, pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, NUM_DESCRIPTOR_SETS,
                           desc_set.data(), 0, NULL);

    const vk::DeviceSize offsets[1] = {0};
    cmd.bindVertexBuffers(0, 1, &vertexBuffer.buffer, offsets);
    cmd.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);



    //        init_viewports(info);
    //        init_scissors(info);


    //        cmd.draw( 3, 1, 0, 0);
    cmd.drawIndexed( 3, 1, 0, 0,1);
    //        vkCmdEndRenderPass(info.cmd);
    cmd.endRenderPass();
    //        res = vkEndCommandBuffer(info.cmd);
    cmd.end();


    const vk::CommandBuffer cmd_bufs[] = {cmd};
    vk::FenceCreateInfo fenceInfo;
    vk::Fence drawFence;
    //        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    //        fenceInfo.pNext = NULL;
    //        fenceInfo.flags = 0;
    device.createFence(&fenceInfo, NULL, &drawFence);



    vk::PipelineStageFlags pipe_stage_flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submit_info[1] = {};
    //        submit_info[0].pNext = NULL;
    //        submit_info[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info[0].waitSemaphoreCount = 1;
    submit_info[0].pWaitSemaphores = &imageAcquiredSemaphore;
    submit_info[0].pWaitDstStageMask = &pipe_stage_flags;
    submit_info[0].commandBufferCount = 1;
    submit_info[0].pCommandBuffers = cmd_bufs;
    submit_info[0].signalSemaphoreCount = 0;
    submit_info[0].pSignalSemaphores = NULL;

    /* Queue the command buffer for execution */

    CHECK_VK(queue.submit( 1, submit_info, drawFence));
    //        res = vkQueueSubmit(info.graphics_queue,


    vk::PresentInfoKHR present;
    //        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    //        present.pNext = NULL;
    present.swapchainCount = 1;
    present.pSwapchains = &swapChain->swapChain;
    present.pImageIndices = &current_buffer;
    present.pWaitSemaphores = NULL;
    present.waitSemaphoreCount = 0;
    present.pResults = NULL;


    /* Make sure command buffer is finished before presenting */
    vk::Result res;
    do {
        res = device.waitForFences(1, &drawFence, VK_TRUE, 1241515);
    } while (res == vk::Result::eTimeout);


    queue.presentKHR(&present);


    std::this_thread::sleep_for(std::chrono::milliseconds(16));
}



}
