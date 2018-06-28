/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkanWindow.h"
#include "saiga/util/assert.h"
#include <chrono>
#include <thread>
namespace Saiga {


VulkanWindow::VulkanWindow()
{


    // ======================= Layers =======================

    window.createWindow(500,500);

    vk::Result res;


    //    init_global_layer_properties();
    //    init_instance();

    createInstance(true);

    init_physical_device();

    createDevice();

    auto surface = window.createSurfaceKHR(inst);


    swapChain = new Vulkan::SwapChain(inst, physicalDevice, device);
    swapChain->setSurface(surface);
    swapChain->create(&window.width, &window.height);


    // ======================= Command Buffer =======================

    {
        vk::CommandPoolCreateInfo cmd_pool_info = {};
        cmd_pool_info.queueFamilyIndex = swapChain->queueNodeIndex;

        res = device.createCommandPool(&cmd_pool_info, nullptr, &cmd_pool);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

        vk::CommandBufferAllocateInfo cmd_info = {};
        cmd_info.commandPool = cmd_pool;
        cmd_info.level = vk::CommandBufferLevel::ePrimary;
        cmd_info.commandBufferCount = 1;

        res = device.allocateCommandBuffers(&cmd_info,&cmd);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

    }


    cmd.begin( vk::CommandBufferBeginInfo() );


//    init_depth_buffer();
    depthBuffer.init(*this,window.width,window.height);


//    init_uniform_buffer();
    uniformBuffer.init(*this);


    // init render path

    {


        // The initial layout for the color and depth attachments will be
        // LAYOUT_UNDEFINED because at the start of the renderpass, we don't
        // care about their contents. At the start of the subpass, the color
        // attachment's layout will be transitioned to LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        // and the depth stencil attachment's layout will be transitioned to
        // LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL.  At the end of the renderpass,
        // the color attachment's layout will be transitioned to
        // LAYOUT_PRESENT_SRC_KHR to be ready to present.  This is all done as part
        // of the renderpass, no barriers are necessary.
        vk::AttachmentDescription attachments[2];
        attachments[0].format = swapChain->colorFormat;
        attachments[0].samples = vk::SampleCountFlagBits::e1;
        attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
        attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
        attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eStore;
        attachments[0].initialLayout = vk::ImageLayout::eUndefined;
        attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;
        //        attachments[0].flags = 0;

        attachments[1].format = depthBuffer.depthFormat;
        attachments[1].samples = vk::SampleCountFlagBits::e1;
        attachments[1].loadOp = vk::AttachmentLoadOp::eClear;
        attachments[1].storeOp = vk::AttachmentStoreOp::eDontCare;
        attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachments[1].initialLayout = vk::ImageLayout::eUndefined;
        attachments[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        //        attachments[1].samples = NUM_SAMPLES;
        //        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        //        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        //        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        //        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        //        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        //        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        //        attachments[1].flags = 0;


        vk::AttachmentReference color_reference = {};
        color_reference.attachment = 0;
        color_reference.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::AttachmentReference depth_reference = {};
        depth_reference.attachment = 1;
        depth_reference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::SubpassDescription subpass = {};
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        //        subpass.flags = 0;
        subpass.inputAttachmentCount = 0;
        subpass.pInputAttachments = NULL;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_reference;
        subpass.pResolveAttachments = NULL;
        subpass.pDepthStencilAttachment = &depth_reference;
        subpass.preserveAttachmentCount = 0;
        subpass.pPreserveAttachments = NULL;

        vk::RenderPassCreateInfo rp_info = {};
        //        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO vk::st;
        //        rp_info.pNext = NULL;
        rp_info.attachmentCount = 2;
        rp_info.pAttachments = attachments;
        rp_info.subpassCount = 1;
        rp_info.pSubpasses = &subpass;
        rp_info.dependencyCount = 0;
        rp_info.pDependencies = NULL;

        //        res = vkCreateRenderPass(info.device, &rp_info, NULL, &info.render_pass);
        res = device.createRenderPass(&rp_info, NULL, &render_pass);
        SAIGA_ASSERT(res == vk::Result::eSuccess);



    }

    // init framebuffers

    {
        vk::ImageView attachments[2];
        attachments[1] = depthBuffer.depthview;

        vk::FramebufferCreateInfo fb_info = {};
        //        fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        //        fb_info.pNext = NULL;
        fb_info.renderPass = render_pass;
        fb_info.attachmentCount = 2;
        fb_info.pAttachments = attachments;
        fb_info.width = window.width;
        fb_info.height = window.height;
        fb_info.layers = 1;

        uint32_t i;
        //        info.framebuffers = (VkFramebuffer *)malloc(info.swapchainImageCount * sizeof(VkFramebuffer));
        framebuffers.resize( swapChain->buffers.size());
        SAIGA_ASSERT(swapChain->buffers.size() >= 1);
        //        assert(info.framebuffers);

        for (i = 0; i < framebuffers.size(); i++)
        {
            //            attachments[0] = swapChainImagesViews[i];
            attachments[0] = swapChain->buffers[i].view;
            res = device.createFramebuffer(&fb_info, NULL, &framebuffers[i]);
            SAIGA_ASSERT(res == vk::Result::eSuccess);
            //            assert(res == VK_SUCCESS);
        }


    }


    // vertex buffer

//    init_vertex_buffer();
    vertexBuffer.init(*this);

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
        writes[0].pBufferInfo = &uniformBuffer.uniformbuffer_info;
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
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions = vertexBuffer.vi_attribs;

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
        pipeline_info.renderPass = render_pass;
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



    // draw

    {
        vk::ClearValue clear_values[2];
        clear_values[0].color.float32[0] = 0.2f;
        clear_values[0].color.float32[1] = 0.2f;
        clear_values[0].color.float32[2] = 0.2f;
        clear_values[0].color.float32[3] = 0.2f;
        clear_values[1].depthStencil.depth = 1.0f;
        clear_values[1].depthStencil.stencil = 0;

        vk::Semaphore imageAcquiredSemaphore;
        vk::SemaphoreCreateInfo imageAcquiredSemaphoreCreateInfo;
        //        imageAcquiredSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        //        imageAcquiredSemaphoreCreateInfo.pNext = NULL;
        //        imageAcquiredSemaphoreCreateInfo.flags = 0;

        res = device.createSemaphore(&imageAcquiredSemaphoreCreateInfo, NULL, &imageAcquiredSemaphore);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

        // Get the index of the next available swapchain image:
        //        current_buffer = device.acquireNextImageKHR(swap_chain, UINT64_MAX, imageAcquiredSemaphore, vk::Fence()).value;
        swapChain->acquireNextImage(imageAcquiredSemaphore,current_buffer);


        vk::RenderPassBeginInfo rp_begin;
        //        rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        //        rp_begin.pNext = NULL;
        rp_begin.renderPass = render_pass;
        rp_begin.framebuffer = framebuffers[current_buffer];
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
        cmd.bindVertexBuffers(0, 1, &vertexBuffer.vertexbuf, offsets);


        vk::Viewport viewport;
        vk::Rect2D scissor;
        viewport.height = (float)window.height;
        viewport.width = (float)window.width;
        viewport.minDepth = (float)0.0f;
        viewport.maxDepth = (float)1.0f;
        viewport.x = 0;
        viewport.y = 0;
        cmd.setViewport(0, 1, &viewport);

        scissor.extent.width = window.width;
        scissor.extent.height = window.height;
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        cmd.setScissor(0, 1, &scissor);
        //        init_viewports(info);
        //        init_scissors(info);


        cmd.draw(12 * 3, 1, 0, 0);
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

        res = queue.submit( 1, submit_info, drawFence);
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
        do {
            res = device.waitForFences(1, &drawFence, VK_TRUE, 1241515);
        } while (res == vk::Result::eTimeout);


        queue.presentKHR(&present);

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        //        res = vkQueuePresentKHR(info.present_queue, &present);
    }
}


VulkanWindow::~VulkanWindow()
{

    //    vkDestroyPipeline(info.device, info.pipeline, NULL);
    //    for (i = 0; i < info.swapchainImageCount; i++) {
    //        vkDestroyFramebuffer(info.device, info.framebuffers[i], NULL);
    //    }

    //    vkDestroyShaderModule(info.device, info.shaderStages[0].module, NULL);
    //    vkDestroyShaderModule(info.device, info.shaderStages[1].module, NULL);
    //    vkDestroyRenderPass(info.device, info.render_pass, NULL);
    //    for (int i = 0; i < NUM_DESCRIPTOR_SETS; i++) vkDestroyDescriptorSetLayout(info.device, info.desc_layout[i], NULL);
    //    vkDestroyPipelineLayout(info.device, info.pipeline_layout, NULL);

    //    for(vk::ImageView& iv : swapChainImagesViews)
    //    {
    //        device.destroyImageView(iv);
    //    }
    //    device.destroySwapchainKHR(swap_chain);

    device.freeCommandBuffers(cmd_pool,cmd);
    device.destroyCommandPool(cmd_pool);
    device.destroy();
    inst.destroy();
}



}
