/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkanTriangle.h"
#include "saiga/util/assert.h"

namespace Saiga {


VulkanWindow::VulkanWindow()
    : Application(500,500)
{
    uniformBuffer.init(*this);

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
        CHECK_VK(device.createDescriptorSetLayout(&descriptor_layout,nullptr,desc_layout.data()));



        /* Now use the descriptor layout to create a pipeline layout */
        vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo = {};
        //        pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        //        pPipelineLayoutCreateInfo.pNext = NULL;
        pPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
        pPipelineLayoutCreateInfo.pPushConstantRanges = NULL;
        pPipelineLayoutCreateInfo.setLayoutCount = NUM_DESCRIPTOR_SETS;
        pPipelineLayoutCreateInfo.pSetLayouts = desc_layout.data();

        CHECK_VK(device.createPipelineLayout(&pPipelineLayoutCreateInfo, NULL, &pipeline_layout));

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
        CHECK_VK(device.createDescriptorPool(&descriptor_pool, NULL, &desc_pool));



        vk::DescriptorSetAllocateInfo alloc_info[1];
        //           alloc_info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        //           alloc_info[0].pNext = NULL;
        alloc_info[0].descriptorPool = desc_pool;
        alloc_info[0].descriptorSetCount = NUM_DESCRIPTOR_SETS;
        alloc_info[0].pSetLayouts = desc_layout.data();

        desc_set.resize(NUM_DESCRIPTOR_SETS);
        //           res = vkAllocateDescriptorSets(info.device, alloc_info, info.desc_set.data());
        CHECK_VK(device.allocateDescriptorSets(alloc_info, desc_set.data()));



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
        CHECK_VK(device.createPipelineCache(&pipelineCacheInfo, NULL, &pipelineCache));


        CHECK_VK(device.createGraphicsPipelines(pipelineCache, 1, &pipeline_info, NULL, &pipeline));

    }


}


VulkanWindow::~VulkanWindow()
{
    device.destroyPipelineCache(pipelineCache);
    device.destroyPipeline(pipeline);
    device.destroyPipelineLayout(pipeline_layout);
    for(auto d : desc_layout)
    device.destroyDescriptorSetLayout(d);

    device.destroyDescriptorPool(desc_pool);

}

void VulkanWindow::update()
{

}

void VulkanWindow::render(vk::CommandBuffer &cmd)
{
    cmd.bindPipeline( vk::PipelineBindPoint::eGraphics, pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, NUM_DESCRIPTOR_SETS,
                           desc_set.data(), 0, NULL);
    cmd.bindVertexBuffers( 0, vertexBuffer.buffer, 0UL);
    cmd.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);
    cmd.drawIndexed( 3, 1, 0, 0,1);
}



}
