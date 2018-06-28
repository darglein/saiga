/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkanWindow.h"
#include "saiga/util/assert.h"

#include "vulkanHelper.h"

namespace Saiga {


void VulkanWindow::init_uniform_buffer()
{
    vk::Result res;
    Projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    View = glm::lookAt(glm::vec3(-5, 3, -10),  // Camera is at (-5,3,-10), in World Space
                            glm::vec3(0, 0, 0),     // and looks at the origin
                            glm::vec3(0, -1, 0)     // Head is up (set to 0,-1,0 to look upside-down)
                            );
    Model = glm::mat4(1.0f);

    // Vulkan clip space has inverted Y and half Z.
    // clang-format off
    Clip = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f,-1.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.5f, 0.0f,
                          0.0f, 0.0f, 0.5f, 1.0f);
    // clang-format on
    MVP = Clip * Projection * View * Model;

    vk::BufferCreateInfo buf_info = {};

    buf_info.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    buf_info.size = sizeof(MVP);
    buf_info.queueFamilyIndexCount = 0;
    buf_info.pQueueFamilyIndices = NULL;
    buf_info.sharingMode = vk::SharingMode::eExclusive;
//    buf_info.flags = 0;
//    res = vkCreateBuffer(info.device, &buf_info, NULL, &info.uniform_data.buf);
    res = device.createBuffer(&buf_info,nullptr,&uniformbuf);
    SAIGA_ASSERT(res == vk::Result::eSuccess);

    vk::MemoryRequirements mem_reqs;
//    vkGetBufferMemoryRequirements(info.device, info.uniform_data.buf, &mem_reqs);
    device.getBufferMemoryRequirements(uniformbuf,&mem_reqs);

    vk::MemoryAllocateInfo alloc_info = {};
    alloc_info.memoryTypeIndex = 0;
    alloc_info.allocationSize = mem_reqs.size;

    auto pass = Vulkan::memory_type_from_properties(memory_properties, mem_reqs.memoryTypeBits,
                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                       &alloc_info.memoryTypeIndex);

    SAIGA_ASSERT(pass);

//    res = vkAllocateMemory(info.device, &alloc_info, NULL, &(info.uniform_data.mem));
    res  = device.allocateMemory(&alloc_info,nullptr,&uniformmem);
    SAIGA_ASSERT(res == vk::Result::eSuccess);
//    assert(res == VK_SUCCESS);

    uint8_t *pData;
//    res = vkMapMemory(info.device, info.uniform_data.mem, 0, mem_reqs.size, 0, (void **)&pData);
    res = device.mapMemory(uniformmem,0,mem_reqs.size, vk::MemoryMapFlags(), (void **)&pData);
//    assert(res == VK_SUCCESS);
    SAIGA_ASSERT(res == vk::Result::eSuccess);


    memcpy(pData, &MVP, sizeof(MVP));

    device.unmapMemory(uniformmem);


//    res = vkBindBufferMemory(info.device, info.uniform_data.buf, info.uniform_data.mem, 0);
    device.bindBufferMemory(uniformbuf,uniformmem,0);


    uniformbuffer_info.buffer = uniformbuf;
    uniformbuffer_info.offset = 0;
    uniformbuffer_info.range = sizeof(MVP);



}

#if 0

void VulkanWindow::createDevice()
{
    {

        vk::Result res;
        std::vector<const char *> device_extension_names;
        device_extension_names.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);


        for(auto str : device_extension_names)
        {
            cout << "[Device Extension] " << str << " enabled." << endl;
        }


        vk::DeviceQueueCreateInfo queue_info;
        float queue_priorities[1] = {0.0};
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = queue_priorities;
        queue_info.queueFamilyIndex = graphics_queue_family_index;



        vk::DeviceCreateInfo device_info = {};
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        //        device_info.enabledExtensionCount = 0;
        //        device_info.ppEnabledExtensionNames = NULL;
        device_info.enabledExtensionCount = device_extension_names.size();
        device_info.ppEnabledExtensionNames = device_info.enabledExtensionCount ? device_extension_names.data() : nullptr;

        device_info.enabledLayerCount = 0;
        device_info.ppEnabledLayerNames = NULL;
        device_info.pEnabledFeatures = NULL;

        res = physicalDevice.createDevice(&device_info,nullptr,&device);
        SAIGA_ASSERT(res == vk::Result::eSuccess);
    }



}
#endif

#if 0
void VulkanWindow::init_vertex_buffer()
{
    {
        vk::Result res;

        /*
         * Set up a vertex buffer:
         * - Create a buffer
         * - Map it and write the vertex data into it
         * - Bind it using vkCmdBindVertexBuffers
         * - Later, at pipeline creation,
         * -      fill in vertex input part of the pipeline with relevent data
         */

        vk::BufferCreateInfo buf_info = {};
//        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
//        buf_info.pNext = NULL;
        buf_info.usage = vk::BufferUsageFlagBits::eVertexBuffer;
        buf_info.size = sizeof(g_vb_solid_face_colors_Data);
        buf_info.queueFamilyIndexCount = 0;
        buf_info.pQueueFamilyIndices = NULL;
        buf_info.sharingMode = vk::SharingMode::eExclusive;
//        buf_info.flags = 0;
        res = device.createBuffer(&buf_info, NULL, &vertexbuf);

        vk::MemoryRequirements mem_reqs;
        device.getBufferMemoryRequirements(vertexbuf, &mem_reqs);

        vk::MemoryAllocateInfo alloc_info = {};
//        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
//        alloc_info.pNext = NULL;
        alloc_info.memoryTypeIndex = 0;

        alloc_info.allocationSize = mem_reqs.size;
        bool pass = Vulkan::memory_type_from_properties(memory_properties,mem_reqs.memoryTypeBits,
                                           vk::MemoryPropertyFlagBits::eHostVisible| vk::MemoryPropertyFlagBits::eHostCoherent,
                                           &alloc_info.memoryTypeIndex);

        SAIGA_ASSERT(pass);

        res = device.allocateMemory(&alloc_info,nullptr,&vertexmem);

        uint8_t *pData;
        device.mapMemory(vertexmem, 0, mem_reqs.size, vk::MemoryMapFlags(), (void **)&pData);
//        assert(res == VK_SUCCESS);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

        memcpy(pData, g_vb_solid_face_colors_Data, sizeof(g_vb_solid_face_colors_Data));

        device.unmapMemory(vertexmem);

        device.bindBufferMemory(vertexbuf,vertexmem,0);

        /* We won't use these here, but we will need this info when creating the
         * pipeline */
        vi_binding.binding = 0;
        vi_binding.inputRate = vk::VertexInputRate::eVertex;
        vi_binding.stride = sizeof(g_vb_solid_face_colors_Data[0]);

        vi_attribs[0].binding = 0;
        vi_attribs[0].location = 0;
        vi_attribs[0].format = vk::Format::eR32G32B32A32Sfloat;
        vi_attribs[0].offset = 0;
        vi_attribs[1].binding = 0;
        vi_attribs[1].location = 1;
        vi_attribs[1].format = vk::Format::eR32G32B32A32Sfloat;
        vi_attribs[1].offset = 16;
    }
}
#endif
}
