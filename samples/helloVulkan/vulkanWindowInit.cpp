/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkanWindow.h"
#include "saiga/util/assert.h"
#include "cube_data.h"
namespace Saiga {

bool memory_type_from_properties(const vk::PhysicalDeviceMemoryProperties& memory_properties, int32_t typeBits, vk::MemoryPropertyFlags requirements_mask, uint32_t *typeIndex) {
    // Search memtypes to find first index with those properties
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            // Type is available, does it match user properties?
            if ((memory_properties.memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                *typeIndex = i;
                return true;
            }
        }
        typeBits >>= 1;
    }
    // No memory types matched, return failure
    return false;
}


static VkResult init_global_extension_properties(LayerPropertiesEx &layer_props) {
    VkExtensionProperties *instance_extensions;
    uint32_t instance_extension_count;
    VkResult res;
    char *layer_name = NULL;

    layer_name = layer_props.properties.layerName;

    do {
        res = vkEnumerateInstanceExtensionProperties(layer_name, &instance_extension_count, NULL);
        if (res) return res;

        if (instance_extension_count == 0) {
            return VK_SUCCESS;
        }

        layer_props.instance_extensions.resize(instance_extension_count);
        instance_extensions = layer_props.instance_extensions.data();
        res = vkEnumerateInstanceExtensionProperties(layer_name, &instance_extension_count, instance_extensions);
    } while (res == VK_INCOMPLETE);

    return res;
}


void VulkanWindow::init_global_layer_properties()
{
    uint32_t instance_layer_count;
    VkLayerProperties *vk_props = NULL;
    VkResult res;
#ifdef __ANDROID__
    // This place is the first place for samples to use Vulkan APIs.
    // Here, we are going to open Vulkan.so on the device and retrieve function pointers using
    // vulkan_wrapper helper.
    if (!InitVulkan()) {
        LOGE("Failied initializing Vulkan APIs!");
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    LOGI("Loaded Vulkan APIs.");
#endif

    /*
     * It's possible, though very rare, that the number of
     * instance layers could change. For example, installing something
     * could include new layers that the loader would pick up
     * between the initial query for the count and the
     * request for VkLayerProperties. The loader indicates that
     * by returning a VK_INCOMPLETE status and will update the
     * the count parameter.
     * The count parameter will be updated with the number of
     * entries loaded into the data pointer - in case the number
     * of layers went down or is smaller than the size given.
     */
    do {
        res = vkEnumerateInstanceLayerProperties(&instance_layer_count, NULL);
//        if (res) return res;

//        if (instance_layer_count == 0) {
//            return VK_SUCCESS;
//        }

        vk_props = (VkLayerProperties *)realloc(vk_props, instance_layer_count * sizeof(VkLayerProperties));

        res = vkEnumerateInstanceLayerProperties(&instance_layer_count, vk_props);
    } while (res == VK_INCOMPLETE);

    /*
     * Now gather the extension list for each instance layer.
     */
    for (uint32_t i = 0; i < instance_layer_count; i++) {
        LayerPropertiesEx layer_props;
        layer_props.properties = vk_props[i];
        res = init_global_extension_properties(layer_props);
//        if (res) return res;
        layerProperties.push_back(layer_props);
    }
    free(vk_props);
//    return res;
}

void VulkanWindow::init_instance()
{

    vk::Result res;
    // ======================= Create Vulkan Instance =======================

    {
        std::vector<const char *> instance_extension_names;

        instance_extension_names.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef __ANDROID__
        info.instance_extension_names.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_WIN32)
        info.instance_extension_names.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
        info.instance_extension_names.push_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
        info.instance_extension_names.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
        info.instance_extension_names.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#else
        instance_extension_names.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
        for(auto str : instance_extension_names)
        {
            cout << "[Instance Extension] " << str << " enabled." << endl;
        }





        // initialize the VkApplicationInfo structure
        vk::ApplicationInfo app_info = {};
        app_info.pApplicationName = name.c_str();
        app_info.applicationVersion = 1;
        app_info.pEngineName = name.c_str();
        app_info.engineVersion = 1;
        app_info.apiVersion = VK_API_VERSION_1_0;

        // initialize the VkInstanceCreateInfo structure
        vk::InstanceCreateInfo inst_info = {};
        inst_info.pApplicationInfo = &app_info;
        //        inst_info.enabledExtensionCount = 0;
        //        inst_info.ppEnabledExtensionNames = nullptr;
        inst_info.enabledExtensionCount = instance_extension_names.size();
        inst_info.ppEnabledExtensionNames = instance_extension_names.data();

        inst_info.enabledLayerCount = 0;
        inst_info.ppEnabledLayerNames = nullptr;

        //    VkInstance inst;



        res = vk::createInstance(&inst_info,nullptr,&inst);
        SAIGA_ASSERT(res == vk::Result::eSuccess);
    }
}


#if 0
void VulkanWindow::init_depth_buffer()
{
    vk::Result res;
    {
        // depth buffer
        vk::ImageCreateInfo image_info = {};
        const vk::Format depth_format = vk::Format::eD16Unorm;
        vk::FormatProperties props;
//        vkGetPhysicalDeviceFormatProperties(info.gpus[0], depth_format, &props);
        props = physicalDevice.getFormatProperties(depth_format);

        if (props.linearTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            image_info.tiling = vk::ImageTiling::eLinear;
        } else if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            image_info.tiling = vk::ImageTiling::eOptimal;
        } else {
            /* Try other depth formats? */
            std::cout << "VK_FORMAT_D16_UNORM Unsupported.\n";
            exit(-1);
        }
        image_info.imageType = vk::ImageType::e2D;
        image_info.format = depth_format;
        image_info.extent.width =  window.width;
        image_info.extent.height = window.height;
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = vk::SampleCountFlagBits::e1;
        image_info.initialLayout = vk::ImageLayout::eUndefined;
        image_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
        image_info.queueFamilyIndexCount = 0;
        image_info.pQueueFamilyIndices = NULL;
        image_info.sharingMode = vk::SharingMode::eExclusive;
//        image_info.flags = 0;



        vk::ImageViewCreateInfo viewInfo = {};

        viewInfo.image = nullptr;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = depth_format;
        viewInfo.components.r = vk::ComponentSwizzle::eR;
        viewInfo.components.g = vk::ComponentSwizzle::eG;
        viewInfo.components.b = vk::ComponentSwizzle::eB;
        viewInfo.components.a = vk::ComponentSwizzle::eA;
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        vk::MemoryRequirements mem_reqs;

        depthFormat = depth_format;

        /* Create image */
//        res = vkCreateImage(info.device, &image_info, NULL, &info.depth.image);
        res = device.createImage(&image_info,nullptr,&depthimage);
        SAIGA_ASSERT(res == vk::Result::eSuccess);
//        assert(res == VK_SUCCESS);


//        vkGetImageMemoryRequirements(info.device, info.depth.image, &mem_reqs);
        mem_reqs = device.getImageMemoryRequirements(depthimage);

        vk::MemoryAllocateInfo mem_alloc = {};
//        mem_alloc.allocationSize = 0;
        mem_alloc.memoryTypeIndex = 0;
        mem_alloc.allocationSize = mem_reqs.size;

        bool pass =
            memory_type_from_properties(memory_properties, mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, &mem_alloc.memoryTypeIndex);
        SAIGA_ASSERT(pass);



        res  = device.allocateMemory(&mem_alloc,nullptr,&depthmem);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

        device.bindImageMemory(depthimage,depthmem,0);


        viewInfo.image = depthimage;
        res = device.createImageView(&viewInfo,nullptr,&depthview);
        SAIGA_ASSERT(res == vk::Result::eSuccess);
    }

}
#endif

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

    auto pass = memory_type_from_properties(memory_properties, mem_reqs.memoryTypeBits,
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
        bool pass = memory_type_from_properties(memory_properties,mem_reqs.memoryTypeBits,
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

#if 0
        const vk::DeviceSize offsets[1] = {0};

        /* We cannot bind the vertex buffer until we begin a renderpass */
        vk::ClearValue clear_values[2];
        clear_values[0].color.float32[0] = 0.2f;
        clear_values[0].color.float32[1] = 0.2f;
        clear_values[0].color.float32[2] = 0.2f;
        clear_values[0].color.float32[3] = 0.2f;
        clear_values[1].depthStencil.depth = 1.0f;
        clear_values[1].depthStencil.stencil = 0;





//        vk::Result res;

        // A semaphore (or fence) is required in order to acquire a
        // swapchain image to prepare it for use in a render pass.
        // The semaphore is normally used to hold back the rendering
        // operation until the image is actually available.
        // But since this sample does not render, the semaphore
        // ends up being unused.
        vk::Semaphore imageAcquiredSemaphore;
        vk::SemaphoreCreateInfo imageAcquiredSemaphoreCreateInfo;
//        imageAcquiredSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
//        imageAcquiredSemaphoreCreateInfo.pNext = NULL;
//        imageAcquiredSemaphoreCreateInfo.flags = 0;

//        res = vkCreateSemaphore(info.device, &imageAcquiredSemaphoreCreateInfo, NULL, &imageAcquiredSemaphore);
        res = device.createSemaphore( &imageAcquiredSemaphoreCreateInfo, NULL, &imageAcquiredSemaphore);
//        assert(res == VK_SUCCESS);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

        // Acquire the swapchain image in order to set its layout
//        res = vkAcquireNextImageKHR(info.device, info.swap_chain, UINT64_MAX, imageAcquiredSemaphore, VK_NULL_HANDLE,
//                                    &info.current_buffer);
        current_buffer = device.acquireNextImageKHR(swap_chain, UINT64_MAX, imageAcquiredSemaphore, vk::Fence()).value;
        SAIGA_ASSERT(res == vk::Result::eSuccess);



        vk::RenderPassBeginInfo rp_begin = {};
//        rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
//        rp_begin.pNext = NULL;
        rp_begin.renderPass = render_pass;
        rp_begin.framebuffer = framebuffers[current_buffer];
        rp_begin.renderArea.offset.x = 0;
        rp_begin.renderArea.offset.y = 0;
        rp_begin.renderArea.extent.width = width;
        rp_begin.renderArea.extent.height = height;
        rp_begin.clearValueCount = 2;
        rp_begin.pClearValues = clear_values;

//        vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
        cmd.beginRenderPass(&rp_begin, vk::SubpassContents::eInline);
        cmd.bindVertexBuffers(0,1,&vertexbuf,offsets);

//        vkCmdBindVertexBuffers(info.cmd, 0,             /* Start Binding */
//                               1,                       /* Binding Count */
//                               &info.vertex_buffer.buf, /* pBuffers */
//                               offsets);                /* pOffsets */

        cmd.endRenderPass();
//        vkCmdEndRenderPass(info.cmd);


//        vkDestroySemaphore(info.device, imageAcquiredSemaphore, NULL);
        device.destroySemaphore(imageAcquiredSemaphore,nullptr);
#endif
    }
}

}
