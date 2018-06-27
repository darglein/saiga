/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkanWindow.h"
#include "saiga/util/assert.h"
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

void VulkanWindow::init_physical_device()
{
    // ======================= Physical Devices =======================


    {
        // Print all physical devices and choose first one.
        physicalDevices = inst.enumeratePhysicalDevices();
        SAIGA_ASSERT(physicalDevices.size() >= 1);
        for(vk::PhysicalDevice& d : physicalDevices)
        {
            vk::PhysicalDeviceProperties props = d.getProperties();
            cout << "[Device] Id=" << props.deviceID << " "  << props.deviceName << " Type=" << (int)props.deviceType << endl;
        }
        physicalDevice = physicalDevices[0];
    }

    {

        cout << "Creating a device from physical id " << physicalDevice.getProperties().deviceID << "." << endl;
        queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
        SAIGA_ASSERT(queueFamilyProperties.size() >= 1);
        for(vk::QueueFamilyProperties& qf : queueFamilyProperties)
        {
            cout << "[QueueFamily] Count=" << qf.queueCount << " flags=" << (unsigned int)qf.queueFlags << endl;
        }
    }

    /* This is as good a place as any to do this */
//    vkGetPhysicalDeviceMemoryProperties(info.gpus[0], &info.memory_properties);
//    vkGetPhysicalDeviceProperties(info.gpus[0], &info.gpu_props);
    memory_properties = physicalDevice.getMemoryProperties();
    gpu_props = physicalDevice.getProperties();

}

void VulkanWindow::init_swapchain_extension()
{
    vk::Result res;
    {
        // construct surface
        vk::XcbSurfaceCreateInfoKHR createInfo = {};
        createInfo.connection = connection;
        createInfo.window = window;
        //        res = vkCreateXcbSurfaceKHR(info.inst, &createInfo, NULL, &info.surface);

        res = inst.createXcbSurfaceKHR( &createInfo, NULL, &surface);
        SAIGA_ASSERT(res == vk::Result::eSuccess);

        // Iterate over each queue to learn whether it supports presenting:
        //        VkBool32 *pSupportsPresent = (VkBool32 *)malloc(queue_family_count * sizeof(VkBool32));

        // search for a graphics queue
        for (unsigned int i = 0; i < queueFamilyProperties.size(); i++) {

            if (queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)
            {
                if(graphics_queue_family_index == -1)
                {
                    graphics_queue_family_index = i;
                }
                vk::Bool32 b = physicalDevice.getSurfaceSupportKHR(i,surface);
                if(b)
                {
                    graphics_queue_family_index = i;
                    present_queue_family_index = i;
                    break;
                }
            }
        }
    }

    SAIGA_ASSERT(graphics_queue_family_index != -1);
    SAIGA_ASSERT(present_queue_family_index != -1);

    {

        std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);
        SAIGA_ASSERT(surfaceFormats.size() >= 1);
        // If the format list includes just one entry of VK_FORMAT_UNDEFINED,
        // the surface has no preferred format.  Otherwise, at least one
        // supported format will be returned.
        if (surfaceFormats.size() == 1 && surfaceFormats[0].format == vk::Format::eUndefined) {
            format = vk::Format::eB8G8R8A8Unorm;
        } else {
            assert(surfaceFormats.size() >= 1);
            format = surfaceFormats[0].format;
        }



    }
}

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
        image_info.extent.width =  width;
        image_info.extent.height = height;
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

void VulkanWindow::createWindow()
{

    // ======================= XCB =======================
    {
        const xcb_setup_t *setup;
        xcb_screen_iterator_t iter;
        int scr;

        connection = xcb_connect(NULL, &scr);
        if (connection == NULL || xcb_connection_has_error(connection)) {
            std::cout << "Unable to make an XCB connection\n";
            exit(-1);
        }

        setup = xcb_get_setup(connection);
        iter = xcb_setup_roots_iterator(setup);
        while (scr-- > 0) xcb_screen_next(&iter);

        screen = iter.data;
    }

    {
        // init window
        SAIGA_ASSERT(width > 0);
        SAIGA_ASSERT(height > 0);

        uint32_t value_mask, value_list[32];

        window = xcb_generate_id(connection);

        value_mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
        value_list[0] = screen->black_pixel;
        value_list[1] = XCB_EVENT_MASK_KEY_RELEASE | XCB_EVENT_MASK_EXPOSURE;

        xcb_create_window(connection, XCB_COPY_FROM_PARENT, window, screen->root, 0, 0, width, height, 0,
                          XCB_WINDOW_CLASS_INPUT_OUTPUT, screen->root_visual, value_mask, value_list);

        /* Magic code that will send notification when window is destroyed */
        xcb_intern_atom_cookie_t cookie = xcb_intern_atom(connection, 1, 12, "WM_PROTOCOLS");
        xcb_intern_atom_reply_t *reply = xcb_intern_atom_reply(connection, cookie, 0);

        xcb_intern_atom_cookie_t cookie2 = xcb_intern_atom(connection, 0, 16, "WM_DELETE_WINDOW");
        atom_wm_delete_window = xcb_intern_atom_reply(connection, cookie2, 0);

        xcb_change_property(connection, XCB_PROP_MODE_REPLACE, window, (*reply).atom, 4, 32, 1,
                            &(*atom_wm_delete_window).atom);
        free(reply);

        xcb_map_window(connection, window);

        // Force the x/y coordinates to 100,100 results are identical in consecutive
        // runs
        const uint32_t coords[] = {100, 100};
        xcb_configure_window(connection, window, XCB_CONFIG_WINDOW_X | XCB_CONFIG_WINDOW_Y, coords);
        xcb_flush(connection);

        xcb_generic_event_t *e;
        while ((e = xcb_wait_for_event(connection))) {
            if ((e->response_type & ~0x80) == XCB_EXPOSE) break;
        }
    }

}

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

}
