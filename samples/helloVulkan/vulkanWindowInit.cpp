/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkanWindow.h"
#include "saiga/util/assert.h"
namespace Saiga {



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
