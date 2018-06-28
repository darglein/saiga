/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "vulkan.h"
#include "vulkanBase.h"

namespace Saiga {
namespace Vulkan {

/**
 * This class capsules the platform specific code for window creation.
 */
class SAIGA_GLOBAL Window
{
public:
    Window();
    ~Window();

    void createWindow(int w, int h);
      vk::SurfaceKHR createSurfaceKHR(vk::Instance& inst);
public:
    uint32_t width = 500, height = 500;
//    vk::SurfaceKHR surface;
protected:
    xcb_connection_t *connection;
    xcb_screen_t* screen;
    xcb_window_t window;
    xcb_intern_atom_reply_t *atom_wm_delete_window;
};

}
}
