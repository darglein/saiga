/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "window.h"

namespace Saiga {
namespace Vulkan {

Window::Window()
{

}

Window::~Window()
{

}

void Window::createWindow(int w, int h)
{
    width = w;
    height = h;
    // init window
    SAIGA_ASSERT(width > 0);
    SAIGA_ASSERT(height > 0);

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

vk::SurfaceKHR Window::createSurfaceKHR(vk::Instance &inst)
{
    // Create the os-specific surface
#ifdef _WIN32
    vk::Win32SurfaceCreateInfoKHR surfaceCreateInfo;
    surfaceCreateInfo.hinstance = (HINSTANCE)platformHandle;
    surfaceCreateInfo.hwnd = (HWND)platformWindow;
    surface = instance.createWin32SurfaceKHR(surfaceCreateInfo);
#else
#ifdef __ANDROID__
    vk::AndroidSurfaceCreateInfoKHR surfaceCreateInfo;
    surfaceCreateInfo.window = window;
    surface = instance.createAndroidSurfaceKHR(surfaceCreateInfo);
#else
#if defined(_DIRECT2DISPLAY)
    createDirect2DisplaySurface(width, height);
#else
    vk::XcbSurfaceCreateInfoKHR surfaceCreateInfo;
    surfaceCreateInfo.connection = connection;
    surfaceCreateInfo.window = window;
    return inst.createXcbSurfaceKHR(surfaceCreateInfo);
#endif
#endif
#endif
}

}
}
