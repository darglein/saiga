/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/framework/framework.h"
#include "saiga/core/imgui/imgui_saiga.h"

namespace Saiga
{
struct SAIGA_CORE_API WindowParameters
{
    enum class Mode
    {
        Windowed             = 0,
        WindowedBorderless   = 1,
        Fullscreen           = 2,
        FullscreenWindowed   = 3,
        FullscreenBorderless = 4,

    };

    // Where the window is created. The backend is chosen at runtime, so one build of an application can run with or
    // without a window system.
    enum class Backend
    {
        // A window of the window system (needs a display).
        GLFW = 0,
        // No window system: OpenGL renders into an EGL pbuffer (without EGL, as on Windows, into a GLFW window that
        // is never shown), Vulkan into offscreen images. There is no input from the window system; the application
        // feeds ImGui's input queue itself if it needs input.
        Headless = 1,
    };

    std::string name     = "Saiga";
    Backend backend      = Backend::GLFW;
    int width            = 1600;
    int height           = 900;
    int selected_display = 0;
    Mode mode            = Mode::FullscreenWindowed;

    bool finishBeforeSwap = false;  // adds a glFinish before swapBuffers
    bool hidden           = false;  // for offscreen rendering
    bool alwaysOnTop      = false;
    bool resizeAble       = true;
    bool vsync            = false;
    bool updateJoystick   = false;
    int monitorId         = 0;  // Important for Fullscreen mode. 0 is always the primary monitor.

    // The content scale (the ratio of the UI scale to the default) a headless window reports. A window of the window
    // system reports the scale of its monitor instead.
    float headlessContentScale = 1.0f;

    bool createWindowMenuItem = true;


    // time in seconds between debug screenshots. negativ for no debug screenshots
    float debugScreenshotTime       = -1.0f;
    std::string debugScreenshotPath = "debug/";

    ImGuiParameters imguiParameters;
    SaigaParameters saigaParameters;



    bool borderLess() { return mode == Mode::WindowedBorderless || mode == Mode::FullscreenBorderless; }
    bool fullscreen()
    {
        return mode == Mode::Fullscreen || mode == Mode::FullscreenBorderless || mode == Mode::FullscreenWindowed;
    }

    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};


}  // namespace Saiga
