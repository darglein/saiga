/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "windowParameters.h"

#include "saiga/core/util/ini/ini.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>
#include <cctype>
#include <iostream>

namespace Saiga
{

void WindowParameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());


    int window_mode = (int)mode;
    auto section    = "window";

    INI_GETADD(ini, section, name);

    std::string window_backend = backend == Backend::Headless ? "headless" : "glfw";
    INI_GETADD_COMMENT(ini, section, window_backend, "# glfw or headless");
    std::string backend_name = window_backend;
    std::transform(backend_name.begin(), backend_name.end(), backend_name.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    if (backend_name == "headless")
        backend = Backend::Headless;
    else if (backend_name == "glfw")
        backend = Backend::GLFW;
    else
        std::cerr << "Unknown window_backend '" << window_backend << "' in " << file << ", using "
                  << (backend == Backend::Headless ? "headless" : "glfw") << std::endl;

    INI_GETADD(ini, section, selected_display);
    INI_GETADD(ini, section, width);
    INI_GETADD(ini, section, height);

    INI_GETADD_COMMENT(ini, section, window_mode,
                            "# 0 Windowed\n"
                            "# 1 WindowedBorderless\n"
                            "# 2 Fullscreen\n"
                            "# 3 FullscreenWindowed\n"
                            "# 4 FullscreenBorderless");

    INI_GETADD(ini, section, alwaysOnTop);
    INI_GETADD(ini, section, resizeAble);
    INI_GETADD(ini, section, vsync);

    SAIGA_ASSERT(window_mode >= 0 && window_mode < 4);
    mode = (Mode)window_mode;

    if (ini.changed()) ini.SaveFile(file.c_str());

    saigaParameters.fromConfigFile(file);
    imguiParameters.fromConfigFile(file);
}



}  // namespace Saiga
