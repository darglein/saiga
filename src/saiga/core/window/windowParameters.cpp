/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "windowParameters.h"

#include "saiga/core/util/ini/ini.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{

void WindowParameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());


    int window_mode = (int)mode;
    auto section    = "window";

    INI_GETADD_STRING(ini, section, name);
    INI_GETADD_LONG(ini, section, selected_display);
    INI_GETADD_LONG(ini, section, width);
    INI_GETADD_LONG(ini, section, height);

    INI_GETADD_LONG_COMMENT(ini, section, window_mode,
                            "# 0 Windowed\n"
                            "# 1 WindowedBorderless\n"
                            "# 2 Fullscreen\n"
                            "# 3 Fullscreen\n"
                            "# 4 FullscreenBorderless");

    INI_GETADD_BOOL(ini, section, alwaysOnTop);
    INI_GETADD_BOOL(ini, section, resizeAble);
    INI_GETADD_BOOL(ini, section, vsync);

    SAIGA_ASSERT(window_mode >= 0 && window_mode < 4);
    mode = (Mode)window_mode;

    if (ini.changed()) ini.SaveFile(file.c_str());

    saigaParameters.fromConfigFile(file);
    imguiParameters.fromConfigFile(file);
}



}  // namespace Saiga
