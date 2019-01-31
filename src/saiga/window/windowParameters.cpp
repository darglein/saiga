/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "windowParameters.h"

#include "saiga/util/ini/ini.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
void WindowParameters::setMode(bool fullscreen, bool borderLess)
{
    if (fullscreen)
    {
        mode = (borderLess) ? Mode::borderLessFullscreen : Mode::fullscreen;
    }
    else
    {
        mode = (borderLess) ? Mode::borderLessWindowed : Mode::windowed;
    }
}

void WindowParameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    bool fullscreen = mode == Mode::borderLessFullscreen || mode == Mode::fullscreen;
    bool borderless = mode == Mode::borderLessWindowed || mode == Mode::borderLessFullscreen;

    name             = ini.GetAddString("window", "name", name.c_str());
    selected_display = static_cast<int>(ini.GetLongValue("window", "display", 0));
    width            = static_cast<int>(ini.GetAddLong("window", "width", width));
    height           = static_cast<int>(ini.GetAddLong("window", "height", height));
    fullscreen       = ini.GetAddBool("window", "fullscreen", fullscreen);
    borderless       = ini.GetAddBool("window", "borderless", borderless);
    alwaysOnTop      = ini.GetAddBool("window", "alwaysOnTop", alwaysOnTop);
    resizeAble       = ini.GetAddBool("window", "resizeAble", resizeAble);
    vsync            = ini.GetAddBool("window", "vsync", vsync);
    //    monitorId    = ini.GetAddLong ("window","monitorId",monitorId);


    if (ini.changed()) ini.SaveFile(file.c_str());

    setMode(fullscreen, borderless);

    saigaParameters.fromConfigFile(file);
    imguiParameters.fromConfigFile(file);
}



}  // namespace Saiga
