/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/window/windowParameters.h"
#include "saiga/util/ini/ini.h"

namespace Saiga {

void WindowParameters::setMode(bool fullscreen, bool borderLess)
{
    if(fullscreen){
        mode = (borderLess) ? Mode::borderLessFullscreen : Mode::fullscreen;
    }else{
        mode = (borderLess) ? Mode::borderLessWindowed : Mode::windowed;
    }
}

void WindowParameters::fromConfigFile(const std::string &file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    bool fullscreen = mode == Mode::borderLessFullscreen || mode == Mode::fullscreen;
    bool borderless = mode == Mode::borderLessWindowed || mode == Mode::borderLessFullscreen;

    name         = ini.GetAddString ("window","name",name.c_str());
    width        = ini.GetAddLong ("window","width",width);
    height       = ini.GetAddLong ("window","height",height);
    fullscreen   = ini.GetAddBool ("window","fullscreen",fullscreen);
    borderless   = ini.GetAddBool ("window","borderless",borderless);
    alwaysOnTop  = ini.GetAddBool ("window","alwaysOnTop",alwaysOnTop);
    resizeAble   = ini.GetAddBool ("window","resizeAble",resizeAble);
    vsync        = ini.GetAddBool ("window","vsync",vsync);
    //    monitorId    = ini.GetAddLong ("window","monitorId",monitorId);


    if(ini.changed()) ini.SaveFile(file.c_str());

    setMode(fullscreen,borderless);


    openglparameters.fromConfigFile(file);
    saigaParameters.fromConfigFile(file);
}

void OpenGLParameters::fromConfigFile(const std::string &file)
{

    bool core = true;

    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    debug               = ini.GetAddBool ("opengl","debug",debug);
    forwardCompatible   = ini.GetAddBool ("opengl","forwardCompatible",forwardCompatible);
    versionMajor        = ini.GetAddLong ("opengl","versionMajor",versionMajor);
    versionMinor        = ini.GetAddLong ("opengl","versionMinor",versionMinor);
    core                = ini.GetAddBool ("opengl","core",core);

    if(ini.changed()) ini.SaveFile(file.c_str());

    profile = core ? Profile::CORE : Profile::COMPATIBILITY;
}

void MainLoopParameters::fromConfigFile(const std::string &file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    updatesPerSecond      = ini.GetAddLong  ("mainloop","updatesPerSecond",updatesPerSecond);
    framesPerSecond       = ini.GetAddLong  ("mainloop","framesPerSecond",framesPerSecond);
    mainLoopInfoTime      = ini.GetAddDouble("mainloop","mainLoopInfoTime",mainLoopInfoTime);
    maxFrameSkip          = ini.GetAddLong  ("mainloop","maxFrameSkip",maxFrameSkip);
    parallelUpdate        = ini.GetAddBool  ("mainloop","parallelUpdate",parallelUpdate);
    catchUp               = ini.GetAddBool  ("mainloop","catchUp",catchUp);
    printInfoMsg          = ini.GetAddBool  ("mainloop","printInfoMsg",printInfoMsg);

    if(ini.changed()) ini.SaveFile(file.c_str());
}


}
