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

    std::string profileString;
    switch(profile)
    {
    case Profile::ANY:
        profileString = "ANY";
        break;
    case Profile::CORE:
        profileString = "CORE";
        break;
    case Profile::COMPATIBILITY:
        profileString = "COMPATIBILITY";
        break;
    }

    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    debug               = ini.GetAddBool ("opengl","debug",debug);
    forwardCompatible   = ini.GetAddBool ("opengl","forwardCompatible",forwardCompatible);
    versionMajor        = ini.GetAddLong ("opengl","versionMajor",versionMajor);
    versionMinor        = ini.GetAddLong ("opengl","versionMinor",versionMinor);
    profileString       = ini.GetAddString ("opengl","profile",profileString.c_str(),"# One of the following: 'ANY' 'CORE' 'COMPATIBILITY'");

    if(ini.changed()) ini.SaveFile(file.c_str());


    profile = profileString == "ANY" ? Profile::ANY : profileString == "CORE" ? Profile::CORE : Profile::COMPATIBILITY;
}




}
