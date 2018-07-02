/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/framework.h"


namespace Saiga {


struct SAIGA_GLOBAL OpenGLParameters
{
    enum class Profile{
        ANY,
        CORE,
        COMPATIBILITY
    };
    Profile profile = Profile::CORE;

    bool debug = true;

    //all functionality deprecated in the requested version of OpenGL is removed
    bool forwardCompatible = false;

    int versionMajor = 3;
    int versionMinor = 2;

    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};

struct SAIGA_GLOBAL WindowParameters
{
    enum class Mode{
        windowed,
        fullscreen,
        borderLessWindowed,
        borderLessFullscreen
    };

    std::string name = "Saiga";
    int width = 1280;
    int height = 720;
    Mode mode =  Mode::windowed;

    bool finishBeforeSwap = false; //adds a glFinish before swapBuffers
    bool hidden = false; //for offscreen rendering
    bool alwaysOnTop = false;
    bool resizeAble = true;
    bool vsync = false;
    bool updateJoystick = false;
    int monitorId = 0; //Important for fullscreen mode. 0 is always the primary monitor.

    //time in seconds between debug screenshots. negativ for no debug screenshots
    float debugScreenshotTime = -1.0f;
    std::string debugScreenshotPath = "debug/";

    OpenGLParameters openglparameters;
    SaigaParameters saigaParameters;

    bool createImgui = true;
    std::string imguiFont = "";
    int imguiFontSize = 15;

    bool borderLess(){ return mode==Mode::borderLessWindowed || mode==Mode::borderLessFullscreen;}
    bool fullscreen(){ return mode==Mode::fullscreen || mode==Mode::borderLessFullscreen;}

    void setMode(bool fullscreen, bool borderLess);

    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);


};


}
