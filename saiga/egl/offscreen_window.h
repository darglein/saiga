/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/window/window.h"

namespace Saiga {

typedef void *EGLDisplay;
typedef void *EGLSurface;

/**
 * OpenGL context without an actual window with EGL
 * https://devblogs.nvidia.com/parallelforall/egl-eye-opengl-visualization-without-x-server/
 */
class SAIGA_GLOBAL OffscreenWindow : public OpenGLWindow{
public:
    EGLDisplay eglDpy;
    EGLSurface eglSurf;
protected:

    //there are no inputs and events without a window
    virtual bool initInput() { return true; }
    virtual void checkEvents() {}


    virtual void swapBuffers();
    virtual void freeContext();

    virtual bool initWindow() override;
public:

    OffscreenWindow(WindowParameters windowParameters);
};

}
