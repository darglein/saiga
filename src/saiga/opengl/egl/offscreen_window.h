/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/opengl/window/OpenGLWindow.h"

namespace Saiga
{
typedef void* EGLDisplay;
typedef void* EGLSurface;
typedef void* EGLContext;

/**
 * OpenGL context without an actual window with EGL
 * https://devblogs.nvidia.com/parallelforall/egl-eye-opengl-visualization-without-x-server/
 *
 * The display is taken from the first EGL device that works (no X server needed), then the Mesa surfaceless
 * platform, then EGL_DEFAULT_DISPLAY.
 */
class SAIGA_OPENGL_API OffscreenWindow : public OpenGLWindow
{
   public:
    EGLDisplay eglDpy  = nullptr;
    EGLSurface eglSurf = nullptr;
    EGLContext eglCtx  = nullptr;

   protected:
    // there are no inputs and events without a window
    virtual bool initInput() override { return true; }
    virtual void checkEvents() override {}


    virtual void swapBuffers() override;
    virtual void freeContext() override;

    virtual bool initWindow() override;
    virtual void loadGLFunctions() override;

   public:
    OffscreenWindow(WindowParameters windowParameters, OpenGLParameters openglParameter = OpenGLParameters());
};

// Create a raw EGL context without an OpenGL window and renderer.
// Do not render or use the default framebuffer with this method.
// Returns false if no EGL display or context could be created (for example, no GPU and no Mesa).
//
// There is also no swapping mechanism.
//  -> Create and render to your own framebuffers
SAIGA_OPENGL_API bool CreateEGLContext();
SAIGA_OPENGL_API void TerminateEGLContext();


}  // namespace Saiga
