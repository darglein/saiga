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

/**
 * OpenGL context without an actual window with EGL
 * https://devblogs.nvidia.com/parallelforall/egl-eye-opengl-visualization-without-x-server/
 */
class SAIGA_OPENGL_API OffscreenWindow : public OpenGLWindow
{
   public:
    EGLDisplay eglDpy;
    EGLSurface eglSurf;

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
//
// There is also no swapping mechanism.
//  -> Create and render to your own framebuffers
SAIGA_OPENGL_API void CreateEGLContext();
SAIGA_OPENGL_API void TerminateEGLContext();


}  // namespace Saiga
