/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_OPENGL
#    include "saiga/egl/offscreen_window.h"
#    include "saiga/rendering/deferredRendering/deferred_renderer.h"
#    include "saiga/util/assert.h"

#    include <EGL/egl.h>

namespace Saiga
{
OffscreenWindow::OffscreenWindow(WindowParameters windowParameters, OpenGLParameters openglParameter)
    : OpenGLWindow(windowParameters, openglParameter)
{
    create();
}

void OffscreenWindow::swapBuffers()
{
    eglSwapBuffers(eglDpy, eglSurf);
}

void OffscreenWindow::freeContext()
{
    // 6. Terminate EGL when finished
    eglTerminate(eglDpy);
}

bool OffscreenWindow::initWindow()
{
    // 1. Initialize EGL
    eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    EGLint major, minor;

    eglInitialize(eglDpy, &major, &minor);

    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT, EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8, EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,  EGL_NONE};
    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

    // 3. Create a surface
    EGLint pbufferAttribs[] = {
        EGL_WIDTH, windowParameters.width, EGL_HEIGHT, windowParameters.height, EGL_NONE,
    };
    eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);

    // 4. Bind the API
    eglBindAPI(EGL_OPENGL_API);

    // 5. Create a context and make it current
    EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);

    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);


    return true;
}

void OffscreenWindow::loadGLFunctions()
{
    initOpenGL(eglGetProcAddress);
}

}  // namespace Saiga
#endif
