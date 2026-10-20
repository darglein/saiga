/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#if defined(SAIGA_USE_OPENGL) && defined(SAIGA_USE_EGL)
#    include "saiga/core/util/assert.h"
#    include "saiga/opengl/rendering/deferredRendering/deferred_renderer.h"

#    include <EGL/egl.h>
#    include <EGL/eglext.h>
#    include <cstdlib>
#    include <cstring>

#    include "offscreen_window.h"

namespace Saiga
{
struct EGLHandles
{
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLSurface surface = EGL_NO_SURFACE;
    EGLContext context = EGL_NO_CONTEXT;
};

static bool HasClientExtension(const char* name)
{
    const char* extensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    if (!extensions) return false;
    std::string list = std::string(" ") + extensions + " ";
    return list.find(std::string(" ") + name + " ") != std::string::npos;
}

// Mesa's software rasterizer (llvmpipe) is an EGL device with this extension.
static bool IsSoftwareDevice(EGLDeviceEXT device)
{
    auto queryDeviceString = (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");
    if (!queryDeviceString) return false;
    const char* extensions = queryDeviceString(device, EGL_EXTENSIONS);
    return extensions && std::strstr(extensions, "EGL_MESA_device_software");
}

// The displays to try, in order: every EGL device (headless, no X server), the Mesa surfaceless platform, the
// default display.
static std::vector<EGLDisplay> CandidateDisplays()
{
    std::vector<EGLDisplay> displays;

    auto getPlatformDisplay = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    auto queryDevices       = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");

    // Mesa's LIBGL_ALWAYS_SOFTWARE: use only its software device (Mesa crashes initializing a hardware device then).
    const char* force_software = std::getenv("LIBGL_ALWAYS_SOFTWARE");
    bool software_only         = force_software && std::strcmp(force_software, "0") != 0;

    if (getPlatformDisplay && queryDevices && HasClientExtension("EGL_EXT_platform_device"))
    {
        EGLDeviceEXT devices[16];
        EGLint num_devices = 0;
        if (queryDevices(16, devices, &num_devices))
        {
            for (int i = 0; i < num_devices; ++i)
            {
                if (software_only && !IsSoftwareDevice(devices[i])) continue;
                EGLDisplay dpy = getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
                if (dpy != EGL_NO_DISPLAY) displays.push_back(dpy);
            }
        }
    }

    if (getPlatformDisplay && HasClientExtension("EGL_MESA_platform_surfaceless"))
    {
        EGLDisplay dpy = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        if (dpy != EGL_NO_DISPLAY) displays.push_back(dpy);
    }

    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy != EGL_NO_DISPLAY) displays.push_back(dpy);

    return displays;
}

static EGLContext CreateContext(EGLDisplay dpy, EGLConfig cfg, int major, int minor, bool core)
{
    EGLint attribs[] = {EGL_CONTEXT_MAJOR_VERSION,
                        major,
                        EGL_CONTEXT_MINOR_VERSION,
                        minor,
                        EGL_CONTEXT_OPENGL_PROFILE_MASK,
                        core ? EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT : EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
                        EGL_NONE};
    return eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, attribs);
}

static bool TryDisplay(EGLDisplay dpy, int width, int height, const std::vector<std::pair<int, int>>& versions,
                       bool core, EGLHandles& out)
{
    EGLint major, minor;
    if (!eglInitialize(dpy, &major, &minor)) return false;

    EGLint numConfigs = 0;
    EGLConfig eglCfg;
    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT, EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8, EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,  EGL_NONE};
    if (!eglChooseConfig(dpy, configAttribs, &eglCfg, 1, &numConfigs) || numConfigs < 1 || !eglBindAPI(EGL_OPENGL_API))
    {
        eglTerminate(dpy);
        return false;
    }

    EGLint pbufferAttribs[] = {
        EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE,
    };
    EGLSurface surf = eglCreatePbufferSurface(dpy, eglCfg, pbufferAttribs);
    if (surf == EGL_NO_SURFACE)
    {
        eglTerminate(dpy);
        return false;
    }

    EGLContext ctx = EGL_NO_CONTEXT;
    for (auto v : versions)
    {
        ctx = CreateContext(dpy, eglCfg, v.first, v.second, core);
        if (ctx != EGL_NO_CONTEXT) break;
    }
    // Last resort: whatever version the driver gives by default.
    if (ctx == EGL_NO_CONTEXT) ctx = eglCreateContext(dpy, eglCfg, EGL_NO_CONTEXT, nullptr);

    if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, surf, surf, ctx))
    {
        if (ctx != EGL_NO_CONTEXT) eglDestroyContext(dpy, ctx);
        eglDestroySurface(dpy, surf);
        eglTerminate(dpy);
        return false;
    }

    out.display = dpy;
    out.surface = surf;
    out.context = ctx;
    return true;
}

static bool CreateEGLHandles(int width, int height, const std::vector<std::pair<int, int>>& versions, bool core,
                             EGLHandles& out)
{
    for (auto dpy : CandidateDisplays())
    {
        if (TryDisplay(dpy, width, height, versions, core, out)) return true;
    }
    std::cerr << "Failed to create an EGL context (EGL error 0x" << std::hex << eglGetError() << std::dec << ")"
              << std::endl;
    return false;
}

static void DestroyEGLHandles(EGLHandles& h)
{
    if (h.display == EGL_NO_DISPLAY) return;
    eglMakeCurrent(h.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (h.context != EGL_NO_CONTEXT) eglDestroyContext(h.display, h.context);
    if (h.surface != EGL_NO_SURFACE) eglDestroySurface(h.display, h.surface);
    eglTerminate(h.display);
    h = EGLHandles();
}

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
    EGLHandles h;
    h.display = eglDpy;
    h.surface = eglSurf;
    h.context = eglCtx;
    DestroyEGLHandles(h);
    eglDpy  = nullptr;
    eglSurf = nullptr;
    eglCtx  = nullptr;
}

bool OffscreenWindow::initWindow()
{
    bool core = openglParameters.profile == OpenGLParameters::Profile::CORE;
    EGLHandles h;
    if (!CreateEGLHandles(windowParameters.width, windowParameters.height,
                          {{openglParameters.versionMajor, openglParameters.versionMinor}}, core, h))
    {
        return false;
    }
    eglDpy  = h.display;
    eglSurf = h.surface;
    eglCtx  = h.context;
    return true;
}

void OffscreenWindow::loadGLFunctions()
{
    initOpenGL(eglGetProcAddress);
}

static EGLHandles raw_context;

bool CreateEGLContext()
{
    SAIGA_ASSERT(raw_context.display == EGL_NO_DISPLAY, "An EGL context was already created");

    // The newest core profile the driver offers; 4.3 is the minimum for compute shaders.
    if (!CreateEGLHandles(4, 4, {{4, 6}, {4, 5}, {4, 3}, {3, 3}}, true, raw_context))
    {
        return false;
    }

    initOpenGL(eglGetProcAddress);
    initSaigaGL(OpenGLParameters());
    return true;
}

void TerminateEGLContext()
{
    cleanupSaigaGL();
    terminateOpenGL();
    DestroyEGLHandles(raw_context);
}

}  // namespace Saiga
#endif
