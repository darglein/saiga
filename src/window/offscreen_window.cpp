
#include "saiga/window/offscreen_window.h"
#include "saiga/rendering/deferred_renderer.h"

#include <GL/glx.h>
#include <GL/gl.h>

#include <EGL/egl.h>

#include "saiga/util/assert.h"

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);
static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = NULL;
static glXMakeContextCurrentARBProc   glXMakeContextCurrentARB   = NULL;


OffscreenWindow::OffscreenWindow(WindowParameters windowParameters):OpenGLWindow(windowParameters)
{
}

bool initWindow2()
{
    glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc) glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );
    glXMakeContextCurrentARB   = (glXMakeContextCurrentARBProc)   glXGetProcAddressARB( (const GLubyte *) "glXMakeContextCurrent"      );

    assert(glXCreateContextAttribsARB);
    assert(glXMakeContextCurrentARB);

    const char *displayName = NULL;
    Display* display = XOpenDisplay( displayName );
    assert(display);

    static int visualAttribs[] = { None };
    int numberOfFramebufferConfigurations = 0;
    GLXFBConfig* fbConfigs = glXChooseFBConfig( display, DefaultScreen(display), visualAttribs, &numberOfFramebufferConfigurations );


    int context_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
        GLX_CONTEXT_MINOR_VERSION_ARB, 2,
        GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_DEBUG_BIT_ARB,
        GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None
    };

    GLXContext openGLContext = glXCreateContextAttribsARB( display, fbConfigs[0], 0, True, context_attribs);

    int pbufferAttribs[] = {
        GLX_PBUFFER_WIDTH,  32,
        GLX_PBUFFER_HEIGHT, 32,
        None
    };
    GLXPbuffer pbuffer = glXCreatePbuffer( display, fbConfigs[0], pbufferAttribs );

    // clean up:
    XFree( fbConfigs );
    XSync( display, False );

    if ( !glXMakeContextCurrent( display, pbuffer, pbuffer, openGLContext ) )
    {
        // something went wrong
        assert(0);
    }

    cout << "OffscreenWindow::initWindow complete" << endl;

    return true;
}



static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
};


static const int pbufferWidth = 9;
static const int pbufferHeight = 9;

static const EGLint pbufferAttribs[] = {
      EGL_WIDTH, pbufferWidth,
      EGL_HEIGHT, pbufferHeight,
      EGL_NONE,
};

bool OffscreenWindow::initWindow(){
    // 1. Initialize EGL
      EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

      EGLint major, minor;

      eglInitialize(eglDpy, &major, &minor);

      // 2. Select an appropriate configuration
      EGLint numConfigs;
      EGLConfig eglCfg;

      eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

      // 3. Create a surface
      EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg,
                                                   pbufferAttribs);

      // 4. Bind the API
      eglBindAPI(EGL_OPENGL_API);

      // 5. Create a context and make it current
      EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT,
                                           NULL);

      eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);


      cout << "egl initialized!!" << endl;
      // from now on use your OpenGL context
    return true;
}


