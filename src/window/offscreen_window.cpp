
#include "saiga/window/offscreen_window.h"
#include "saiga/rendering/deferred_renderer.h"

#include <GL/glx.h>
#include <GL/gl.h>

#include "saiga/util/assert.h"

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
typedef Bool (*glXMakeContextCurrentARBProc)(Display*, GLXDrawable, GLXDrawable, GLXContext);
static glXCreateContextAttribsARBProc glXCreateContextAttribsARB = NULL;
static glXMakeContextCurrentARBProc   glXMakeContextCurrentARB   = NULL;


OffscreenWindow::OffscreenWindow(WindowParameters windowParameters):OpenGLWindow(windowParameters)
{
}

bool OffscreenWindow::initWindow()
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




