#pragma once

#include <saiga/config.h>
#include "saiga/window/window.h"



class SAIGA_GLOBAL OffscreenWindow : public OpenGLWindow{
public:

protected:

    virtual bool initInput() { return true; }
    virtual void checkEvents() {}
    virtual void swapBuffers() {}
    virtual void freeContext() {}

    virtual bool initWindow() override;
public:

    OffscreenWindow(WindowParameters windowParameters);
};

