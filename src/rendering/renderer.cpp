#include "saiga/rendering/renderer.h"
#include "saiga/window/window.h"

Program::Program(OpenGLWindow *parent) : parentWindow(parent)
{
    parentWindow->setProgram(this);
}
