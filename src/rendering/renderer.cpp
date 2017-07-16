#include "saiga/rendering/renderer.h"
#include "saiga/window/window.h"

namespace Saiga {

Program::Program(OpenGLWindow *parent) : parentWindow(parent)
{
    parentWindow->setProgram(this);
}

}
