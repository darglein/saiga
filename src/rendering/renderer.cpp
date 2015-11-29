#include "saiga/rendering/renderer.h"
#include "saiga/window/window.h"

Program::Program(Window *parent) : parentWindow(parent)
{
    parentWindow->setProgram(this);
}
