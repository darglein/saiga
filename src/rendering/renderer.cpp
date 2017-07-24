/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/rendering/renderer.h"
#include "saiga/window/window.h"

namespace Saiga {

Program::Program(OpenGLWindow *parent) : parentWindow(parent)
{
    parentWindow->setProgram(this);
}

}
