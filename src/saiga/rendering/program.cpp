/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/rendering/program.h"
#include "saiga/window/window.h"

namespace Saiga {

Updating::Updating(OpenGLWindow &parent)
    : parentWindow(parent)
{
    parent.setUpdateObject(*this);
}

Rendering::Rendering(Renderer &parent)
    : parentRenderer(parent)
{
    parent.setRenderObject(*this);
}



}
