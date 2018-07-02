/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Interfaces.h"
#include "saiga/window/WindowBase.h"


namespace Saiga {

Updating::Updating(WindowBase &parent)
    : parentWindow(parent)
{
    parent.setUpdateObject(*this);
}

RenderingBase::RenderingBase(RendererBase &parent)
    : parentRenderer(parent)
{
    parent.setRenderObject(*this);
}


}
