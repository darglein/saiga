/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Interfaces.h"

#include "internal/noGraphicsAPI.h"

#include "WindowBase.h"

namespace Saiga
{
Updating::Updating(WindowBase& parent) : parentWindow(parent)
{
    parent.setUpdateObject(*this);
}

RenderingInterfaceBase::RenderingInterfaceBase(RendererBase& parent) : parentRenderer(parent)
{
    parent.setRenderObject(*this);
}


}  // namespace Saiga
