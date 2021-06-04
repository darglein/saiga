/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Interfaces.h"

#include "saiga/core/imgui/imgui_main_menu.h"

#include "internal/noGraphicsAPI.h"

#include "WindowBase.h"
namespace Saiga
{
RendererBase::RendererBase()
{
    main_menu.AddItem(
        "Saiga", "Renderer", [this]() { should_render_imgui = !should_render_imgui; }, 296, "F7");
}

}  // namespace Saiga
