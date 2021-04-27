/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/imgui/imgui_saiga.h"

struct ImDrawData;

namespace Saiga
{
class SAIGA_CORE_API ImGuiRenderer
{
   public:
    ImGuiRenderer(ImGuiParameters params);
    virtual ~ImGuiRenderer();


    virtual void beginFrame() = 0;
    virtual void endFrame();

    //    void render();

   protected:
    //    virtual void renderDrawLists(ImDrawData* draw_data) = 0;
};

}  // namespace Saiga
