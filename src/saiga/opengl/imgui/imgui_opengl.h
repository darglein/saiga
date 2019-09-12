/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_renderer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/shader/all.h"
#include "saiga/opengl/texture/Texture.h"

namespace Saiga
{
class SAIGA_OPENGL_API ImGui_GL_Renderer : public ImGuiRenderer
{
   public:
    ImGui_GL_Renderer(const ImGuiParameters& params);
    virtual ~ImGui_GL_Renderer();

    void render();

   protected:
    std::shared_ptr<Shader> shader;
    std::shared_ptr<Texture> texture;
    IndexedVertexBuffer<ImDrawVert, ImDrawIdx> buffer;

    virtual void renderDrawLists(ImDrawData* draw_data);
};

}  // namespace Saiga
