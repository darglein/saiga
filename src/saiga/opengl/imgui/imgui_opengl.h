/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/imgui/imgui.h"
#include "saiga/imgui/imgui_renderer.h"
#include "saiga/opengl/shader/all.h"
#include <saiga/opengl/texture/texture.h>
#include <saiga/opengl/indexedVertexBuffer.h>

namespace Saiga {


class SAIGA_GLOBAL ImGui_GL_Renderer : public ImGuiRenderer
{
public:
    ImGui_GL_Renderer(std::string font, float fontSize = 15.0f);
    ~ImGui_GL_Renderer();

protected:

    std::shared_ptr<Shader> shader;
    std::shared_ptr<Texture> texture;
    IndexedVertexBuffer<ImDrawVert,ImDrawIdx> buffer;

    virtual void renderDrawLists(ImDrawData *draw_data) override;
};

}
