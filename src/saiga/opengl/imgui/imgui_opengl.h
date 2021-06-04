/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_renderer.h"
#include "saiga/core/imgui/imgui_timer_system.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/query/gpuTimer.h"
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



// A system which tracks OpenGL time on a per frame basis.
// The GLRenderer has one object which should be used by every subprocess.
class GLTimerSystem : public TimerSystem
{
   public:
    GLTimerSystem() : TimerSystem("OpenGL Timer") {}

   protected:
    virtual std::unique_ptr<TimestampTimer> CreateTimer() override
    {
        auto timer = std::make_unique<MultiFrameOpenGLTimer>();
        timer->create();
        return timer;
    }
};


}  // namespace Saiga

namespace ImGui
{
SAIGA_OPENGL_API void Texture(Saiga::TextureBase* texture, const ImVec2& size, bool flip_y,
                              const ImVec4& tint_col   = ImVec4(1, 1, 1, 1),
                              const ImVec4& border_col = ImVec4(0, 0, 0, 0));

}
