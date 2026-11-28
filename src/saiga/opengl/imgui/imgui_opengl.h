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

#include <vector>

#include <unordered_map>

namespace Saiga
{
// The base of the ImGui renderers of the OpenGL windows.
//
// ImGui draws a texture by the ImTextureID handed to it (ImGui::Image, ImDrawList::AddImage). Add a texture to get its
// ID and remove it before the texture is deleted, as with Vulkan::ImGuiVulkanRenderer::addTexture: code written that
// way works with both renderers. In OpenGL the ID is the texture's name, so adding only counts the references.
class SAIGA_OPENGL_API ImGuiOpenGLRenderer : public ImGuiRenderer
{
   public:
    using ImGuiRenderer::ImGuiRenderer;

    // The ID for ImGui, valid until removeTexture. A texture may be added several times; each needs its removeTexture.
    ImTextureID addTexture(const TextureBase& texture);
    void removeTexture(ImTextureID id);
    // The textures added and not removed (without the font), each counted once.
    size_t textureCount() const { return textures.size(); }

    // The IDs ImGui's draw commands use (in all viewports) that were neither added nor are the font atlas. The Vulkan
    // renderer skips those commands, so this finds code that hands texture names to ImGui directly.
    std::vector<ImTextureID> unknownTextures() const;
    // Asserts after each frame that unknownTextures() is empty.
    bool requireAddedTextures = false;

   protected:
    // Checks the textures with requireAddedTextures; the renderers call it in render().
    void checkTextures() const;

   private:
    // The references per added texture.
    std::unordered_map<ImTextureID, int> textures;
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
// These hand the texture's name to ImGui directly, without ImGuiOpenGLRenderer::addTexture.
SAIGA_OPENGL_API void Texture(Saiga::TextureBase* texture, const ImVec2& size, bool flip_y,
                              const ImVec4& tint_col   = ImVec4(1, 1, 1, 1),
                              const ImVec4& border_col = ImVec4(0, 0, 0, 0));

SAIGA_OPENGL_API void Texture(Saiga::TextureBase* texture, const ImVec2& size, ImVec2 uv0 = ImVec2(0, 0),
                              ImVec2 uv1 = ImVec2(1, 1), const ImVec4& tint_col = ImVec4(1, 1, 1, 1),
                              const ImVec4& border_col = ImVec4(0, 0, 0, 0));

SAIGA_OPENGL_API void TextureRotate90(Saiga::TextureBase* texture, const ImVec2& size, bool flip_y, bool right_rotate,
                                      const ImVec4& tint_col   = ImVec4(1, 1, 1, 1),
                                      const ImVec4& border_col = ImVec4(0, 0, 0, 0));

}  // namespace ImGui
