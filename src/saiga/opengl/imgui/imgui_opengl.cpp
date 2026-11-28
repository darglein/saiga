/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imgui_opengl.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_internal.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/statistics.h"

#include <algorithm>
#include <string>

namespace Saiga
{
ImTextureID ImGuiOpenGLRenderer::addTexture(const TextureBase& texture)
{
    SAIGA_ASSERT(texture.getId() != 0, "ImGuiOpenGLRenderer::addTexture: the texture is not created");
    const ImTextureID id = static_cast<ImTextureID>(texture.getId());
    textures[id]++;
    return id;
}

void ImGuiOpenGLRenderer::removeTexture(ImTextureID id)
{
    auto it = textures.find(id);
    SAIGA_ASSERT(it != textures.end(), "ImGuiOpenGLRenderer::removeTexture: unknown texture");
    if (it == textures.end()) return;
    if (--it->second == 0) textures.erase(it);
}

std::vector<ImTextureID> ImGuiOpenGLRenderer::unknownTextures() const
{
    std::vector<ImTextureID> unknown;
    if (ImGui::GetCurrentContext() != imguiContext()) return unknown;
    const ImTextureID font = ImGui::GetIO().Fonts->TexRef.GetTexID();
    for (ImGuiViewport* viewport : ImGui::GetPlatformIO().Viewports)
    {
        const ImDrawData* data = viewport->DrawData;
        if (!data || !data->Valid) continue;
        for (const ImDrawList* list : data->CmdLists)
        {
            for (const ImDrawCmd& cmd : list->CmdBuffer)
            {
                if (cmd.UserCallback) continue;
                const ImTextureID id = cmd.GetTexID();
                if (id == font || textures.count(id) > 0) continue;
                if (std::find(unknown.begin(), unknown.end(), id) == unknown.end()) unknown.push_back(id);
            }
        }
    }
    return unknown;
}

void ImGuiOpenGLRenderer::checkTextures() const
{
    if (!requireAddedTextures) return;
    const std::vector<ImTextureID> unknown = unknownTextures();
    SAIGA_ASSERT(unknown.empty(), "ImGui draws texture " + std::to_string(unknown.empty() ? 0 : unknown.front()) +
                                      ", which was not added with ImGuiOpenGLRenderer::addTexture");
}

}  // namespace Saiga

void ImGui::Texture(Saiga::TextureBase* texture, const ImVec2& size, bool flip_y, const ImVec4& tint_col,
                    const ImVec4& border_col)
{
    size_t tid     = texture->getId();
    ImTextureID id = (ImTextureID)tid;

    if (flip_y)
    {
        Image(id, size, ImVec2(0, 1), ImVec2(1, 0), tint_col, border_col);
    }
    else
    {
        Image(id, size, ImVec2(0, 0), ImVec2(1, 1), tint_col, border_col);
    }
}

void ImGui::TextureRotate90(Saiga::TextureBase* texture, const ImVec2& size, bool flip_y, bool right_rotate,
                            const ImVec4& tint_col, const ImVec4& border_col)
{
    size_t tid = texture->getId();

    ImTextureID id = (ImTextureID)tid;

    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems) return;

    ImRect bb(window->DC.CursorPos, window->DC.CursorPos + size);
    if (border_col.w > 0.0f)
    {
        bb.Max.x += 2;
        bb.Max.y += 2;
        ItemSize(bb);
    }
    if (!ItemAdd(bb, 0)) return;

    ImVec2 size_u = ImVec2((bb.Max.x - bb.Min.x), (bb.Max.y - bb.Min.y));
    ImVec2 pos[4] = {
        bb.Min + ImVec2(0, 0),                // ImRotate(ImVec2(-size_u.x * 0.5f, -size_u.y * 0.5f), cos_a, sin_a), //
        bb.Min + ImVec2(size_u.x, 0),         // ImRotate(ImVec2(+size_u.x * 0.5f, -size_u.y * 0.5f), cos_a, sin_a), //
        bb.Min + ImVec2(size_u.x, size_u.y),  // ImRotate(ImVec2(+size_u.x * 0.5f, +size_u.y * 0.5f), cos_a, sin_a), //
        bb.Min + ImVec2(0, size_u.y)          // ImRotate(ImVec2(-size_u.x * 0.5f, +size_u.y * 0.5f), cos_a, sin_a) //
    };
    ImVec2 uvs[4] = {ImVec2(1, 0), ImVec2(1, 1), ImVec2(0, 1), ImVec2(0, 0)};
    if (right_rotate)
    {
        uvs[0] = ImVec2(0, 1);
        uvs[1] = ImVec2(0, 0);
        uvs[2] = ImVec2(1, 0);
        uvs[3] = ImVec2(1, 1);
    }

    // window->DrawList->AddImage(id, bb.Min + ImVec2(1, 1), bb.Max + ImVec2(-1, -1), uv0, uv1, GetColorU32(tint_col));
    window->DrawList->AddImageQuad(id, pos[0], pos[1], pos[2], pos[3], uvs[0], uvs[1], uvs[2], uvs[3], IM_COL32_WHITE);
}

void ImGui::Texture(Saiga::TextureBase* texture, const ImVec2& size, ImVec2 uv0, ImVec2 uv1, const ImVec4& tint_col,
                    const ImVec4& border_col)
{
    size_t tid     = texture->getId();
    ImTextureID id = (ImTextureID)tid;

    Image(id, size, uv0, uv1, tint_col, border_col);
}
