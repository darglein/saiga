/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imgui_opengl.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_internal.h"
#include "saiga/core/util/statistics.h"

namespace Saiga
{



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
