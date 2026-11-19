/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imgui_headless_platform.h"

#include "saiga/core/imgui/imgui.h"

#include <algorithm>

namespace Saiga
{
ImGuiHeadlessPlatform::ImGuiHeadlessPlatform() : context(ImGui::GetCurrentContext())
{
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags &= ~ImGuiConfigFlags_ViewportsEnable;
    io.BackendPlatformName = "saiga_headless";

    ImGuiPlatformIO& platform_io            = ImGui::GetPlatformIO();
    platform_io.Platform_ClipboardUserData  = this;
    platform_io.Platform_GetClipboardTextFn = GetClipboardText;
    platform_io.Platform_SetClipboardTextFn = SetClipboardText;
}

ImGuiHeadlessPlatform::~ImGuiHeadlessPlatform()
{
    ImGuiContext* previous = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(context);
    ImGuiPlatformIO& platform_io            = ImGui::GetPlatformIO();
    platform_io.Platform_ClipboardUserData  = nullptr;
    platform_io.Platform_GetClipboardTextFn = nullptr;
    platform_io.Platform_SetClipboardTextFn = nullptr;
    ImGui::GetIO().BackendPlatformName      = nullptr;
    ImGui::SetCurrentContext(previous);
}

void ImGuiHeadlessPlatform::NewFrame(int width, int height)
{
    ImGuiIO& io                = ImGui::GetIO();
    io.DisplaySize             = ImVec2((float)width, (float)height);
    io.DisplayFramebufferScale = ImVec2(1, 1);

    auto now = std::chrono::steady_clock::now();
    float dt = first_frame ? 1.0f / 60.0f : std::chrono::duration<float>(now - last_frame).count();
    // ImGui requires a positive time step.
    io.DeltaTime = std::max(dt, 1e-5f);
    last_frame   = now;
    first_frame  = false;
}

const char* ImGuiHeadlessPlatform::GetClipboardText(ImGuiContext*)
{
    auto* self = static_cast<ImGuiHeadlessPlatform*>(ImGui::GetPlatformIO().Platform_ClipboardUserData);
    return self ? self->clipboard.c_str() : nullptr;
}

void ImGuiHeadlessPlatform::SetClipboardText(ImGuiContext*, const char* text)
{
    auto* self = static_cast<ImGuiHeadlessPlatform*>(ImGui::GetPlatformIO().Platform_ClipboardUserData);
    if (self) self->clipboard = text ? text : "";
}

}  // namespace Saiga
