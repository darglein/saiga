/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <chrono>
#include <string>

struct ImGuiContext;

namespace Saiga
{
/**
 * The platform side of ImGui for a window without a window system (the headless OpenGL and Vulkan windows).
 *
 * Turns multi-viewports off (they need OS windows), keeps the clipboard in this process and sets the display size and
 * time step of each frame. No input reaches ImGui unless the application adds it to ImGui's input queue
 * (ImGuiIO::AddMousePosEvent, AddKeyEvent, ...).
 *
 * Create it after the ImGui context and destroy it before the context.
 */
class SAIGA_CORE_API ImGuiHeadlessPlatform
{
   public:
    ImGuiHeadlessPlatform();
    ~ImGuiHeadlessPlatform();

    ImGuiHeadlessPlatform(const ImGuiHeadlessPlatform&)            = delete;
    ImGuiHeadlessPlatform& operator=(const ImGuiHeadlessPlatform&) = delete;

    // Before ImGui::NewFrame(): the display size in pixels and the time since the last frame.
    void NewFrame(int width, int height);

   private:
    std::string clipboard;
    std::chrono::steady_clock::time_point last_frame;
    bool first_frame = true;
    // The context whose callbacks this object set.
    ImGuiContext* context = nullptr;

    static const char* GetClipboardText(ImGuiContext* ctx);
    static void SetClipboardText(ImGuiContext* ctx, const char* text);
};

}  // namespace Saiga
