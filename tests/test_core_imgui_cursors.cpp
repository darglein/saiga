/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Guards the local Dear ImGui patches in src/saiga/core/imgui/patches/. An ImGui update that replaces the vendored
// sources without reapplying the patches makes these tests fail.
#include "saiga/core/imgui/imgui.h"

#include <functional>
#include <gtest/gtest.h>

namespace
{
struct ItemRect
{
    ImVec2 min;
    ImVec2 max;
};

ItemRect last_item_rect()
{
    return {ImGui::GetItemRectMin(), ImGui::GetItemRectMax()};
}

class ImGuiCursor : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        context        = ImGui::CreateContext();
        ImGuiIO& io    = ImGui::GetIO();
        io.DisplaySize = ImVec2(800, 600);
        io.DeltaTime   = 1.0f / 60.0f;
        io.IniFilename = nullptr;
        io.Fonts->Build();
    }

    void TearDown() override { ImGui::DestroyContext(context); }

    // `widget` submits the widget under test and returns its rect.
    ImGuiMouseCursor cursor_while_hovering(const std::function<ItemRect()>& widget)
    {
        ItemRect rect{};
        // The first frames lay out the window and widget (tab bars settle their widths one frame late).
        for (int i = 0; i < 3; ++i) rect = frame(widget);

        ImVec2 center((rect.min.x + rect.max.x) * 0.5f, (rect.min.y + rect.max.y) * 0.5f);
        ImGui::GetIO().AddMousePosEvent(center.x, center.y);
        // Hovering takes effect one frame after the mouse moves.
        for (int i = 0; i < 2; ++i) frame(widget);
        return ImGui::GetMouseCursor();
    }

   private:
    ItemRect frame(const std::function<ItemRect()>& widget)
    {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(400, 400));
        ImGui::Begin("Test", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove);
        ItemRect rect = widget();
        ImGui::End();
        ImGui::EndFrame();
        return rect;
    }

    ImGuiContext* context = nullptr;
};

TEST_F(ImGuiCursor, Button)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ImGui::Button("Button");
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, SmallButton)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ImGui::SmallButton("Small");
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, ImageButton)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ImGui::ImageButton("Image", ImTextureID(1), ImVec2(32, 32));
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, Checkbox)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      static bool value = false;
                      ImGui::Checkbox("Checkbox", &value);
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, RadioButton)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ImGui::RadioButton("Radio", false);
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, Combo)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      if (ImGui::BeginCombo("Combo", "Preview")) ImGui::EndCombo();
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, TreeNode)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      if (ImGui::TreeNode("Node")) ImGui::TreePop();
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, CollapsingHeader)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ImGui::CollapsingHeader("Header");
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, Selectable)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ImGui::Selectable("Selectable");
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, MenuItem)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ImGui::MenuItem("Menu item");
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, Tab)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ItemRect rect{};
                      if (ImGui::BeginTabBar("Tabs"))
                      {
                          if (ImGui::BeginTabItem("Tab"))
                          {
                              rect = last_item_rect();
                              ImGui::EndTabItem();
                          }
                          ImGui::EndTabBar();
                      }
                      return rect;
                  }),
              ImGuiMouseCursor_Hand);
}

TEST_F(ImGuiCursor, InvisibleButtonKeepsArrow)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ImGui::InvisibleButton("Canvas", ImVec2(100, 100));
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Arrow);
}

TEST_F(ImGuiCursor, DisabledButtonKeepsArrow)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ImGui::BeginDisabled();
                      ImGui::Button("Disabled");
                      ImGui::EndDisabled();
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_Arrow);
}

TEST_F(ImGuiCursor, LaterSetMouseCursorWins)
{
    EXPECT_EQ(cursor_while_hovering(
                  []
                  {
                      ImGui::Button("Splitter");
                      if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
                      return last_item_rect();
                  }),
              ImGuiMouseCursor_ResizeEW);
}

TEST_F(ImGuiCursor, CrosshairExists)
{
    EXPECT_LT(ImGuiMouseCursor_Crosshair, ImGuiMouseCursor_COUNT);
}
}  // namespace
