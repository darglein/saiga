#include "saiga/core/imgui/imgui_main_menu.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_internal.h"
#include "saiga/core/math/random.h"
#include "saiga/core/util/color.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include <fstream>

#include "imgui_saiga.h"

namespace Saiga
{
MainMenu main_menu;

MainMenu::MainMenu()
{
    AddItem(
        "Saiga", "Menu Bar", [this]() { visible = !visible; }, 294, "F5");
}

void MainMenu::AddItem(const std::string& menu, const std::string& item, MainMenu::MenuFunction function, int shortcut,
                       const std::string& shortcut_name)
{
    Item new_item;
    new_item.name          = item;
    new_item.function      = function;
    new_item.shortcut      = shortcut;
    new_item.shortcut_name = shortcut_name;

    // search for menu and insert
    for (auto& men : menus)
    {
        if (men.name == menu)
        {
            men.items.push_back(new_item);
            return;
        }
    }

    // create new menu
    Menu men;
    men.name = menu;
    men.items.push_back(new_item);
    menus.push_back(men);
}

void MainMenu::render()
{
    if (!visible) return;
    if (ImGui::BeginMainMenuBar())
    {
        for (auto& men : menus)
        {
            if (ImGui::BeginMenu(men.name.c_str()))
            {
                for (auto& item : men.items)
                {
                    if (ImGui::MenuItem(item.name.c_str(), item.shortcut_name.c_str()))
                    {
                        item.function();
                    }
                }
                ImGui::EndMenu();
            }
        }

        ImGui::EndMainMenuBar();
    }
}

void MainMenu::Keypressed(int key_code)
{
    if (!hotkeys) return;
    // Linear search. Maybe create hashmap in the future
    for (auto& men : menus)
    {
        for (auto& item : men.items)
        {
            if (item.shortcut == key_code)
            {
                item.function();
            }
        }
    }
}

int MainMenu::Height()
{
    return ImGui::GetFrameHeight();
}

bool Splitter(bool split_vertically, float thickness, float* size1, float* size2, float min_size1, float min_size2,
              float splitter_long_axis_size = -1.0f)
{
    using namespace ImGui;
    ImGuiContext& g     = *GImGui;
    ImGuiWindow* window = g.CurrentWindow;
    ImGuiID id          = window->GetID("##Splitter");
    ImRect bb;
    bb.Min = window->DC.CursorPos + (split_vertically ? ImVec2(*size1, 0.0f) : ImVec2(0.0f, *size1));
    bb.Max = bb.Min + CalcItemSize(split_vertically ? ImVec2(thickness, splitter_long_axis_size)
                                                    : ImVec2(splitter_long_axis_size, thickness),
                                   0.0f, 0.0f);

    return ImGui::SplitterBehavior(bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, size1, size2, min_size1,
                                   min_size2, 0.0f);
}



void EditorGui::render(int w, int h)
{
    int bar_height = MainMenu::Height();
    ImGui::SetNextWindowPos(ImVec2(0, bar_height), ImGuiCond_Always);
    int height = h - bar_height;
    ImGui::SetNextWindowSize(ImVec2(400, height), ImGuiCond_Always);
    ImGui::Begin("editor", nullptr,
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysUseWindowPadding);

    height = ImGui::GetWindowHeight() - ImGui::GetStyle().WindowPadding.y * 2;

    static float alpha = 0.5;

    float sz1 = alpha * height;
    float sz2 = height - sz1;

    float thickness = 8;
    Splitter(false, 8.0f, &sz1, &sz2, 8, 8, h);


    alpha = sz1 / height;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(1, 0, 0, 1));


    ImGui::BeginChild(1, ImVec2(0, sz1), true);
    ImGui::EndChild();

    ImGui::GetCurrentWindow()->DC.CursorPos.y += 1 + thickness - ImGui::GetStyle().ItemSpacing.y;

    //    ImGui::Dummy(ImVec2(10, 1));


    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 10);
    ImGui::BeginChild(2, ImVec2(0, sz2), true);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();


    ImGui::End();
}

}  // namespace Saiga
