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

MainMenu::MainMenu() {}

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
    //    if (ImGui::BeginMainMenuBar())
    if (ImGui::BeginMenuBar())
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

        //        ImGui::EndMainMenuBar();
        ImGui::EndMenuBar();
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



EditorGui::EditorGui()
{
    main_menu.AddItem(
        "Saiga", "Editor GUI",
        [this]() {
            enabled          = !enabled;
            reset_work_space = true;
            std::cout << "Set Editor GUI " << enabled << std::endl;
        },
        294, "F5");
    enabled = false;
}

void EditorGui::render(int w, int h)
{
    if (!enabled) return;
    //    return;
    {
        //        ImGui::SetNextWindowPos(ImVec2(400, 0), ImGuiCond_Once);
        //        ImGui::SetNextWindowSize(ImVec2(800, 800), ImGuiCond_Once);

        ImGuiViewport* viewport = ImGui::GetMainViewport();

        ImVec2 main_pos  = viewport->Pos;
        ImVec2 main_size = viewport->Size;

        //        main_pos  = main_pos + ImVec2(0, MainMenu::Height());
        //        main_size = main_size + ImVec2(0, -MainMenu::Height());



        //        ImGui::SetNextWindowPos(viewport->Pos + ImVec2(0, MainMenu::Height()));
        //        ImGui::SetNextWindowSize(viewport->Size + ImVec2(0, -MainMenu::Height()));
        ImGui::SetNextWindowPos(main_pos);
        ImGui::SetNextWindowSize(main_size);

        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

        ImGuiWindowFlags flags = ImGuiWindowFlags_MenuBar;
        flags |= ImGuiWindowFlags_NoDocking;
        flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove;
        flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));



        ImGui::Begin("Master Window", nullptr, flags);
        ImGui::PopStyleVar();



        static ImGuiID dockspace_id = 1;
        // Declare Central dockspace
        ImGui::DockSpace(
            dockspace_id, main_size,
            ImGuiDockNodeFlags_None | ImGuiDockNodeFlags_PassthruCentralNode /*|ImGuiDockNodeFlags_NoResize*/);



        if (reset_work_space)
        {
            ImGui::DockBuilderRemoveNode(dockspace_id);  // Clear out existing layout
            ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_PassthruCentralNode);  // Add empty node
            ImGui::DockBuilderSetNodeSize(dockspace_id, main_size);


            ImGuiID dock_main_id = dockspace_id;  // This variable will track the document node, however we are not
                                                  // using it here as we aren't docking anything into it.
            ImGuiID dock_id_prop = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.20f, NULL, &dock_main_id);
            ImGuiID dock_id_bottom =
                ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.20f, NULL, &dock_main_id);


            ImGui::DockBuilderDockWindow("Log", dock_id_bottom);
            ImGui::DockBuilderDockWindow("Properties", dock_id_prop);
            // ImGui::DockBuilderDockWindow("Mesh", dock_main_id);
            ImGui::DockBuilderDockWindow("3DView", dock_main_id);
            // ImGui::DockBuilderDockWindow("Extra", dock_id_prop);
            ImGui::DockBuilderFinish(dockspace_id);
            reset_work_space = false;
        }


        main_menu.render();

        ImGui::End();
        ImGui::PopStyleVar();
    }

    //    ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_Once);
    ImGui::Begin("Log");
    ImGui::End();

    //    ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_Once);
    ImGui::Begin("Properties");
    ImGui::End();

    //    ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_Once);
    //    ImGui::Begin("Mesh");
    //    ImGui::End();


#if 0
    int bar_height = MainMenu::Height();


    static ImGuiID dockspaceID1 = 1;
    static ImGuiID dockspaceID2 = 2;

    ImGui::SetNextWindowPos(ImVec2(0, bar_height), ImGuiCond_Always);
    ImGui::Begin("Master Window");

    ImGui::TextUnformatted("DockSpace below");

    // Declare Central dockspace
    ImGui::DockSpace(dockspaceID1, ImVec2(0.0f, 800.0f),
                     ImGuiDockNodeFlags_None | ImGuiDockNodeFlags_PassthruCentralNode /*|ImGuiDockNodeFlags_NoResize*/);

    ImGui::TextUnformatted("DockSpace below");
//    ImGui::DockSpace(
//        dockspaceID2, ImVec2(0.0f, 200.0f), ImGuiDockNodeFlags_None | ImGuiDockNodeFlags_PassthruCentralNode
//        /*|ImGuiDockNodeFlags_NoResize*/);

    ImGui::End();



    // ImGui::SetNextWindowDockID(dockspaceID, ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Dockable Window"))
    {
        ImGui::TextUnformatted("Test");
    }
    ImGui::End();

    return;
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
#endif
}

}  // namespace Saiga
