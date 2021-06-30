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
EditorGui editor_gui;

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

void MainMenu::EraseItem(const std::string& menu, const std::string& item)
{
    for (auto& men : menus)
    {
        if (men.name == menu)
        {
            men.items.erase(
                std::remove_if(men.items.begin(), men.items.end(), [item](Item& i) { return i.name == item; }),
                men.items.end());
        }
    }

    // erase empty menus
    menus.erase(std::remove_if(menus.begin(), menus.end(), [](Menu& m) { return m.items.empty(); }), menus.end());
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

std::ostream& operator<<(std::ostream& strm, const MainMenu& menu)
{
    for (auto& m : menu.menus)
    {
        strm << "> " << m.name << std::endl;
        for (auto& i : m.items)
        {
            strm << ">>> " << i.name << std::endl;
        }
    }

    return strm;
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

    layout = std::make_unique<EditorLayoutL>();
}


void EditorLayout::PlaceWindows()
{
    for (auto& windows : initial_layout)
    {
        ImGui::DockBuilderDockWindow(windows.first.c_str(), node_map[windows.second]);
    }
}


EditorLayoutL::EditorLayoutL()
{
    RegisterImguiWindow("Forward Renderer", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("Deferred Renderer", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("Uber Deferred Renderer", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("Clusterer", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("OpenGLWindow", WINDOW_POSITION_LEFT);

    RegisterImguiWindow("Lighting", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("Light Data", WINDOW_POSITION_LEFT_BOTTOM);
    RegisterImguiWindow("VideoEncoder", WINDOW_POSITION_LEFT);

    RegisterImguiWindow("Log", EditorLayoutL::WINDOW_POSITION_BOTTOM);

    RegisterImguiWindow("3DView", WINDOW_POSITION_3DVIEW);

    // Timers
    RegisterImguiWindow("OpenGL Timer", EditorLayoutL::WINDOW_POSITION_BOTTOM);
    RegisterImguiWindow("CUDA Timer", EditorLayoutL::WINDOW_POSITION_BOTTOM);
}


void EditorLayoutL::BuildNodes(int dockspace_id)
{
    ImGui::DockBuilderAddNode(dockspace_id);

    ImGuiID dock_viewport    = dockspace_id;
    ImGuiID dock_left        = ImGui::DockBuilderSplitNode(dock_viewport, ImGuiDir_Left, 0.20f, NULL, &dock_viewport);
    ImGuiID dock_left_bottom = ImGui::DockBuilderSplitNode(dock_left, ImGuiDir_Down, 0.20f, NULL, &dock_left);

    ImGuiID dock_bottom = ImGui::DockBuilderSplitNode(dock_viewport, ImGuiDir_Down, 0.20f, NULL, &dock_viewport);

    node_map = {dock_left, dock_bottom, dock_viewport, dock_left_bottom};
}

EditorLayoutLSplit2x2::EditorLayoutLSplit2x2()
{
    RegisterImguiWindow("Forward Renderer", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("Deferred Renderer", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("Clusterer", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("OpenGLWindow", WINDOW_POSITION_LEFT);

    RegisterImguiWindow("Lighting", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("Light Data", WINDOW_POSITION_LEFT_BOTTOM);
    RegisterImguiWindow("VideoEncoder", WINDOW_POSITION_LEFT);

    RegisterImguiWindow("Log", EditorLayoutL::WINDOW_POSITION_BOTTOM);

    RegisterImguiWindow("3DView", WINDOW_POSITION_MAIN_11);

    // Timers
    RegisterImguiWindow("OpenGL Timer", EditorLayoutL::WINDOW_POSITION_BOTTOM);
    RegisterImguiWindow("CUDA Timer", EditorLayoutL::WINDOW_POSITION_BOTTOM);
}


void EditorLayoutLSplit2x2::BuildNodes(int dockspace_id)
{
    ImGui::DockBuilderAddNode(dockspace_id);


    ImGuiID dock_left, dock_bottom, dock_11, dock_12, dock_21, dock_22;

    ImGuiID dock_viewport    = dockspace_id;
    dock_left                = ImGui::DockBuilderSplitNode(dock_viewport, ImGuiDir_Left, 0.20f, NULL, &dock_viewport);
    ImGuiID dock_left_bottom = ImGui::DockBuilderSplitNode(dock_left, ImGuiDir_Down, 0.20f, NULL, &dock_left);
    dock_bottom              = ImGui::DockBuilderSplitNode(dock_viewport, ImGuiDir_Down, 0.20f, NULL, &dock_viewport);

    ImGuiID dock1, dock2;
    ImGui::DockBuilderSplitNode(dock_viewport, ImGuiDir_Up, 0.50f, &dock1, &dock2);

    ImGui::DockBuilderSplitNode(dock1, ImGuiDir_Left, 0.50f, &dock_11, &dock_12);
    ImGui::DockBuilderSplitNode(dock2, ImGuiDir_Left, 0.50f, &dock_21, &dock_22);


    //    ImGuiID dock_1   = ImGui::DockBuilderSplitNode(dock_viewport, ImGuiDir_Down, 0.50f, NULL, &dock_viewport);



    node_map = {dock_left, dock_bottom, dock_11, dock_12, dock_21, dock_22, dock_left_bottom};
}

EditorLayoutU::EditorLayoutU(bool split_left_right, float left_size, float right_size, float bottom_size,
                             float left_split_size, float right_split_size)
    : split_left_right(split_left_right),
      left_size(left_size),
      right_size(right_size),
      bottom_size(bottom_size),
      left_split_size(left_split_size),
      right_split_size(right_split_size)
{
    RegisterImguiWindow("Forward Renderer", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("Deferred Renderer", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("Clusterer", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("OpenGLWindow", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("VideoEncoder", WINDOW_POSITION_LEFT);

    RegisterImguiWindow("Lighting", WINDOW_POSITION_LEFT);
    RegisterImguiWindow("Light Data", WINDOW_POSITION_LEFT_BOTTOM);

    RegisterImguiWindow("Log", EditorLayoutU::WINDOW_POSITION_BOTTOM);
    RegisterImguiWindow("3DView", WINDOW_POSITION_3DVIEW);

    // Timers
    RegisterImguiWindow("OpenGL Timer", EditorLayoutU::WINDOW_POSITION_BOTTOM);
    RegisterImguiWindow("CUDA Timer", EditorLayoutU::WINDOW_POSITION_BOTTOM);
}
void EditorLayoutU::BuildNodes(int dockspace_id)
{
    ImGui::DockBuilderAddNode(dockspace_id);

    ImGuiID dock_viewport = dockspace_id;

    ImGuiID dock_left   = ImGui::DockBuilderSplitNode(dock_viewport, ImGuiDir_Left, left_size, NULL, &dock_viewport);
    ImGuiID dock_right  = ImGui::DockBuilderSplitNode(dock_viewport, ImGuiDir_Right, right_size, NULL, &dock_viewport);
    ImGuiID dock_bottom = ImGui::DockBuilderSplitNode(dock_viewport, ImGuiDir_Down, bottom_size, NULL, &dock_viewport);

    if (split_left_right)
    {
        ImGuiID dock_left_bottom =
            ImGui::DockBuilderSplitNode(dock_left, ImGuiDir_Down, left_split_size, NULL, &dock_left);
        ImGuiID dock_right_bottom =
            ImGui::DockBuilderSplitNode(dock_right, ImGuiDir_Down, right_split_size, NULL, &dock_right);
        node_map = {dock_left, dock_right, dock_bottom, dock_viewport, dock_left_bottom, dock_right_bottom};
    }
    else
    {
        node_map = {dock_left, dock_right, dock_bottom, dock_viewport};
    }
}

void EditorGui::render(int w, int h)
{
    if (!enabled) return;

    ImGuiViewport* viewport = ImGui::GetMainViewport();

    ImVec2 main_pos  = viewport->Pos;
    ImVec2 main_size = viewport->Size;

    ImGui::SetNextWindowPos(main_pos);
    ImGui::SetNextWindowSize(main_size);

    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_MenuBar;
    flags |= ImGuiWindowFlags_NoDocking;
    flags |=
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    //        flags |= ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));



    ImGui::Begin("Master Window", nullptr, flags);
    ImGui::PopStyleVar();

    main_menu.render();


    static ImGuiID dockspace_id = 1;

    if (reset_work_space)
    {
        ImGui::DockBuilderRemoveNode(dockspace_id);  // Clear out existing layout

        SAIGA_ASSERT(layout);
        layout->BuildNodes(dockspace_id);
        layout->PlaceWindows();

        ImGui::DockBuilderFinish(dockspace_id);

        reset_work_space = false;
    }
    ImGui::DockSpace(dockspace_id, ImVec2(0, 0), ImGuiDockNodeFlags_PassthruCentralNode);



    ImGui::End();
    ImGui::PopStyleVar();
}

}  // namespace Saiga
