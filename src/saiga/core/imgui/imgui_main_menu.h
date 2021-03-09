/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Align.h"
#include "saiga/core/util/table.h"

#include <functional>
#include <vector>


namespace Saiga
{
class SAIGA_CORE_API MainMenu
{
   public:
    using MenuFunction = std::function<void(void)>;


    MainMenu();
    void AddItem(const std::string& menu, const std::string& item, MenuFunction function, int shortcut = -1,
                 const std::string& shortcut_name = "");


    void render();

    // Can be SDL or GLFW keys
    // Will be called automatically when creating a window
    void Keypressed(int key_code);

    // Set to false to disable hotkeys
    bool hotkeys = true;

    static int Height();

   private:
    struct Item
    {
        std::string name;

        MenuFunction function;

        // -1 == no shortcut
        //
        int shortcut = -1;
        std::string shortcut_name;
    };

    struct Menu
    {
        std::string name;
        std::vector<Item> items;
    };

    std::vector<Menu> menus;
};

SAIGA_CORE_API extern MainMenu main_menu;



class SAIGA_CORE_API EditorGui
{
   public:
    enum EditorLayout
    {
        WINDOW_POSITION_SYSTEM,
        WINDOW_POSITION_DETAILS,
        WINDOW_POSITION_LOG,
        WINDOW_POSITION_3DVIEW,
    };

    EditorGui();
    void render(int w, int h);

    // If enabled the dockspace + the menu bar is rendered
    // Otherwise these elements are not rendered.
    // All imgui-windows that are docked into the dockspace will disappear.
    // Only "floating" windows are then shown
    bool enabled = true;

   private:
    bool reset_work_space = true;

    // Maps the windows (by string) to a layout location.
    // When the gui is created (or reset)
    std::vector<std::pair<std::string, EditorLayout>> initial_layout;
};

inline EditorGui editor_gui;

}  // namespace Saiga
