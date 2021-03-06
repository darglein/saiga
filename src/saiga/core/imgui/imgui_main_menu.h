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

    // Controlls if the menu bar is rendered.
    // Hotkeys are still accepted (see flag below to disable them)
    bool visible = true;

    // Set to false to disable hotkeys
    bool hotkeys = true;

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

}  // namespace Saiga
