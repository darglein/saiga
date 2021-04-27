/**
 * Copyright (c) 2021 Darius Rückert
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

    void EraseItem(const std::string& menu, const std::string& item);



    void render();

    // Can be SDL or GLFW keys
    // Will be called automatically when creating a window
    void Keypressed(int key_code);

    // Set to false to disable hotkeys
    bool hotkeys = true;

    static int Height();

    friend SAIGA_CORE_API std::ostream& operator<<(std::ostream& strm, const MainMenu& menu);

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


class SAIGA_CORE_API EditorLayout
{
   public:
    EditorLayout() {}
    virtual ~EditorLayout() {}

    virtual void BuildNodes(int dockspace_id) = 0;

    void PlaceWindows();

    // Maps the windows (by string) to a layout location.
    // When the gui is created (or reset)
    std::vector<std::pair<std::string, int>> initial_layout;


    void RegisterImguiWindow(const std::string& name, int position) { initial_layout.push_back({name, position}); }

   protected:
    std::vector<unsigned int> node_map;
};


// The "L" layout has the gui at the left and at the bottom.
// The 3DViewport ist at the top right
class SAIGA_CORE_API EditorLayoutL : public EditorLayout
{
   public:
    enum
    {
        WINDOW_POSITION_LEFT = 0,
        WINDOW_POSITION_BOTTOM,
        WINDOW_POSITION_3DVIEW,
        WINDOW_POSITION_LEFT_BOTTOM,
    };

    EditorLayoutL();
    void BuildNodes(int dockspace_id) override;
};

// Similar to the L-Layout but the main viewport is divided into 2x2 windows.
class SAIGA_CORE_API EditorLayoutLSplit2x2 : public EditorLayout
{
   public:
    enum
    {
        WINDOW_POSITION_LEFT = 0,
        WINDOW_POSITION_BOTTOM,
        WINDOW_POSITION_MAIN_11,
        WINDOW_POSITION_MAIN_12,
        WINDOW_POSITION_MAIN_21,
        WINDOW_POSITION_MAIN_22,
        WINDOW_POSITION_LEFT_BOTTOM,
    };

    EditorLayoutLSplit2x2();
    void BuildNodes(int dockspace_id) override;
};

class SAIGA_CORE_API EditorLayoutU : public EditorLayout
{
   public:
    enum
    {
        WINDOW_POSITION_LEFT = 0,
        WINDOW_POSITION_RIGHT,
        WINDOW_POSITION_BOTTOM,
        WINDOW_POSITION_3DVIEW,
        // These 2 are only available if we set the split_left_right flag at construction time
        WINDOW_POSITION_LEFT_BOTTOM,
        WINDOW_POSITION_RIGHT_BOTTOM,

    };

    // Splits the left and right column horizontally
    EditorLayoutU(bool split_left_right, float left_size = 0.2, float right_size = 0.2, float bottom_size = 0.2,
                  float left_split_size = 0.5, float right_split_size = 0.5);
    void BuildNodes(int dockspace_id) override;

   private:
    bool split_left_right;
    float left_size, right_size, bottom_size;
    float left_split_size, right_split_size;
};



class SAIGA_CORE_API EditorGui
{
   public:
    EditorGui();
    void render(int w, int h);

    // If enabled the dockspace + the menu bar is rendered
    // Otherwise these elements are not rendered.
    // All imgui-windows that are docked into the dockspace will disappear.
    // Only "floating" windows are then shown
    bool enabled = true;


    void SetLayout(std::unique_ptr<EditorLayout> _layout)
    {
        layout           = std::move(_layout);
        reset_work_space = true;
    }

    // The layout defines how the windows are distributed over that screen
    // The saiga default is 'EditorLayoutL'
    std::unique_ptr<EditorLayout> layout;

   private:
    bool reset_work_space = true;
};

SAIGA_CORE_API extern EditorGui editor_gui;

}  // namespace Saiga
