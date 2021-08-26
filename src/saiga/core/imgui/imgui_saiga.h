/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/imgui/imgui_main_menu.h"
#include "saiga/core/math/math.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Align.h"
#include "saiga/core/util/table.h"

#include <vector>

struct ImDrawList;
namespace ImGui
{
class SAIGA_CORE_API IMConsole : public std::ostream, public std::streambuf
{
   public:
    IMConsole(const std::string& name = "Console", const Saiga::ivec2& position = {0, 0},
              const Saiga::ivec2& size = {500, 250}, bool write_to_cout = false);

    void render();
    void BeginWindow();
    void EndWindow();
    void RenderTextArea();

    // additionally log to the given file.
    // Note: calling this method will clear the exsisting content!
    void setOutputFile(const std::string& file);

    // additonally write to std::cout (default = false)
    void setWriteToCout(bool b) { writeToCout = b; }

    // derived
    int overflow(int c) override;

    std::string name;
    Saiga::ivec2 position, size;
    bool should_render = true;

   private:
    bool scrollDownAtNextRender = true;
    bool writeToCout            = false;
    bool scrollToBottom         = true;
    std::string data;
    std::shared_ptr<std::ofstream> outFile;
    std::streambuf* cout_buf;
};

// An ImGUi table window where you can add new lines like in Saiga::Table.
// Example:
//
//   ImGui::IMTable test_table("Fancy Table", {10, 10}, {"First", "Second"});
//   test_table << 2346346 << 1424;
//   test_table << 23 << 1424;
//
//   test_table.Render();
//
class SAIGA_CORE_API IMTable : public Saiga::Table
{
   public:
    IMTable(const std::string& name, const std::vector<int>& colum_width, const std::vector<std::string>& colum_name);
    void Render();

   public:
    std::string header;
    ImGui::IMConsole console;
};


class SAIGA_CORE_API Graph
{
   public:
    Graph(const std::string& name = "Graph", int numValues = 80);
    virtual ~Graph() {}
    void addValue(float t);
    void renderImGui();
    void SetName(const std::string& n) { name = n; }

   protected:
    virtual void renderImGuiDerived();
    std::string name;
    int numValues;

    float maxValue   = 0;
    float lastValue  = 0;
    float average    = 0;
    int currentIndex = 0;
    int r;
    std::vector<float> values;
};


class SAIGA_CORE_API TimeGraph : public Graph
{
   public:
    TimeGraph(const std::string& name = "Time", int numValues = 80);
    void addTime(float t);

   protected:
    virtual void renderImGuiDerived();
    float hzExp = 0;
    Saiga::Timer timer;
};


class SAIGA_CORE_API HzTimeGraph : public Graph
{
   public:
    HzTimeGraph(const std::string& name = "Hz", int numValues = 80);
    void addTime();

   protected:
    virtual void renderImGuiDerived();
    float hzExp = 0;
    Saiga::Timer timer;
};



class SAIGA_CORE_API ColoredBar
{
   public:
    using vec4 = Saiga::vec4;
    using vec2 = Saiga::vec2;
    struct BarColor
    {
        vec4 fill;
        vec4 outline;
    };

   private:
    vec2 m_size;
    BarColor m_back_color;
    bool m_auto_size;
    uint32_t m_rows;
    std::vector<vec2> m_lastCorner;
    ImDrawList* m_lastDrawList;
    float m_rounding;
    int m_rounding_corners;

   private:
    void DrawOutlinedRect(const vec2& begin, const vec2& end, const BarColor& color);
    void DrawRect(const vec2& begin, const vec2& end, const BarColor& color);

   public:
    ColoredBar(vec2 size, BarColor background, bool auto_size = false, uint32_t rows = 1, float rounding = 0.0f,
               int rounding_corners = 0)
        : m_size(size),
          m_back_color(background),
          m_auto_size(auto_size),
          m_rows(rows),
          m_lastCorner(rows),
          m_lastDrawList(nullptr),
          m_rounding(rounding),
          m_rounding_corners(rounding_corners)
    {
        SAIGA_ASSERT(rows >= 1, "Must have a positive number of rows");
    }
    void renderBackground();
    void renderArea(float begin, float end, const BarColor& color, bool outline = true);
};

/**
 * A helper function that checks if a context is present and
 * if ImGui wants to capture the mouse inputs.
 *
 * A typical use-case is to update the camera only if no ImGui widgets are active:
 *
 *   if (renderer->use_keyboard_input_in_3dview)
 *   {
 *       camera.update(dt);
 *   }
 *   if (renderer->use_mouse_input_in_3dview)
 *   {
 *       camera.interpolate(dt, 0);
 *   }
 *
 */
SAIGA_CORE_API bool captureMouse();
SAIGA_CORE_API bool captureKeyboard();


// Similar to ImGui::Combo but with C++ strings instead of a char array.
SAIGA_CORE_API bool StringCombo(const char* label, int* current_item, const std::vector<std::string>& data);


}  // namespace ImGui

namespace Saiga
{
enum class ImGuiTheme : int
{
    SAIGA = 0,
    IMGUI,  // imgui default theme
};

struct SAIGA_CORE_API ImGuiParameters
{
    // imgui parameters
    bool enable          = true;
    std::string font     = "SourceSansPro-Regular.ttf";
    int fontSize         = 18;
    float fontBrightness = 2;
    ImGuiTheme theme     = ImGuiTheme::SAIGA;

    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};

SAIGA_CORE_API void initImGui(const ImGuiParameters& params);

// The main console from saiga
SAIGA_CORE_API extern ImGui::IMConsole console;

}  // namespace Saiga
