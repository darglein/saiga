#include "imgui_saiga.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/core/util/color.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/core/util/statistics.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include <fstream>


namespace ImGui
{
Graph::Graph(const std::string& name, int numValues) : name(name), numValues(numValues), values(numValues, 0)
{
    r = Saiga::Random::rand();
}

void Graph::addValue(float t)
{
    maxValue             = std::max(t, maxValue);
    lastValue            = t;
    values[currentIndex] = t;
    currentIndex         = (currentIndex + 1) % numValues;
    float alpha          = 0.1;
    average              = (1 - alpha) * average + alpha * t;
}

void Graph::renderImGui()
{
    ImGui::PushID(r);

    renderImGuiDerived();
    ImGui::PlotLines("", values.data(), numValues, currentIndex, ("avg " + Saiga::to_string(average)).c_str(), 0,
                     maxValue, ImVec2(std::max(numValues, 250), 80));
    ImGui::SameLine();
    if (ImGui::Button("R"))
    {
        maxValue = 0;
        for (auto v : values) maxValue = std::max(v, maxValue);
    }

    ImGui::PopID();
}

void Graph::renderImGuiDerived()
{
    ImGui::Text("%s", name.c_str());
}


TimeGraph::TimeGraph(const std::string& name, int numValues) : Graph(name, numValues)
{
    timer.start();
}

void TimeGraph::addTime(float t)
{
    timer.stop();

    addValue(t);

    float alpha = 0.1;
    hzExp       = (1 - alpha) * hzExp + alpha * timer.getTimeMS();
    timer.start();
}

void TimeGraph::renderImGuiDerived()
{
    ImGui::Text("%s Time: %fms Hz: %f", name.c_str(), lastValue, 1000.0f / hzExp);
}


HzTimeGraph::HzTimeGraph(const std::string& name, int numValues) : Graph(name, numValues)
{
    timer.start();
}

void HzTimeGraph::addTime()
{
    timer.stop();

    float t = timer.getTimeMS();
    addValue(t);

    float alpha = 0.1;
    hzExp       = (1 - alpha) * hzExp + alpha * t;
    timer.start();
}

void HzTimeGraph::renderImGuiDerived()
{
    ImGui::Text("%s Time: %fms Hz: %f", name.c_str(), lastValue, 1000.0f / hzExp);
}


void ColoredBar::renderBackground()
{
    m_lastDrawList = ImGui::GetWindowDrawList();

    if (m_auto_size)
    {
        m_size[0] = ImGui::GetContentRegionAvailWidth();
    }

    for (uint32_t i = 0; i < m_rows; ++i)
    {
        m_lastCorner[i] = ImGui::GetCursorScreenPos();
        DrawOutlinedRect(m_lastCorner[i], m_lastCorner[i] + m_size, m_back_color);
        ImGui::Dummy(m_size);
    }
}



void ColoredBar::renderArea(float begin, float end, const ColoredBar::BarColor& color, bool outline)
{
    SAIGA_ASSERT(m_lastDrawList, "renderBackground() was not called before renderArea()");

    const float factor = 1.0f / m_rows;


    int first = static_cast<int>(floor(begin / factor));
    int last  = static_cast<int>(ceil(end / factor));

    for (int i = first; i < last; ++i)
    {
        float row_start = std::max(i * factor, begin);
        float row_end   = std::min((i + 1) * factor, end);

        auto& corner = m_lastCorner[i];

        float start_01 = m_rows * (row_start - i * factor);
        float end_01   = m_rows * (row_end - i * factor);
        const ImVec2 left{corner[0] + start_01 * m_size[0], corner[1]};
        const ImVec2 right{corner[0] + end_01 * m_size[0], corner[1] + m_size[1]};

        if (outline)
        {
            DrawOutlinedRect(left, right, color);
        }
        else
        {
            DrawRect(left, right, color);
        }
    }
}

void ColoredBar::DrawOutlinedRect(const vec2& begin, const vec2& end, const ColoredBar::BarColor& color)
{
    m_lastDrawList->AddRectFilled(begin, end, ImColor(color.fill), m_rounding, m_rounding_corners);
    m_lastDrawList->AddRect(begin, end, ImColor(color.outline), m_rounding, m_rounding_corners);
}

void ColoredBar::DrawRect(const vec2& begin, const vec2& end, const ColoredBar::BarColor& color)
{
    m_lastDrawList->AddRectFilled(begin, end, ImColor(color.fill), m_rounding, m_rounding_corners);
}


bool captureMouse()
{
    return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse;
}

bool captureKeyboard()
{
    return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard;
}

std::streambuf* test;

IMConsole::IMConsole(const std::string& name, const Saiga::ivec2& position, const Saiga::ivec2& size,
                     bool write_to_cout)
    : std::ostream(this), name(name), position(position), size(size), writeToCout(write_to_cout)
{
    cout_buf = std::cout.rdbuf();
}

void IMConsole::render()
{
    if (should_render)
    {
        BeginWindow();
        RenderTextArea();
        EndWindow();
    }
}

void IMConsole::BeginWindow()
{
    if (!Saiga::editor_gui.enabled)
    {
        ImGui::SetNextWindowPos(ImVec2(position(0), position(1)), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(size(0), size(1)), ImGuiCond_Once);
    }
    ImGui::Begin(name.c_str(), &should_render, ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar);
}

void IMConsole::EndWindow()
{
    ImGui::End();
}


void IMConsole::RenderTextArea()
{
    float footer_height_to_reserve =
        ImGui::GetStyle().ItemSpacing.y + ImGui::GetFrameHeightWithSpacing();  // 1 separator, 1 input text

    // use full frame
    footer_height_to_reserve = 0;
    ImGui::BeginChild("ScrollingRegion", ImVec2(0, -footer_height_to_reserve), false,
                      ImGuiWindowFlags_HorizontalScrollbar |
                          ImGuiWindowFlags_AlwaysVerticalScrollbar);  // Leave room for 1 separator + 1 InputText

    // right click action
    if (ImGui::BeginPopupContextWindow())
    {
        if (ImGui::Selectable("Clear")) data.clear();
        if (ImGui::Selectable("ScrollToBottom")) scrollToBottom = !scrollToBottom;

        ImGui::EndPopup();
    }

    ImGui::TextUnformatted(&data.front(), &data.back());

    if (scrollToBottom && scrollDownAtNextRender)
    {
        // For some reason GetScrollMaxY doesn't completely scroll down
        ImGui::SetScrollY(GetScrollMaxY() + 100);
        scrollDownAtNextRender = false;
    }
    ImGui::EndChild();
}

void IMConsole::setOutputFile(const std::string& file)
{
    outFile = std::make_shared<std::ofstream>(file);
    SAIGA_ASSERT(outFile->is_open());
}

int IMConsole::overflow(int c)
{
    data.push_back(c);
    if (outFile)
    {
        (*outFile) << (char)c;
        if (c == '\n') outFile->flush();
    }
    if (writeToCout)
    {
        cout_buf->sputc(c);
    }

    if (scrollToBottom)
    {
        scrollDownAtNextRender = true;
    }
    return 0;
}



bool StringCombo(const char* label, int* current_item, const std::vector<std::string>& data)
{
    std::vector<const char*> strings;
    for (auto& d : data) strings.push_back(d.data());
    return ImGui::Combo(label, current_item, strings.data(), strings.size());
}

IMTable::IMTable(const std::string& name, const std::vector<int>& colum_width,
                 const std::vector<std::string>& colum_name)
    : Table(colum_width), console(name)
{
    SAIGA_ASSERT(colum_width.size() == colum_name.size());

    std::stringstream header_strm;
    setStream(header_strm);
    for (auto cn : colum_name)
    {
        (*this) << cn;
    }

    header = header_strm.str();
    setStream(console);
}


void IMTable::Render()
{
    console.BeginWindow();
    ImGui::Text("%s", header.c_str());
    ImGui::Separator();
    console.RenderTextArea();
    console.EndWindow();
}

}  // namespace ImGui

namespace Saiga
{
void initImGui(const ImGuiParameters& params)
{
    ImGuiIO& io = ImGui::GetIO();

    auto fontFile = SearchPathes::font(params.font);
    if (!fontFile.empty())
    {
        ImFontConfig conf;
        conf.RasterizerMultiply = params.fontBrightness;
        io.Fonts->AddFontFromFileTTF(fontFile.c_str(), params.fontSize, &conf);
    }
    else
    {
        // use default integrated imgui font
        io.Fonts->AddFontDefault();
    }

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigDockingWithShift = false;

    vec3 color_text;
    vec3 color_background_low;
    vec3 color_background_medium;
    vec3 color_background_high;
    vec3 color_highlight_low;
    vec3 color_highlight_high;


    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors    = ImGui::GetStyle().Colors;

    switch (params.theme)
    {
        case ImGuiTheme::SAIGA:
        {
            style.Alpha          = 1;
            style.WindowRounding = 0;
            style.FrameRounding  = 0;
            color_text           = vec3(0.0, 1.0, 0.0);

            color_background_low    = make_vec3(0.2);
            color_background_medium = make_vec3(0.3);
            color_background_high   = make_vec3(0.4);
            color_highlight_low     = make_vec3(0.5);
            color_highlight_high    = make_vec3(0.6);
            break;
        }
        default:
        {
            // Use default imgui theme
            return;
        }
    }

#define COL_ALPHA(_col, _alpha) ImVec4(_col[0], _col[1], _col[2], _alpha);

    colors[ImGuiCol_Text]         = COL_ALPHA(color_text, 1.00f);
    colors[ImGuiCol_TextDisabled] = COL_ALPHA(color_text, 0.58f);
    colors[ImGuiCol_WindowBg]     = COL_ALPHA(color_background_low, 0.95f);
    colors[ImGuiCol_ChildBg]      = COL_ALPHA(color_background_low, 0.58f);
    colors[ImGuiCol_Border]       = COL_ALPHA(color_highlight_high, 0.00f);
    colors[ImGuiCol_BorderShadow] = COL_ALPHA(color_background_low, 0.00f);

    // Background of checkbox, radio button, plot, slider, text input
    colors[ImGuiCol_FrameBg]        = COL_ALPHA(color_background_medium, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = COL_ALPHA(color_highlight_low, 0.78f);
    colors[ImGuiCol_FrameBgActive]  = COL_ALPHA(color_highlight_high, 1.00f);

    // title bar
    colors[ImGuiCol_TitleBg]          = COL_ALPHA(color_background_high, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed] = COL_ALPHA(color_highlight_low, 0.75f);
    colors[ImGuiCol_TitleBgActive]    = COL_ALPHA(color_highlight_low, 1.00f);
    colors[ImGuiCol_MenuBarBg]        = COL_ALPHA(color_background_low, 0.47f);

    colors[ImGuiCol_Tab]                = COL_ALPHA(color_background_medium, 1.f);
    colors[ImGuiCol_TabHovered]         = COL_ALPHA(color_highlight_high, 1.f);
    colors[ImGuiCol_TabActive]          = COL_ALPHA(color_highlight_low, 1.f);
    colors[ImGuiCol_TabUnfocused]       = COL_ALPHA(color_background_low, 1.f);
    colors[ImGuiCol_TabUnfocusedActive] = COL_ALPHA(color_background_low, 1.f);

    colors[ImGuiCol_ScrollbarBg]          = COL_ALPHA(color_background_medium, 1.00f);
    colors[ImGuiCol_ScrollbarGrab]        = COL_ALPHA(color_background_high, 0.8);
    colors[ImGuiCol_ScrollbarGrabHovered] = COL_ALPHA(color_highlight_low, 0.8);
    colors[ImGuiCol_ScrollbarGrabActive]  = COL_ALPHA(color_highlight_high, 1.00f);

    colors[ImGuiCol_CheckMark]        = COL_ALPHA(color_highlight_high, 0.80f);
    colors[ImGuiCol_SliderGrab]       = COL_ALPHA(color_background_low, 0.50f);
    colors[ImGuiCol_SliderGrabActive] = COL_ALPHA(color_background_low, 1.00f);
    colors[ImGuiCol_Button]           = COL_ALPHA(color_background_high, 0.50f);
    colors[ImGuiCol_ButtonHovered]    = COL_ALPHA(color_highlight_low, 0.86f);
    colors[ImGuiCol_ButtonActive]     = COL_ALPHA(color_highlight_high, 1.00f);

    colors[ImGuiCol_Header]        = COL_ALPHA(color_background_high, 0.76f);
    colors[ImGuiCol_HeaderHovered] = COL_ALPHA(color_highlight_low, 0.86f);
    colors[ImGuiCol_HeaderActive]  = COL_ALPHA(color_highlight_high, 1.00f);

    //    colors[ImGuiCol_Column]        = COL_ALPHA(color_background_high, 0.8);
    //    colors[ImGuiCol_ColumnHovered] = COL_ALPHA(color_highlight_low, 0.8);
    //    colors[ImGuiCol_ColumnActive]  = COL_ALPHA(color_highlight_high, 1.00f);

    colors[ImGuiCol_ResizeGrip]        = COL_ALPHA(color_background_medium, 0.5f);
    colors[ImGuiCol_ResizeGripHovered] = COL_ALPHA(color_background_medium, 1);
    colors[ImGuiCol_ResizeGripActive]  = COL_ALPHA(color_background_high, 1.00f);

    colors[ImGuiCol_PlotLines]            = COL_ALPHA(color_text, 0.63f);
    colors[ImGuiCol_PlotLinesHovered]     = COL_ALPHA(color_text, 1.00f);
    colors[ImGuiCol_PlotHistogram]        = COL_ALPHA(color_text, 0.63f);
    colors[ImGuiCol_PlotHistogramHovered] = COL_ALPHA(color_text, 1.00f);

    colors[ImGuiCol_TextSelectedBg] = COL_ALPHA(color_background_low, 0.43f);
    colors[ImGuiCol_PopupBg]        = COL_ALPHA(color_background_low, 0.92f);
    // colors[ImGuiCol_ModalWindowDarkening] = COL_ALPHA(color_background_low, 0.73f);
}

void ImGuiParameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    enable         = ini.GetAddBool("imgui", "enable", enable);
    font           = ini.GetAddString("imgui", "font", font.c_str());
    fontSize       = ini.GetAddLong("imgui", "fontSize", fontSize);
    fontBrightness = ini.GetAddDouble("imgui", "fontBrightness", fontBrightness);

    std::string comment =
        "# Available Themes: \n"
        "# 0-Saiga, 1-ImGuiDefault";
    theme = (ImGuiTheme)ini.GetAddLong("imgui", "theme", (int)theme, comment.c_str());

    if (ini.changed()) ini.SaveFile(file.c_str());
}

ImGui::IMConsole console = ImGui::IMConsole("Log", {0, 0}, {500, 250}, true);

}  // namespace Saiga
