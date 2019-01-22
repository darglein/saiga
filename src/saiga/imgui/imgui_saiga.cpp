#include "imgui_saiga.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/random.h"
#include "saiga/util/tostring.h"

#include "internal/noGraphicsAPI.h"


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
                     maxValue, ImVec2(0, 80));
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


}  // namespace ImGui
