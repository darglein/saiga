#include "internal/noGraphicsAPI.h"
#include "saiga/imgui/imgui.h"
#include "saiga/util/tostring.h"

namespace ImGui
{
Graph::Graph(const std::string& name, int numValues) : name(name), numValues(numValues), values(numValues, 0)
{
    r = rand();
}

void Graph::addValue(float t)
{
    maxValue             = std::max(t, maxValue);
    values[currentIndex] = t;
    currentIndex         = (currentIndex + 1) % numValues;
    float alpha          = 0.1;
    average              = (1 - alpha) * average + alpha * t;
}

void Graph::renderImGui()
{
    ImGui::PushID(r);

    renderImGuiDerived();
    ImGui::PlotLines("Time", values.data(), numValues, currentIndex, ("avg " + Saiga::to_string(average)).c_str(), 0,
                     maxValue, ImVec2(0, 80));
    ImGui::SameLine();
    if (ImGui::Button("R"))
    {
        maxValue = 0;
    }

    ImGui::PopID();
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
    ImGui::Text("%s Time: %fms Hz: %f", name.c_str(), lastValue, 1000.0f / average);
}



}  // namespace ImGui
