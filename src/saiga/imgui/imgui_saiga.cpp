#include "saiga/imgui/imgui.h"
#include "saiga/util/tostring.h"

#if defined(SAIGA_VULKAN_INCLUDED) || defined(SAIGA_OPENGL_INCLUDED)
#error This module must be independent of any graphics API.
#endif

namespace ImGui {

TimeGraph::TimeGraph(const std::string &name, int numValues)
    : name(name), numValues(numValues), updateTimes(numValues,0)
{
    r = rand();
    timer.start();
}

void TimeGraph::addTime(float t)
{
    timer.stop();
    lastTime = t;
    maxTime = std::max(t,maxTime);
    average = 0;
    for(auto f : updateTimes)
        average += f;
    average /= numValues;
    updateTimes[currentIndex] = t;
    currentIndex = (currentIndex+1) % numValues;
    timer.start();
}

void TimeGraph::renderImGui()
{
    ImGui::PushID(r);

    ImGui::Text("%s Time: %fms Hz: %f",name.c_str(),lastTime, 1000.0f / timer.getTimeMS());
    ImGui::PlotLines("Time", updateTimes.data(), numValues, currentIndex, ("avg "+Saiga::to_string(average)).c_str(), 0,maxTime, ImVec2(0,80));
    ImGui::SameLine();
    if(ImGui::Button("R"))
    {
        maxTime = 0;
    }

    ImGui::PopID();
}


}
