/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/time/timer.h"

#include <glm/glm.hpp>
#include <vector>
struct ImDrawList;
namespace ImGui
{
class SAIGA_GLOBAL Graph
{
   public:
    Graph(const std::string& name = "Graph", int numValues = 80);
    void addValue(float t);
    void renderImGui();

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


class SAIGA_GLOBAL TimeGraph : public Graph
{
   public:
    TimeGraph(const std::string& name = "Time", int numValues = 80);
    void addTime(float t);

   protected:
    virtual void renderImGuiDerived();
    float hzExp = 0;
    Saiga::Timer timer;
};



class SAIGA_GLOBAL ColoredBar
{
   private:
    glm::vec2 m_size;
    glm::vec4 m_back_color;
    bool m_auto_size;

    glm::vec2 m_lastCorner;
    ImDrawList* m_lastDrawList;

   public:
    ColoredBar(glm::vec2 size, glm::vec4 background_color, bool auto_size = false)
        : m_size(size), m_back_color(background_color), m_auto_size(auto_size), m_lastDrawList(nullptr)
    {
    }
    void renderBackground();
    void renderArea(float begin, float end, glm::vec4 color, float rounding = 0.0f, int rounding_corners = 0);
};

}  // namespace ImGui
