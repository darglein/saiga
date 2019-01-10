/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/time/timer.h"
#include "saiga/util/math.h"

#include <vector>

struct ImDrawList;
namespace ImGui
{
class SAIGA_GLOBAL Graph
{
   public:
    Graph(const std::string& name = "Graph", int numValues = 80);
    virtual ~Graph() {}
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
   public:
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
    void renderArea(float begin, float end, const BarColor& color);
};

}  // namespace ImGui
