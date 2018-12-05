/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/time/timer.h"

#include <vector>

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

}  // namespace ImGui
