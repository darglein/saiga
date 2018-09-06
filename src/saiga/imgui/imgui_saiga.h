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

class SAIGA_GLOBAL TimeGraph
{
  public:
    TimeGraph(const std::string& name, int numValues = 80);
    void addTime(float t);
    void renderImGui();
private:
    std::string name;
    int numValues;

    float lastTime = 0;
    float maxTime = 0;
    float average = 0;
    int currentIndex = 0;
    std::vector<float> updateTimes;

    int r;

    Saiga::Timer timer;
};

}
