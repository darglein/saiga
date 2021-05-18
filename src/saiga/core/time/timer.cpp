/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "timer.h"

#ifdef _WIN32
#    include <windows.h>
#endif

#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include "gameTime.h"

#include <algorithm>
#include <iostream>

namespace Saiga
{
//============================================================================================


ExponentialTimer::ExponentialTimer(double alpha) : alpha(alpha) {}

void ExponentialTimer::addMeassurment(tick_t time)
{
    lastTime    = time;
    currentTime = std::chrono::duration_cast<tick_t>(alpha * currentTime + (1 - alpha) * time);
}


//============================================================================================


AverageTimer::AverageTimer(int number) : lastTimes(number, tick_t(0)), number(number) {}



void AverageTimer::addMeassurment(tick_t time)
{
    lastTime                 = time;
    lastTimes[currentTimeId] = time;
    currentTimeId            = (currentTimeId + 1) % lastTimes.size();

    currentTime = tick_t(0);
    minimum     = lastTimes[0];
    maximum     = lastTimes[0];
    for (auto& d : lastTimes)
    {
        currentTime += d;
        minimum = std::min(minimum, d);
        maximum = std::max(maximum, d);
    }
    currentTime /= lastTimes.size();
}

double AverageTimer::getMinimumTimeMS()
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(minimum).count();
}

double AverageTimer::getMaximumTimeMS()
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(maximum).count();
}



}  // namespace Saiga
