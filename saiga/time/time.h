#pragma once

#include "saiga/config.h"
#include <chrono>

typedef std::nano game_ratio_t;
typedef std::chrono::duration<int64_t, game_ratio_t> tick_t;
typedef std::chrono::duration<double, game_ratio_t> tickd_t;

//using a floating point type here because we need to do alot of interpolation stuff
typedef std::chrono::duration<double> animationtime_t;

