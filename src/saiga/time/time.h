/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include <chrono>

using game_ratio_t = std::nano;
using tick_t = std::chrono::duration<int64_t, game_ratio_t>;
using tickd_t = std::chrono::duration<double, game_ratio_t>;

//using a floating point type here because we need to do alot of interpolation stuff
using animationtime_t = std::chrono::duration<double>;
