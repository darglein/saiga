#pragma once

#include "saiga/config.h"
#include <chrono>

typedef std::chrono::nanoseconds tick_t;
typedef std::nano game_ratio_t;

class SAIGA_GLOBAL GameTime{
public:
    tick_t base = std::chrono::seconds(1);

    //time since start of the game
    tick_t time;

    //time at which the last 'update' took place
    //while updating this is equal to time
    //while rendering time should be greater than updatetime
    tick_t updatetime;

    //timestep of 'update'
    tick_t dt;

    //timestep of 'render' (only != 0 if fps are limited)
    tick_t dtr;
};

//use a global variable here so every object can access it easily
SAIGA_GLOBAL extern GameTime gameTime;
