#include "libhello/sdl/timer.h"

Timer::Timer(){

}

void Timer::start(){
    start_time = SDL_GetTicks();
}

float Timer::getTimeMS(){
    uint time = SDL_GetTicks()-start_time;
    return (float)time;
}

float Timer::getTimeS(){
    uint time = SDL_GetTicks()-start_time;
    return (float)time/1000.0f;
}
