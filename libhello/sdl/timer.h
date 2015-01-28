#ifndef TIMER_H
#define TIMER_H

#include <SDL2/SDL.h>
typedef unsigned int uint;
class Timer{
    uint start_time =0;
public:
    Timer();
    void start();
    float getTimeMS();
    float getTimeS();

};

#endif // TIMER_H
