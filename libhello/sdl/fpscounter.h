#ifndef FPSCOUNTER_H
#define FPSCOUNTER_H

#include <libhello/config.h>
#include <SDL2/SDL.h>
//http://sdl.beuc.net/sdl.wiki/SDL_Average_FPS_Measurement

#define FRAME_VALUES 5

typedef unsigned int uint;

class SAIGA_GLOBAL FpsCounter
{
    uint frametimes[FRAME_VALUES];
    uint frametimelast;
    uint framecount;
    float framespersecond;
public:
    FpsCounter();
    void update();
    float getCurrentFPS();
};

#endif // FPSCOUNTER_H
