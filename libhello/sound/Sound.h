#pragma once

#include <string>


namespace sound {

class Sound
{
public:

//    ALuint buffer;
//    ALint channels;
//    ALint bits;
//    ALsizei size;
//    ALint frequency;
//    ALenum format;

    //needed to remove the al dependencies
    unsigned int buffer;
    int channels;
    int bits;
    int size;
    int frequency;
    int format;

    Sound ();
    virtual ~Sound ();



};
}

