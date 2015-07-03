#pragma once

#include <libhello/config.h>
#include <string>


namespace sound {

class SAIGA_GLOBAL Sound
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

