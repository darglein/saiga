#pragma once

#include "saiga/sound/OpenAL.h"

namespace Saiga {

namespace sound {

class SAIGA_GLOBAL Sound
{
public:
    std::string name;

    unsigned int buffer = 0;
    int channels;
    int bitsPerSample;
    int frequency;
    int format;

    Sound ();
    virtual ~Sound ();

    void setFormat(int _channels, int _bitsPerSample, int _frequency);
    void createBuffer(const void* data, int size);
    void deleteBuffer();


    //check if first sample is 0. prints a warning if it's not 0.
    bool checkFirstSample(const void* data);

    int32_t getSample(int sample, int channel, const void* data);
    float toFloat(int32_t sample);

};
}

}
