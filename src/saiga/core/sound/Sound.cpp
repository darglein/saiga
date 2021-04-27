/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Sound.h"

#include "internal/noGraphicsAPI.h"

#include <al.h>
#include <alc.h>
#include <iostream>
namespace Saiga
{
namespace sound
{
Sound::Sound() {}

Sound::~Sound()
{
    deleteBuffer();
}

void Sound::setFormat(int _channels, int _bitsPerSample, int _frequency)
{
    channels      = _channels;
    bitsPerSample = _bitsPerSample;
    frequency     = _frequency;


    SAIGA_ASSERT(channels == 1 || channels == 2);
    SAIGA_ASSERT(bitsPerSample == 8 || bitsPerSample == 16);

    if (channels == 1)
    {
        if (bitsPerSample == 8)
            format = AL_FORMAT_MONO8;
        else if (bitsPerSample == 16)
            format = AL_FORMAT_MONO16;
    }
    else if (channels == 2)
    {
        if (bitsPerSample == 8)
            format = AL_FORMAT_STEREO8;
        else if (bitsPerSample == 16)
            format = AL_FORMAT_STEREO16;
    }
}

void Sound::createBuffer(const void* data, int _size)
{
#if defined(SAIGA_DEBUG)
    if (_size > bitsPerSample / 8)
    {
        if (!checkFirstSample(data))
        {
            std::cerr
                << "Warning: " << name
                << " The first sample of this sound is not zero. This may cause artifacts when playing with OpenAL."
                << std::endl;
            std::cerr << "Value = " << toFloat(getSample(0, 0, data)) << std::endl;
        }
    }
#endif
    alGenBuffers(1, &buffer);
    alBufferData(buffer, format, data, _size, frequency);
    SAIGA_ASSERT(buffer);
    assert_no_alerror();
}

void Sound::deleteBuffer()
{
    if (buffer)
    {
        alDeleteBuffers(1, &buffer);
        buffer = 0;
        assert_no_alerror();
    }
}


bool Sound::checkFirstSample(const void* data)
{
    bool ret = true;
    for (int c = 0; c < channels; ++c)
    {
        float val = toFloat(getSample(0, 0, data));

        ret &= abs(val) <= 0.0001f;
    }
    return ret;
}

int32_t Sound::getSample(int sample, int channel, const void* data)
{
    int32_t val = 0;
    if (bitsPerSample == 16)
    {
        int8_t* d = (int8_t*)data;
        val       = d[sample * channels + channel];
    }
    if (bitsPerSample == 16)
    {
        int16_t* d = (int16_t*)data;
        val        = d[sample * channels + channel];
    }
    return val;
}

float Sound::toFloat(int32_t sample)
{
    int32_t div = (1 << (bitsPerSample - 1));
    return (float)sample / div;
}

}  // namespace sound
}  // namespace Saiga
