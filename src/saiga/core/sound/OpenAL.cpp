/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "OpenAL.h"

#include <al.h>
#include <alc.h>
#include <iostream>

#ifdef SAIGA_USE_ALUT
#    include <alut.h>
#endif

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
namespace sound
{
// only init at first call and quit when init calls is back to 0
int initCalls = 0;

#ifndef SAIGA_USE_ALUT
ALCdevice* device;
ALCcontext* context;
#endif

void initOpenAL()
{
    initCalls++;
    if (initCalls != 1) return;

#ifdef SAIGA_USE_ALUT
    // let alut create the context
    alutInit(0, NULL);
#else
    // manually create a context
    device  = alcOpenDevice(NULL);
    context = alcCreateContext(device, NULL);
    alcMakeContextCurrent(context);
#endif
    assert_no_alerror();
}

extern void quitOpenAL()
{
    initCalls--;
    if (initCalls != 0) return;
    assert_no_alerror();
#ifdef SAIGA_USE_ALUT
    alutExit();
#else
    alcDestroyContext(context);
    alcCloseDevice(device);
#endif
}



bool checkSoundError()
{
    ALCenum error;

    error = alGetError();
    if (error != AL_NO_ERROR)
    {
        std::cout << "AUDIO ERROR! (" << error << ")" << std::endl;
        std::cout << getALCErrorString(error) << std::endl;
        return true;
    }
    return false;
}

std::string getALCErrorString(int err)
{
    switch (err)
    {
        case ALC_NO_ERROR:
            return "AL_NO_ERROR";
        case ALC_INVALID_DEVICE:
            return "ALC_INVALID_DEVICE";
        case ALC_INVALID_CONTEXT:
            return "ALC_INVALID_CONTEXT";
        case ALC_INVALID_ENUM:
            return "ALC_INVALID_ENUM";
        case ALC_INVALID_VALUE:
            return "ALC_INVALID_VALUE";
        case ALC_OUT_OF_MEMORY:
            return "ALC_OUT_OF_MEMORY";
        default:
            return "no such error code";
    }
}

}  // namespace sound
}  // namespace Saiga
