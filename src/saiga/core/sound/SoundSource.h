/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include "OpenAL.h"
#include "Sound.h"

namespace Saiga
{
namespace sound
{
class SAIGA_CORE_API SoundSource
{
    //    ALuint source;
    unsigned int source  = 0;
    Sound* sound         = nullptr;
    bool music           = false;
    float myMasterVolume = 1.f;
    float volume         = 1.f;

   public:
    SoundSource(Sound* sound);
    SoundSource();
    ~SoundSource();

    SoundSource(SoundSource const&) : SoundSource() {}
    SoundSource& operator=(SoundSource const&) { return *this; }

    void play();
    void stop();

    void setSound(Sound* sound);

    void setVolume(float f);

    void setPitch(float pitch);

    void setPosition(const vec3& pos);
    void setVelocity(const vec3& velocity);
    bool isPlaying();
    bool isMusic() { return music; }
    void setLooping(bool looping);
    void setReferenceDistance(float v);

    void reset(bool isMusic, float masterVolume);

    void makeBackground();
    void setMasterVolume(float v);

    void unloadSound();
};

}  // namespace sound
}  // namespace Saiga
