/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "SoundSource.h"

#include "internal/noGraphicsAPI.h"

#include "SoundManager.h"

#include <al.h>
#include <alc.h>

namespace Saiga
{
namespace sound
{
SoundSource::SoundSource(Sound* sound) : sound(sound)
{
    alGenSources(1, &source);

    alSourcei(source, AL_BUFFER, sound->buffer);
    //    reset();
    assert_no_alerror();
}

SoundSource::SoundSource()
{
    alGenSources(1, &source);
    assert_no_alerror();
}

SoundSource::~SoundSource()
{
    alDeleteSources(1, &source);
}

void SoundSource::play()
{
//        std::cout << "playing " << sound->name << std::endl;
#ifndef NO_SOUND
    if (!sound) return;
    alSourcePlay(source);
    assert_no_alerror();
#endif
}

void SoundSource::stop()
{
    alSourceStop(source);
    assert_no_alerror();
}

void SoundSource::setSound(Sound* sound)
{
    this->sound = sound;
    alSourcei(source, AL_BUFFER, sound->buffer);
    assert_no_alerror();
}

void SoundSource::setVolume(float f)
{
    volume = f;
    alSourcef(source, AL_GAIN, volume * myMasterVolume);
    assert_no_alerror();
}

void SoundSource::setMasterVolume(float v)
{
    float oldVolume = myMasterVolume;
    myMasterVolume  = v;

    if (myMasterVolume != oldVolume)
    {
        setVolume(volume);
    }
}

void SoundSource::unloadSound()
{
    stop();
    alSourcei(source, AL_BUFFER, AL_NONE);
    sound = nullptr;
    assert_no_alerror();
}

void SoundSource::setPitch(float pitch)
{
    alSourcef(source, AL_PITCH, pitch);
    assert_no_alerror();
}

void SoundSource::setPosition(const vec3& pos)
{
    alSourcefv(source, AL_POSITION, &pos[0]);
    assert_no_alerror();
}


void SoundSource::setVelocity(const vec3& velocity)
{
    alSourcefv(source, AL_VELOCITY, &velocity[0]);
    assert_no_alerror();
}


bool SoundSource::isPlaying()
{
    ALint source_state;
    alGetSourcei(source, AL_SOURCE_STATE, &source_state);
    assert_no_alerror();
    return source_state == AL_PLAYING;
}

void SoundSource::setLooping(bool looping)
{
    alSourcei(source, AL_LOOPING, looping);
    assert_no_alerror();
}

void SoundSource::setReferenceDistance(float v)
{
    alSourcef(source, AL_REFERENCE_DISTANCE, v);
    assert_no_alerror();
}

void SoundSource::reset(bool isMusic, float masterVolume)
{
    music          = isMusic;
    myMasterVolume = masterVolume;
    setLooping(false);
    setPosition(make_vec3(0));
    setVelocity(make_vec3(0));
    setVolume(1.f);
    //    setPitch(1.f); //this is set with timescale
    setReferenceDistance(1.f);  // TODO dont know if correct?
    // make foreground
    alSourcei(source, AL_SOURCE_RELATIVE, AL_FALSE);
    alSourcef(source, AL_ROLLOFF_FACTOR, 1.0);  // TODO dont know if correct?
    assert_no_alerror();
}

void SoundSource::makeBackground()
{
    alSourcei(source, AL_SOURCE_RELATIVE, AL_TRUE);
    alSourcef(source, AL_ROLLOFF_FACTOR, 0.0);
    setPosition(make_vec3(0));
    assert_no_alerror();
}

}  // namespace sound
}  // namespace Saiga
