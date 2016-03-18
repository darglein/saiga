#pragma once

#include <saiga/config.h>
#include <saiga/util/glm.h>
#include <saiga/sound/Sound.h>

namespace sound {



class SAIGA_GLOBAL SoundSource{
//    ALuint source;
    unsigned int source;
    Sound* sound = nullptr;
public:
    SoundSource( Sound* sound);
    SoundSource();
    ~SoundSource();
    void play();
    void stop();

    void setSound(Sound* sound);

    void setVolume(float f);

    void setPitch(float pitch);

    void setPosition(const vec3& pos);
    void setVelocity(const vec3& velocity);
    bool isPlaying();
    void setLooping(bool looping);
    void setReferenceDistance(float v);

    void reset();

    void makeBackground();
};



}

