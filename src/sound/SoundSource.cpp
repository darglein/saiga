#include <saiga/sound/SoundSource.h>

#include <AL/al.h>
#include <AL/alc.h>



namespace sound {


SoundSource::SoundSource(Sound *sound) : sound(sound){
    alGenSources(1, &source);

    alSourcei(source, AL_BUFFER, sound->buffer);
}

SoundSource::SoundSource()
{
    alGenSources(1, &source);
}

SoundSource::~SoundSource()
{
    alDeleteSources(1,&source);
}

void SoundSource::play(){
#ifndef NO_SOUND
    if (!sound)
        return;
    alSourcePlay(source);
#endif
}

void SoundSource::stop()
{
    alSourceStop(source);
}

void SoundSource::setSound(Sound *sound)
{
    this->sound = sound;
    alSourcei(source, AL_BUFFER, sound->buffer);
}

void SoundSource::setVolume(float f)
{
    alSourcef(source, AL_GAIN, f);
}

void SoundSource::setPitch(float pitch)
{
    alSourcef(source, AL_PITCH, pitch);
}

void SoundSource::setPosition(const glm::vec3 &pos)
{
    alSourcefv(source,AL_POSITION,&pos[0]);
}


void SoundSource::setVelocity(const vec3& velocity){
    alSourcefv(source,AL_VELOCITY,&velocity[0]);
}


bool SoundSource::isPlaying(){
    ALint source_state;
    alGetSourcei(source, AL_SOURCE_STATE, &source_state);

    return source_state == AL_PLAYING;
}

void SoundSource::setLooping(bool looping){
    alSourcei(source, AL_LOOPING, looping);
}

void SoundSource::setReferenceDistance(float v){
    alSourcef(source,AL_REFERENCE_DISTANCE,v);
}

void SoundSource::makeBackground()
{
    alSourcei( source, AL_SOURCE_RELATIVE, AL_TRUE );
    alSourcef( source, AL_ROLLOFF_FACTOR, 0.0 );
}


}
