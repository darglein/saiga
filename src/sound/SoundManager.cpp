#include "saiga/sound/SoundManager.h"

#include "saiga/sound/SoundLoader.h"
#include "saiga/sound/SoundSource.h"
#include <iostream>

#include <AL/al.h>
#include <AL/alc.h>

#ifdef USE_ALUT
#include <AL/alut.h>
#endif

namespace sound {


SoundManager::SoundManager (int maxSources, int fixedSources) : maxSources(maxSources),fixedSources(fixedSources),oldestSource(fixedSources){
    cout << "SoundManager()" << endl;
    initOpenAL();


    quietSoundSource = new SoundSource();

    setListenerPosition(vec3(0));
    setListenerVelocity(vec3(0));
    setListenerOrientation(vec3(0,0,-1),vec3(0,1,0));
    setListenerGain(masterVolumne);

    sources.resize(maxSources);

    assert_no_alerror();
}

SoundManager::~SoundManager () {

    cout<<"~SoundManager"<<endl;
    for(auto it = soundMap.begin() ; it!=soundMap.end();++it){
        Sound* sound = it->second;
        delete sound;
    }

    delete quietSoundSource;

    sources.clear();

    quitOpenAL();


}

SoundSource* SoundManager::getSoundSource(const std::string& file){
    Sound* sound = nullptr;

    auto it = soundMap.find(file);
    if(it==soundMap.end()){
        std::cerr << "Sound not loaded: " << file << endl;
        return quietSoundSource;
    }else{
        sound = it->second;
    }

    SoundSource* s  = &sources[oldestSource];
    if(s->isPlaying()){
        cout << "<SoundManager> Stopping sound before playing a new one!" << endl;
        s->stop();
    }
    s->reset();
    s->setSound(sound);
    oldestSource = glm::max( (oldestSource + 1) % maxSources, fixedSources );
    assert_no_alerror();
    return s;
}

SoundSource *SoundManager::getFixedSoundSource(const std::string &file, int id)
{
    Sound* sound = nullptr;

    auto it = soundMap.find(file);
    if(it==soundMap.end()){
        std::cerr << "Sound not loaded: " << file << endl;
        return quietSoundSource;
    }else{
        sound = it->second;
    }

    SoundSource* s  = &sources[id];
    if(s->isPlaying()){
        s->stop();
    }
    s->reset();
    s->setSound(sound);
    assert_no_alerror();
    return s;
}

SoundSource *SoundManager::getFixedSoundSource(int id)
{
    return &sources[id];
}

void SoundManager::loadWaveSound(const std::string &file)
{
    auto it = soundMap.find(file);
    if(it==soundMap.end()){
        SoundLoader sl;

        Sound* loadedsound;
        if ((loadedsound = sl.loadWaveFile(file))!=0){
            soundMap[file] = loadedsound;
        } else {
            cout << "Could not load sound: " << file << endl;
            assert(0);
        }
    }else{
        cout << "Sound already loaded: " << file << endl;
    }
    assert_no_alerror();
}

void SoundManager::loadOpusSound(const std::string &file)
{
    auto it = soundMap.find(file);
    if(it==soundMap.end()){
        SoundLoader sl;

        Sound* loadedsound;
        if ((loadedsound = sl.loadOpusFile(file))!=0){
            soundMap[file] = loadedsound;
        } else {
            cout << "Could not load sound: " << file << endl;
            assert(0);
        }
    }else{
        cout << "Sound already loaded: " << file << endl;
    }
    assert_no_alerror();
}




void SoundManager::setListenerPosition(const glm::vec3 &pos){
    alListenerfv(AL_POSITION,&pos[0]);
    assert_no_alerror();
}

void SoundManager::setListenerVelocity(const vec3& velocity){
    alListenerfv(AL_VELOCITY,&velocity[0]);
    assert_no_alerror();
}

void SoundManager::setListenerOrientation(const glm::vec3 &at, const glm::vec3 &up){
    ALfloat	listenerOri[6];
    listenerOri[0] = at[0];
    listenerOri[1] = at[1];
    listenerOri[2] = at[2];
    listenerOri[3] = up[0];
    listenerOri[4] = up[1];
    listenerOri[5] = up[2];
    alListenerfv(AL_ORIENTATION,listenerOri);
    assert_no_alerror();
}

void SoundManager::setListenerGain(float g){
    masterVolumne = g;
    alListenerf(AL_GAIN, masterVolumne);
    assert_no_alerror();
}

void SoundManager::setMute(bool b)
{
    muted = b;
    if(muted)
        alListenerf(AL_GAIN, 0);
    else
        alListenerf(AL_GAIN, masterVolumne);
    assert_no_alerror();
}

void SoundManager::setTimeScale(float scale)
{
    //update all sources
    for (SoundSource& s : sources){
        s.setPitch(scale);
    }
    assert_no_alerror();
}





}
