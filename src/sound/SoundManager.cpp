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


//only init at first call and quit when init calls is back to 0
int initCalls = 0;

#ifndef USE_ALUT
ALCdevice* device;
ALCcontext* context;
#endif

void initOpenAL(){
    initCalls++;
    if(initCalls!=1)
        return;

#ifdef USE_ALUT
    //let alut create the context
    alutInit(0, NULL);
#else
    //manually create a context
    device = alcOpenDevice(NULL);
    context = alcCreateContext(device, NULL);
    alcMakeContextCurrent(context);
#endif
}

extern void quitOpenAL(){
    initCalls--;
    if(initCalls!=0)
        return;
#ifdef USE_ALUT
    alutExit();
#else
    alcDestroyContext(context);
    alcCloseDevice(device);
#endif
}



SoundManager::SoundManager (int maxSources) : maxSources(maxSources){
    cout << "SoundManager()" << endl;
    initOpenAL();


    checkForSoundErrors();

    setListenerPosition(vec3(0));
    setListenerVelocity(vec3(0));
    setListenerOrientation(vec3(0,0,-1),vec3(0,1,0));
    setListenerGain(masterVolumne);

    sources.resize(maxSources);
}

SoundManager::~SoundManager () {

    cout<<"~SoundManager"<<endl;
    for(auto it = soundMap.begin() ; it!=soundMap.end();++it){
        Sound* sound = it->second;
        delete sound;
    }

    quitOpenAL();


}

SoundSource* SoundManager::getSoundSource(const std::string& file){
    Sound* sound = nullptr;

    auto it = soundMap.find(file);
    if(it==soundMap.end()){
        std::cerr << "Sound not loaded: " << file << endl;
        return &quietSoundSource;
    }else{
        sound = it->second;
    }

//    int end = sourceIndex-1;
//    if (end < 0) end += sources.size();

//    while(end >= 0 && end != sourceIndex){
//        SoundSource* s  = sources[sourceIndex];
//        sourceIndex = (sourceIndex + 1) % sources.size();
//        if (!s->isPlaying()){
//            s->setSound(sound);
//            return s;
//        }
//    }


//    cout << "adding new soundsource: " << sources.size() << endl;

//    SoundSource* news = new SoundSource(sound);
//    sources.push_back(news);
//    return news;


    SoundSource* s  = &sources[oldestSource];
    if(s->isPlaying()){
        cout << "<SoundManager> Stopping sound before playing a new one!" << endl;
        s->stop();
    }
    s->setSound(sound);
    oldestSource = (oldestSource + 1) % maxSources;
    return s;
}

void SoundManager::loadSound(const std::string &file)
{
    auto it = soundMap.find(file);
    if(it==soundMap.end()){
        SoundLoader sl;

        Sound* loadedsound;
        if ((loadedsound = sl.loadSound(file))){
            soundMap[file] = loadedsound;
        } else {
            cout << "Could not load sound: " << file << endl;
            exit(1);
        }
    }else{
        cout << "Sound already loaded: " << file << endl;
    }


}




void SoundManager::setListenerPosition(const glm::vec3 &pos){
    alListenerfv(AL_POSITION,&pos[0]);
}

void SoundManager::setListenerVelocity(const vec3& velocity){
    alListenerfv(AL_VELOCITY,&velocity[0]);
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
}

void SoundManager::setListenerGain(float g){
    masterVolumne = g;
    alListenerf(AL_GAIN, masterVolumne);
}

void SoundManager::setMute(bool b)
{
    muted = b;
    if(muted)
        alListenerf(AL_GAIN, 0);
    else
        alListenerf(AL_GAIN, masterVolumne);
}

void SoundManager::setTimeScale(float scale)
{
    //update all sources
    for (SoundSource& s : sources){
        s.setPitch(scale);
    }
}



void SoundManager::checkForSoundErrors()
{
    ALCenum error;

    error = alGetError();
    if (error != AL_NO_ERROR){
        std::cout << "AUDIO ERROR! ("  << error << ")" << std::endl;
        std::cout << getALCErrorString(error) << std::endl;
        if (error != ALC_INVALID_DEVICE) //TODO limit number of sound sources
            exit(0);
    }
}

std::string SoundManager::getALCErrorString(int err) {
  switch (err) {
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


}
