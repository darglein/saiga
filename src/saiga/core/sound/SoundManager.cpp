/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "SoundManager.h"

#include "SoundLoader.h"
#include "SoundSource.h"

#include <al.h>
#include <alc.h>
#include <iostream>

#ifdef SAIGA_USE_ALUT
#    include <alut.h>
#endif

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
namespace sound
{
SoundManager::SoundManager(int maxSources, int fixedSources)
    : maxSources(maxSources), fixedSources(fixedSources), oldestSource(fixedSources)
{
    std::cout << "SoundManager()" << std::endl;
    initOpenAL();


    quietSoundSource = new SoundSource();

    setListenerPosition(make_vec3(0));
    setListenerVelocity(make_vec3(0));
    setListenerOrientation(vec3(0, 0, -1), vec3(0, 1, 0));
    setListenerGain(masterVolume);

    sources.resize(maxSources);

    assert_no_alerror();
}

SoundManager::~SoundManager()
{
    std::cout << "~SoundManager" << std::endl;


    delete quietSoundSource;

    sources.clear();


    for (auto it = soundMap.begin(); it != soundMap.end(); ++it)
    {
        Sound* sound = it->second;
        delete sound;
    }



    quitOpenAL();
}

SoundSource* SoundManager::getSoundSource(const std::string& file, bool isMusic)
{
    SAIGA_ASSERT(!parallelSoundLoaderRunning);
    Sound* sound = nullptr;

    auto it = soundMap.find(file);
    if (it == soundMap.end())
    {
        std::cerr << "Sound not loaded: " << file << std::endl;
        SAIGA_ASSERT(false);
        return quietSoundSource;
    }
    else
    {
        sound = it->second;
    }

    //    std::cout << "returning source " << oldestSource << std::endl;
    SoundSource* s = &sources[oldestSource];
    if (s->isPlaying())
    {
        std::cout << "<SoundManager> Stopping sound before playing a new one!" << std::endl;
        s->stop();
    }

    s->reset(isMusic, isMusic ? musicVolume : effectsVolume);
    s->setSound(sound);
    oldestSource = std::max((oldestSource + 1) % maxSources, fixedSources);
    assert_no_alerror();
    return s;
}

SoundSource* SoundManager::getSoundSourceWhileStillLoading(const std::string& file, bool isMusic)
{
    Sound* sound = nullptr;

    {
        std::lock_guard<std::mutex> lock(soundMapLock);  // scoped lock

        auto it = soundMap.find(file);
        if (it == soundMap.end())
        {
            std::cerr << "Sound not loaded: " << file << std::endl;
            return quietSoundSource;
        }
        else
        {
            sound = it->second;
        }
    }



    SoundSource* s = &sources[oldestSource];
    if (s->isPlaying())
    {
        std::cout << "<SoundManager> Stopping sound before playing a new one!" << std::endl;
        s->stop();
    }
    s->reset(isMusic, isMusic ? musicVolume : effectsVolume);
    s->setSound(sound);
    oldestSource = std::max((oldestSource + 1) % maxSources, fixedSources);
    assert_no_alerror();
    return s;
}

SoundSource* SoundManager::getFixedSoundSource(const std::string& file, int id, bool isMusic)
{
    SAIGA_ASSERT(!parallelSoundLoaderRunning);

    Sound* sound = nullptr;


    auto it = soundMap.find(file);
    if (it == soundMap.end())
    {
        std::cerr << "Sound not loaded: " << file << std::endl;
        SAIGA_ASSERT(false);
        return quietSoundSource;
    }
    else
    {
        sound = it->second;
    }

    SoundSource* s = &sources[id];
    if (s->isPlaying())
    {
        s->stop();
    }
    s->reset(isMusic, isMusic ? musicVolume : effectsVolume);
    s->setSound(sound);
    assert_no_alerror();
    return s;
}

SoundSource* SoundManager::getFixedSoundSource(int id)
{
    return &sources[id];
}

void SoundManager::loadWaveSound(const std::string& file)
{
    SAIGA_ASSERT(!parallelSoundLoaderRunning);
    auto it = soundMap.find(file);
    if (it == soundMap.end())
    {
        SoundLoader sl;

        Sound* loadedsound;
        if ((loadedsound = sl.loadWaveFile(file)) != 0)
        {
            soundMap[file] = loadedsound;
            //            loadedsound->name = file;
        }
        else
        {
            std::cout << "Could not load sound: " << file << std::endl;
            SAIGA_ASSERT(0);
        }
    }
    else
    {
        std::cout << "Sound already loaded: " << file << std::endl;
        SAIGA_ASSERT(0);
    }
    assert_no_alerror();
}

#ifdef SAIGA_USE_OPUS
void SoundManager::loadOpusSound(const std::string& file)
{
    SAIGA_ASSERT(!parallelSoundLoaderRunning);
    auto it = soundMap.find(file);
    if (it == soundMap.end())
    {
        SoundLoader sl;

        Sound* loadedsound;
        if ((loadedsound = sl.loadOpusFile(file)) != 0)
        {
            soundMap[file] = loadedsound;
            //            loadedsound->name = file;
        }
        else
        {
            std::cout << "Could not load sound: " << file << std::endl;
            SAIGA_ASSERT(0);
        }
    }
    else
    {
        std::cout << "Sound already loaded: " << file << std::endl;
        SAIGA_ASSERT(0);
    }
    assert_no_alerror();
}
#endif

void SoundManager::loadSoundByEnding(const std::string& file)
{
    SAIGA_ASSERT(!parallelSoundLoaderRunning);

#ifdef SAIGA_USE_OPUS
    if (file.substr(file.find_last_of(".") + 1) == "opus")
    {
        loadOpusSound(file);
    }
    else
#endif
        if ((file.substr(file.find_last_of(".") + 1) == "wav"))
    {
        loadWaveSound(file);
    }
    else
    {
        std::cout << "Unknown file extension for sound file: " << file << std::endl;
        SAIGA_ASSERT(0);
    }
}

void SoundManager::unloadSound(const std::string& file)
{
    SAIGA_ASSERT(!parallelSoundLoaderRunning);
    SAIGA_ASSERT(soundMap.find(file) != soundMap.end());


    auto it = soundMap.find(file);
    if (it == soundMap.end())
    {
        std::cerr << "Sound to unload was not loaded: " << file << std::endl;
        SAIGA_ASSERT(false);
        return;
    }
    else
    {
        Sound* sound = it->second;
        delete sound;
        soundMap.erase(it);
    }
}

void SoundManager::addSoundToParallelQueue(const std::string& file)
{
    SAIGA_ASSERT(!parallelSoundLoaderRunning);
    soundQueue.push_back(file);
}

void SoundManager::addSoundToParallelQueueLock(const std::string& file)
{
    std::lock_guard<std::mutex> lock(soundQueueLock);  // scoped lock
    soundQueue.push_back(file);
}


bool SoundManager::soundAlreadyLoaded(const std::string& file) const
{
    std::lock_guard<std::mutex> lock(soundMapLock);  // scoped lock

    auto it = soundMap.find(file);
    if (it == soundMap.end())
    {
        return false;
    }
    else
    {
        std::cout << "Sound already loaded: " << file << std::endl;
        SAIGA_ASSERT(0);
        return true;
    }
}

void SoundManager::insertLoadedSoundIntoMap(const std::string& file, Sound* sound)
{
    std::lock_guard<std::mutex> lock(soundMapLock);  // scoped lock
    soundMap[file] = sound;
    //    sound->name = file;
}

void SoundManager::startParallelSoundLoader(int threadCount)
{
    //    SAIGA_ASSERT(!parallelSoundLoaderRunning);
    //    SAIGA_ASSERT(soundLoaderThreads.size() == 0);
    SAIGA_ASSERT(threadCount > 0);
    std::cout << "startParallelSoundLoader " << threadCount << std::endl;

    int oldthreadCount = this->threadCount;
    this->threadCount += threadCount;
    loadingDoneCounter += threadCount;
    parallelSoundLoaderRunning = true;
    for (int i = oldthreadCount; i < this->threadCount; ++i)
    {
        soundLoaderThreads.push_back(new std::thread(&SoundManager::loadSoundsThreadStart, this));
    }
}

void SoundManager::joinParallelSoundLoader()
{
    std::cout << "joinParallelSoundLoader " << std::endl;
    SAIGA_ASSERT(parallelSoundLoaderRunning);
    for (int i = 0; i < threadCount; ++i)
    {
        soundLoaderThreads[i]->join();
        delete soundLoaderThreads[i];
    }
    soundLoaderThreads.clear();
    parallelSoundLoaderRunning = false;
    this->threadCount          = 0;

    SAIGA_ASSERT(loadingDoneCounter.load() == 0);
}

bool SoundManager::isParallelLoadingDone()
{
    SAIGA_ASSERT(parallelSoundLoaderRunning);
    return loadingDoneCounter.load() == 0;
}

bool SoundManager::isParallelSoundLoaderNotJoined()
{
    return parallelSoundLoaderRunning;
}

void SoundManager::loadSoundsThreadStart()
{
    while (parallelSoundLoaderRunning)
    {
        bool took = false;
        soundQueueLock.lock();
        std::string f;
        if (!soundQueue.empty())
        {
            f    = soundQueue.front();
            took = true;
            soundQueue.pop_front();
        }

        soundQueueLock.unlock();

        if (took)
        {
            //            std::cout << "parallel loadsound " << f << std::endl;
            SAIGA_ASSERT(!soundAlreadyLoaded(f));

            SoundLoader sl;
            Sound* loadedsound = 0;
#ifdef SAIGA_USE_OPUS
            if (f.substr(f.find_last_of(".") + 1) == "opus")
            {
                loadedsound = sl.loadOpusFile(f);
            }
            else
#endif

                if ((f.substr(f.find_last_of(".") + 1) == "wav"))
            {
                loadedsound = sl.loadWaveFile(f);
            }
            else
            {
                std::cout << "Unknown file extension for sound file: " << f << std::endl;
                SAIGA_ASSERT(0);
            }


            if (loadedsound != 0)
            {
                insertLoadedSoundIntoMap(f, loadedsound);
            }
            else
            {
                std::cout << "Could not load sound (parallel): " << f << std::endl;
                SAIGA_ASSERT(0);
            }
        }
        else
        {
            //            std::cout << "sound list empty! joining... " << f << std::endl;
            break;
            //            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    loadingDoneCounter -= 1;
    std::cout << "sound loader thread finished " << std::endl;
}



void SoundManager::setListenerPosition(const vec3& pos)
{
    alListenerfv(AL_POSITION, &pos[0]);
    assert_no_alerror();
}

void SoundManager::setListenerVelocity(const vec3& velocity)
{
    alListenerfv(AL_VELOCITY, &velocity[0]);
    assert_no_alerror();
}

void SoundManager::setListenerOrientation(const vec3& at, const vec3& up)
{
    ALfloat listenerOri[6];
    listenerOri[0] = at[0];
    listenerOri[1] = at[1];
    listenerOri[2] = at[2];
    listenerOri[3] = up[0];
    listenerOri[4] = up[1];
    listenerOri[5] = up[2];
    alListenerfv(AL_ORIENTATION, listenerOri);
    assert_no_alerror();
}

void SoundManager::setListenerGain(float g)
{
    masterVolume = g;
    alListenerf(AL_GAIN, masterVolume);
    assert_no_alerror();
}

void SoundManager::setMusicVolume(float v)
{
    musicVolume = v;
    // update all sources
    for (SoundSource& s : sources)
    {
        if (s.isMusic())
        {
            s.setMasterVolume(musicVolume);
        }
    }
}

void SoundManager::setEffectsVolume(float v)
{
    effectsVolume = v;

    // update all sources
    for (SoundSource& s : sources)
    {
        if (!s.isMusic())
        {
            s.setMasterVolume(effectsVolume);
        }
    }
}

void SoundManager::setMute(bool b)
{
    muted = b;
    if (muted)
        alListenerf(AL_GAIN, 0);
    else
        alListenerf(AL_GAIN, masterVolume);
    assert_no_alerror();
}



void SoundManager::setTimeScale(float scale)
{
    // update all sources
    for (SoundSource& s : sources)
    {
        s.setPitch(scale);
    }
    assert_no_alerror();
}

void SoundManager::setTimeScaleNonFixed(float scale)
{
    for (int i = fixedSources; i < maxSources; ++i)
    {
        SoundSource& s = sources[i];
        s.setPitch(scale);
    }
    assert_no_alerror();
}

void SoundManager::startCapturing()
{
    const int SRATE = 44100;
    //    const int SSIZE = 1024;

    captureDevice = alcCaptureOpenDevice(NULL, SRATE, AL_FORMAT_STEREO16, SRATE / 2);
    SAIGA_ASSERT(captureDevice);
    alcCaptureStart(captureDevice);

    captureBuffer.resize(SRATE * 2 * 2);
    assert_no_alerror();
}

void SoundManager::stopCapturing()
{
    alcCaptureStop(captureDevice);
    alcCaptureCloseDevice(captureDevice);
    assert_no_alerror();
}

int SoundManager::getCapturedSamples()
{
    if (!captureDevice) return 0;

    ALint sampleCount;
    alcGetIntegerv(captureDevice, ALC_CAPTURE_SAMPLES, 4, &sampleCount);

    alcCaptureSamples(captureDevice, (ALCvoid*)captureBuffer.data(), sampleCount);
    //    std::cout << "captured "<<sampleCount << " samples"<<endl;
    assert_no_alerror();

    return sampleCount;
}


}  // namespace sound
}  // namespace Saiga
