/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/core/math/math.h"

#include "OpenAL.h"
#include "SoundSource.h"

#include <atomic>
#include <list>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include <alc.h>

namespace Saiga
{
namespace sound
{
class SoundSource;
class Sound;



/**
 * @brief The SoundManager class
 * It generates "maxSources" OpenAL soundsources and reuses them cyclic.
 * Be careful, on reuse a "clicking noise" may be heard, that happens probably because the sound does not start with
 * some quiet samples. Fix 1: Add silence on the start of the sound. Fix 2: Rewrite this class to not reuse sound
 * sources but generate new ones, but care for performance!
 */
class SAIGA_CORE_API SoundManager
{
   private:
    SoundSource* quietSoundSource;
    std::vector<SoundSource> sources;
    //    int sourceIndex = 0;

    std::map<std::string, Sound*> soundMap;

    float masterVolume  = 1.0f;
    float musicVolume   = 1.f;
    float effectsVolume = 1.f;

    bool muted = false;
    int maxSources, fixedSources;
    int oldestSource = 0;
    bool soundAlreadyLoaded(const std::string& file) const;
    void insertLoadedSoundIntoMap(const std::string& file, Sound* sound);
    void loadSoundsThreadStart();

    int threadCount = 0;
    std::vector<std::thread*> soundLoaderThreads;

    std::list<std::string> soundQueue;
    mutable std::mutex soundQueueLock;
    mutable std::mutex soundMapLock;

    bool parallelSoundLoaderRunning     = false;
    std::atomic<int> loadingDoneCounter = {0};

   public:
    SoundManager(int maxSources, int fixedSources = 0);
    virtual ~SoundManager();

    void setListenerPosition(const vec3& pos);
    void setListenerVelocity(const vec3& velocity);
    void setListenerOrientation(const vec3& at, const vec3& up);
    void setListenerGain(float g);
    void setMusicVolume(float v);
    void setEffectsVolume(float v);

    void setMute(bool b);

    void setTimeScale(float scale);
    void setTimeScaleNonFixed(float scale);


    /**
     * @brief getSoundSource
     * Fast function of getting a sound source, can not be called if the sound loaders are still working in parallel
     */
    SoundSource* getSoundSource(const std::string& file, bool isMusic = false);
    SoundSource* getFixedSoundSource(const std::string& file, int id, bool isMusic = false);
    SoundSource* getFixedSoundSource(int id);

    /**
     * @brief getSoundSourceWhileStillLoading
     * Thread safe function of getting a sound source while the parallel sound loaders are still working, may return a
     * quiet sound source if the sound is not loaded
     */
    SoundSource* getSoundSourceWhileStillLoading(const std::string& file, bool isMusic = false);

    void loadWaveSound(const std::string& file);

#ifdef SAIGA_USE_OPUS
    void loadOpusSound(const std::string& file);
#endif

    void loadSoundByEnding(const std::string& file);

    void unloadSound(const std::string& file);

    void addSoundToParallelQueue(const std::string& file);
    void startParallelSoundLoader(int threadCount);
    void joinParallelSoundLoader();
    bool isParallelLoadingDone();
    bool isParallelSoundLoaderNotJoined();
    void addSoundToParallelQueueLock(const std::string& file);


    // capturing
    ALCdevice* captureDevice = nullptr;
    std::vector<unsigned char> captureBuffer;
    void startCapturing();
    void stopCapturing();
    int getCapturedSamples();
};


}  // namespace sound
}  // namespace Saiga
