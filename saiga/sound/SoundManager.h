#pragma once

#include <string>
#include <vector>
#include <map>

#include <saiga/util/glm.h>
#include <saiga/sound/SoundSource.h>

namespace sound {
class SoundSource;
class Sound;


extern void initOpenAL();
extern void quitOpenAL();


class SAIGA_GLOBAL SoundManager
{
private:
    SoundSource quietSoundSource;
    std::vector<SoundSource> sources;
    int sourceIndex = 0;

    std::map<std::string,Sound*> soundMap;

    float masterVolumne = 1.0f;

    bool muted = false;
    int maxSources, fixedSources;
    int oldestSource = 0;
public:

    SoundManager (int maxSources, int fixedSources=0);
    virtual ~SoundManager ();

    void setListenerPosition(const vec3& pos);
    void setListenerVelocity(const vec3& velocity);
    void setListenerOrientation(const vec3& at, const vec3& up);
    void setListenerGain(float g);

    void setMute(bool b);

    void setTimeScale(float scale);


    SoundSource *getSoundSource(const std::string &file);
    SoundSource *getFixedSoundSource(const std::string &file, int id);
    SoundSource *getFixedSoundSource(int id);

    void loadSound(const std::string &file);

    void checkForSoundErrors();

    static std::string getALCErrorString(int err);



};


}

