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
    int maxSources;
    int oldestSource = 0;
public:

    SoundManager (int maxSources);
    virtual ~SoundManager ();

    void setListenerPosition(const vec3& pos);
    void setListenerVelocity(const vec3& velocity);
    void setListenerOrientation(const vec3& at, const vec3& up);
    void setListenerGain(float g);

    void setMute(bool b);

    void setTimeScale(float scale);


    SoundSource *getSoundSource(const std::string &file);
    void loadSound(const std::string &file);

    void checkForSoundErrors();

    static std::string getALCErrorString(int err);



};


}

