#pragma once

#include <string>


#include <libhello/config.h>
#include <libhello/util/glm.h>
#include <libhello/sound/Sound.h>


namespace sound {



struct RIFF_Header {
  char chunkID[4];
  int chunkSize;//size not including chunkSize or chunkID
  char format[4];
};

/*
 * Struct to hold fmt subchunk data for WAVE files.
 */
struct WAVE_Format {
  char subChunkID[4];
  int subChunkSize;
  short audioFormat;
  short numChannels;
  int sampleRate;
  int byteRate;
  short blockAlign;
  short bitsPerSample;
};

/*
* Struct to hold the data of the wave file
*/
struct WAVE_Data {
  char subChunkID[4]; //should contain the word data
  int subChunk2Size; //Stores the size of the data block
};


class SoundLoader{

public:

    //loads with alut if possible
    Sound* loadSound(const std::string &filename);



    Sound* loadWaveFile(const std::string &filename);
#ifdef USE_ALUT
    //use alut for sound loading
    Sound* loadSoundALUT(const std::string &filename);
#endif
};

}


