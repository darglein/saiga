/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include "Sound.h"

namespace Saiga
{
namespace sound
{
struct SAIGA_LOCAL RIFF_Header
{
    unsigned char chunkID[4];
    int chunkSize;  // size not including chunkSize or chunkID
    unsigned char format[4];
};

/*
 * Struct to hold fmt subchunk data for WAVE files.
 */
struct SAIGA_LOCAL WAVE_Format
{
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
struct SAIGA_LOCAL WAVE_Data
{
    char subChunkID[4];  // should contain the word data
    int subChunk2Size;   // Stores the size of the data block
};


class SAIGA_CORE_API SoundLoader
{
   public:
    // loads with alut if possible
    Sound* loadWaveFile(const std::string& filename);



    Sound* loadWaveFileRaw(const std::string& filename);
#ifdef SAIGA_USE_ALUT
    // use alut for sound loading
    Sound* loadWaveFileALUT(const std::string& filename);
#endif

#ifdef SAIGA_USE_OPUS
    Sound* loadOpusFile(const std::string& filename);
#endif
};

}  // namespace sound
}  // namespace Saiga
