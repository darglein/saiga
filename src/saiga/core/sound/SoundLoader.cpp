/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "SoundLoader.h"

#include "saiga/core/util/assert.h"

#include <al.h>
#include <alc.h>
#include <fstream>
#include <iostream>

#ifdef SAIGA_USE_ALUT
#    include <alut.h>
#endif

#ifdef SAIGA_USE_OPUS
#    include "OpusCodec.h"
#    include "opusfile.h"
#endif


#include "internal/noGraphicsAPI.h"

#include <cstdint>
#include <cstring>

namespace Saiga
{
namespace sound
{
Sound* SoundLoader::loadWaveFile(const std::string& filename)
{
#ifdef SAIGA_USE_ALUT123
    return loadWaveFileALUT(filename);
#else
    return loadWaveFileRaw(filename);
#endif
}

void readDecode(std::ifstream& stream, void* dst, int size, int offset)
{
    std::vector<char> bytes(size);
    stream.read(&bytes[0], size);

    for (char& c : bytes)
    {
        c -= offset;
    }
    std::memcpy(dst, bytes.data(), size);
}

// http://www.dunsanyinteractive.com/blogs/oliver/?p=72
/*
 * Load wave file function. No need for ALUT with this
 */
Sound* SoundLoader::loadWaveFileRaw(const std::string& filename)
{
    //    std::cout << "loadWaveFileRaw " << filename << std::endl;
    int allowedOffset = 0x42;
    int offset        = 0;

    WAVE_Format wave_format;
    RIFF_Header riff_header;
    WAVE_Data wave_data;

    std::ifstream stream(filename, std::ifstream::binary);
    if (!stream.is_open())
    {
        std::cout << "Could not open file " << filename << std::endl;
        return nullptr;
    }

    // Read in the first chunk into the struct
    stream.read((char*)&riff_header, sizeof(RIFF_Header));

    // check for RIFF and WAVE tag in memeory
    if (riff_header.chunkID[0] == 'R' && riff_header.chunkID[1] == 'I' && riff_header.chunkID[2] == 'F' &&
        riff_header.chunkID[3] == 'F' && riff_header.format[0] == 'W' && riff_header.format[1] == 'A' &&
        riff_header.format[2] == 'V' && riff_header.format[3] == 'E')
    {
        // normal riff wave header.
        offset = 0;
    }
    else if (riff_header.chunkID[0] == 'R' + allowedOffset && riff_header.chunkID[1] == 'I' + allowedOffset &&
             riff_header.chunkID[2] == 'F' + allowedOffset && riff_header.chunkID[3] == 'F' + allowedOffset &&
             riff_header.format[0] == 'W' + allowedOffset && riff_header.format[1] == 'A' + allowedOffset &&
             riff_header.format[2] == 'V' + allowedOffset && riff_header.format[3] == 'E' + allowedOffset)
    {
        //'encoded' wave header.
        offset = 0x42;
        //        std::cout << "found encoded riff wave header! " << std::endl;
        //        return nullptr;
    }
    else
    {
        //        std::cout << (int)riff_header.chunkID[0] << " " << " " << (int) 'R' << " " <<  (int)'R' +
        //        allowedOffset << " " << (int)(riff_header.chunkID[0] - (char)allowedOffset) <<  std::endl;
        std::cout << "Invalid RIFF or WAVE Header" << std::endl;
        return nullptr;
    }

    // Read in the 2nd chunk for the wave info
    //    stream.read( (char*)&wave_format,sizeof(WAVE_Format));
    readDecode(stream, &wave_format, sizeof(WAVE_Format), offset);
    // check for fmt tag in memory
    if (wave_format.subChunkID[0] != 'f' || wave_format.subChunkID[1] != 'm' || wave_format.subChunkID[2] != 't' ||
        wave_format.subChunkID[3] != ' ')
    {
        std::cout << "Invalid Wave Format" << std::endl;
        return nullptr;
    }

    // check for extra parameters;
    if (wave_format.subChunkSize > 16)
    {
        short bla;  // ignore them
        stream.read((char*)&bla, sizeof(short));
    }

    // Read in the the last byte of data before the sound file
    //    stream.read( (char*)&wave_data,sizeof(WAVE_Data));

    readDecode(stream, &wave_data, sizeof(WAVE_Data), offset);

    // check for data tag in memory
    if (wave_data.subChunkID[0] != 'd' || wave_data.subChunkID[1] != 'a' || wave_data.subChunkID[2] != 't' ||
        wave_data.subChunkID[3] != 'a')
    {
        std::cout << "Invalid data header" << std::endl;
        return nullptr;
    }
    //        std::cout << "size of data: " << wave_data.subChunk2Size << std::endl;

    std::vector<unsigned char> data(wave_data.subChunk2Size);

    // Read in the sound data into the soundData variable
    //    stream.read( (char*)data.data(),wave_data.subChunk2Size);

    readDecode(stream, data.data(), wave_data.subChunk2Size, offset);

    // Now we set the variables that we passed in with the
    // data from the structs

    Sound* sound = new Sound();
    sound->name  = filename;
    sound->setFormat(wave_format.numChannels, wave_format.bitsPerSample, wave_format.sampleRate);

    if (!sound->checkFirstSample(data.data()))
    {
        int numberOfZeroSamples = 2;

        int bytes = wave_format.bitsPerSample / 8 * wave_format.numChannels * numberOfZeroSamples;
        std::vector<unsigned char> zerobytes(bytes, 0);
        data.insert(data.begin(), zerobytes.begin(), zerobytes.end());
#if defined(SAIGA_DEBUG)
        std::cerr << "Inserting " << bytes << " zero padding bytes (" << numberOfZeroSamples
                  << " samples) at the beginning of sound " << sound->name << std::endl;
#endif
    }

    sound->createBuffer(data.data(), wave_data.subChunk2Size);

    return sound;
}

#ifdef SAIGA_USE_OPUS
Sound* SoundLoader::loadOpusFile(const std::string& filename)
{
    // The <tt>libopusfile</tt> API always decodes files to 48kHz.
    // The original sample rate is not preserved by the lossy compression.
    const int sampleRate = 48000;

    int error;
    OggOpusFile* file = op_open_file(filename.c_str(), &error);
    if (error)
    {
        std::cout << "could not open file: " << filename << std::endl;
        return nullptr;
    }

    int linkCount = op_link_count(file);
    SAIGA_ASSERT(linkCount == 1);
    int currentLink = op_current_link(file);
    //    int bitRate = op_bitrate(file,currentLink); //TODO
    //    int total = op_raw_total(file,currentLink);
    //    int pcmtotal = op_pcm_total(file,currentLink);
    int channels = op_channel_count(file, currentLink);
    SAIGA_ASSERT(channels == 1 || channels == 2);

    std::vector<opus_int16> data;

    std::vector<opus_int16> readBuf(10000);
    int readSamples = 0;
    do
    {
        readSamples = op_read(file, readBuf.data(), readBuf.size(), &currentLink);
        data.insert(data.end(), readBuf.begin(), readBuf.begin() + readSamples * 2);

    } while (readSamples > 0);

    op_free(file);

    Sound* sound = new Sound();
    sound->name  = filename;
    sound->setFormat(channels, 16, sampleRate);

    sound->createBuffer(data.data(), data.size() * sizeof(opus_int16));


    //     std::cout<<"Loaded opus file: "<<filename<<" ( "<<"bitRate="<<bitRate<<" samplestotal="<<data.size() <<"
    //     channels="<<channels<<" )"<<endl;
    return sound;
}
#endif

#ifdef SAIGA_USE_ALUT
Sound* SoundLoader::loadWaveFileALUT(const std::string& filename)
{
    ALuint buffer = alutCreateBufferFromFile(filename.c_str());

    if (buffer == 0)
    {
        ALenum e = alutGetError();
        std::cout << "Could not load " << filename << "! (Error: " << e << ")" << std::endl;
        switch (e)
        {
            case ALUT_ERROR_AL_ERROR_ON_ENTRY:
                std::cout << "ALUT_ERROR_AL_ERROR_ON_ENTRY" << std::endl;
                break;
            case ALUT_ERROR_ALC_ERROR_ON_ENTRY:
                std::cout << "ALUT_ERROR_ALC_ERROR_ON_ENTRY" << std::endl;
                break;
            case ALUT_ERROR_BUFFER_DATA:
                std::cout << "ALUT_ERROR_BUFFER_DATA" << std::endl;
                break;
            case ALUT_ERROR_CORRUPT_OR_TRUNCATED_DATA:
                std::cout << "ALUT_ERROR_CORRUPT_OR_TRUNCATED_DATA" << std::endl;
                break;
            case ALUT_ERROR_GEN_BUFFERS:
                std::cout << "ALUT_ERROR_GEN_BUFFERS" << std::endl;
                break;
            case ALUT_ERROR_INVALID_OPERATION:
                std::cout << "ALUT_ERROR_INVALID_OPERATION" << std::endl;
                break;
            case ALUT_ERROR_IO_ERROR:
                std::cout << "ALUT_ERROR_IO_ERROR" << std::endl;
                break;
            case ALUT_ERROR_NO_CURRENT_CONTEXT:
                std::cout << "ALUT_ERROR_NO_CURRENT_CONTEXT" << std::endl;
                break;
            case ALUT_ERROR_OUT_OF_MEMORY:
                std::cout << "ALUT_ERROR_OUT_OF_MEMORY" << std::endl;
                break;
            case ALUT_ERROR_UNSUPPORTED_FILE_SUBTYPE:
                std::cout << "ALUT_ERROR_UNSUPPORTED_FILE_SUBTYPE" << std::endl;
                break;
            case ALUT_ERROR_UNSUPPORTED_FILE_TYPE:
                std::cout << "ALUT_ERROR_UNSUPPORTED_FILE_TYPE" << std::endl;
                break;
            case 519:
                std::cout << "Failed to open OpenAL Device, maybe reinstall OpenAL!" << std::endl;
                break;
        }


        SAIGA_ASSERT(0);
        return nullptr;
    }


    Sound* sound  = new Sound();
    sound->buffer = buffer;

    //    alGetBufferi(buffer, AL_SIZE, &sound->size);
    alGetBufferi(buffer, AL_CHANNELS, &sound->channels);
    alGetBufferi(buffer, AL_BITS, &sound->bitsPerSample);
    alGetBufferi(buffer, AL_FREQUENCY, &sound->frequency);

    return sound;
}
#endif

}  // namespace sound
}  // namespace Saiga
