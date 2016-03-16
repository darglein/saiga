#include "saiga/sound/SoundLoader.h"
#include "saiga/util/assert.h"
#include <fstream>

#include <AL/al.h>
#include <AL/alc.h>

#ifdef USE_ALUT
#include <AL/alut.h>
#endif

#ifdef USE_OPUS
#include "saiga/sound/OpusCodec.h"
#endif

#include "opusfile.h"
#include <cstdint>
#include <cstring>
namespace sound {


Sound* SoundLoader::loadWaveFile(const std::string &filename){
#ifdef USE_ALUT
    return loadWaveFileALUT(filename);
#else
    return loadWaveFileRaw(filename);
#endif
}

//http://www.dunsanyinteractive.com/blogs/oliver/?p=72
/*
 * Load wave file function. No need for ALUT with this
 */
Sound* SoundLoader::loadWaveFileRaw(const std::string &filename) {
    //Local Declarations
    FILE* soundFile = NULL;
    WAVE_Format wave_format;
    RIFF_Header riff_header;
    WAVE_Data wave_data;
    unsigned char* data;

    Sound* sound = new Sound();

    try {
        soundFile = fopen(filename.c_str(), "rb");
        if (!soundFile)
            throw (std::string(filename));

        // Read in the first chunk into the struct
        size_t s = fread(&riff_header, sizeof(RIFF_Header), 1, soundFile);
        (void)s;

        //check for RIFF and WAVE tag in memeory
        if ((riff_header.chunkID[0] != 'R' ||
             riff_header.chunkID[1] != 'I' ||
             riff_header.chunkID[2] != 'F' ||
             riff_header.chunkID[3] != 'F') ||
                (riff_header.format[0] != 'W' ||
                 riff_header.format[1] != 'A' ||
                 riff_header.format[2] != 'V' ||
                 riff_header.format[3] != 'E'))
            throw (std::string("Invalid RIFF or WAVE Header"));

        //Read in the 2nd chunk for the wave info
        s = fread(&wave_format, sizeof(WAVE_Format), 1, soundFile);
        //check for fmt tag in memory
        if (wave_format.subChunkID[0] != 'f' ||
                wave_format.subChunkID[1] != 'm' ||
                wave_format.subChunkID[2] != 't' ||
                wave_format.subChunkID[3] != ' ')
            throw (std::string("Invalid Wave Format"));

        //check for extra parameters;
        if (wave_format.subChunkSize > 16)
            fseek(soundFile, sizeof(short), SEEK_CUR);

        //Read in the the last byte of data before the sound file
        s = fread(&wave_data, sizeof(WAVE_Data), 1, soundFile);


        //check for data tag in memory
        if (wave_data.subChunkID[0] != 'd' ||
                wave_data.subChunkID[1] != 'a' ||
                wave_data.subChunkID[2] != 't' ||
                wave_data.subChunkID[3] != 'a')
            throw (std::string("Invalid data header"));

        //Allocate memory for data
        data = new unsigned char[wave_data.subChunk2Size];

        // Read in the sound data into the soundData variable
        if (!fread(data, wave_data.subChunk2Size, 1, soundFile))
            throw (std::string("error loading WAVE data into struct!"));

        //Now we set the variables that we passed in with the
        //data from the structs
        sound->size = wave_data.subChunk2Size;
        sound->frequency = wave_format.sampleRate;
        //The format is worked out by looking at the number of
        //channels and the bits per sample.
        if (wave_format.numChannels == 1) {
            if (wave_format.bitsPerSample == 8 )
                sound->format = AL_FORMAT_MONO8;
            else if (wave_format.bitsPerSample == 16)
                sound->format = AL_FORMAT_MONO16;
        } else if (wave_format.numChannels == 2) {
            if (wave_format.bitsPerSample == 8 )
                sound->format = AL_FORMAT_STEREO8;
            else if (wave_format.bitsPerSample == 16)
                sound->format = AL_FORMAT_STEREO16;
        }
        //create our openAL buffer and check for success
        alGenBuffers(1, &sound->buffer);
        //        checkForSoundErrors();
        //now we put our data into the openAL buffer and
        //check for success
        alBufferData(sound->buffer, sound->format, (void*)data,
                     sound->size, sound->frequency);
        //        checkForSoundErrors();
        //clean up and return true if successful
        delete[] data;
        fclose(soundFile);
        return sound;
    } catch(const std::string &error) {
        //our catch statement for if we throw a string
        std::cerr << error << " : trying to load "
                  << filename << std::endl;
        if (soundFile != NULL)
            fclose(soundFile);

        return nullptr;
    }
}

#ifdef USE_OPUS
Sound *SoundLoader::loadOpusFile(const std::string &filename)
{

    int sampleRate = 48000;

    int error;
    OggOpusFile * file = op_open_file(filename.c_str(), &error);
    if(error){
        cout<<"could not open file: "<<filename<<endl;
        return nullptr;
    }

    int linkCount = op_link_count(file);
    assert(linkCount==1);
    int currentLink = op_current_link(file);
    int bitRate = op_bitrate(file,currentLink);
//    int total = op_raw_total(file,currentLink);
//    int pcmtotal = op_pcm_total(file,currentLink);
    int channels = op_channel_count(file,currentLink);
    assert(channels==1 || channels==2);

    std::vector<opus_int16> data;

    std::vector<opus_int16> readBuf(10000);
    int readSamples = 0;
    do{
        readSamples = op_read(file,readBuf.data(),readBuf.size(),&currentLink);
        data.insert(data.end(),readBuf.begin(),readBuf.begin()+readSamples*2);

    }while(readSamples>0);

    op_free(file);

    Sound* sound = new Sound();
    sound->size = data.size()*sizeof(opus_int16);
    sound->frequency = sampleRate;


    if (channels == 1) {
        sound->format = AL_FORMAT_MONO16;
    } else if (channels == 2) {
        sound->format = AL_FORMAT_STEREO16;
    }

    alGenBuffers(1, &sound->buffer);
    alBufferData(sound->buffer, sound->format, (void*)data.data(),
                 sound->size, sound->frequency);

    cout<<"Loaded opus file: "<<filename<<" ( "<<"bitRate="<<bitRate<<" memorydecoded="<<sound->size <<" channels="<<channels<<" )"<<endl;
    return sound;
}
#endif

#ifdef USE_ALUT
Sound *SoundLoader::loadWaveFileALUT(const std::string &filename)
{
    ALuint buffer = alutCreateBufferFromFile(filename.c_str());

    if(buffer==0){
        ALenum e = alutGetError();
        std::cout<<"Could not load "<<filename<<"! (Error: " << e << ")" <<std::endl;
        switch (e){
        case ALUT_ERROR_AL_ERROR_ON_ENTRY:
            cout << "ALUT_ERROR_AL_ERROR_ON_ENTRY" << endl;
            break;
        case ALUT_ERROR_ALC_ERROR_ON_ENTRY:
            cout << "ALUT_ERROR_ALC_ERROR_ON_ENTRY" << endl;
            break;
        case ALUT_ERROR_BUFFER_DATA:
            cout << "ALUT_ERROR_BUFFER_DATA" << endl;
            break;
        case ALUT_ERROR_CORRUPT_OR_TRUNCATED_DATA:
            cout << "ALUT_ERROR_CORRUPT_OR_TRUNCATED_DATA" << endl;
            break;
        case ALUT_ERROR_GEN_BUFFERS:
            cout << "ALUT_ERROR_GEN_BUFFERS" << endl;
            break;
        case ALUT_ERROR_INVALID_OPERATION:
            cout << "ALUT_ERROR_INVALID_OPERATION" << endl;
            break;
        case ALUT_ERROR_IO_ERROR:
            cout << "ALUT_ERROR_IO_ERROR" << endl;
            break;
        case ALUT_ERROR_NO_CURRENT_CONTEXT:
            cout << "ALUT_ERROR_NO_CURRENT_CONTEXT" << endl;
            break;
        case ALUT_ERROR_OUT_OF_MEMORY:
            cout << "ALUT_ERROR_OUT_OF_MEMORY" << endl;
            break;
        case ALUT_ERROR_UNSUPPORTED_FILE_SUBTYPE:
            cout << "ALUT_ERROR_UNSUPPORTED_FILE_SUBTYPE" << endl;
            break;
        case ALUT_ERROR_UNSUPPORTED_FILE_TYPE:
            cout << "ALUT_ERROR_UNSUPPORTED_FILE_TYPE" << endl;
            break;
        case 519:
            cout << "Failed to open OpenAL Device, maybe reinstall OpenAL!" << endl;
            break;

        }


        assert(0);
        return nullptr;
    }


    Sound* sound = new Sound();
    sound->buffer = buffer;

    alGetBufferi(buffer, AL_SIZE, &sound->size);
    alGetBufferi(buffer, AL_CHANNELS, &sound->channels);
    alGetBufferi(buffer, AL_BITS, &sound->bits);
    alGetBufferi(buffer, AL_FREQUENCY, &sound->frequency);

    return sound;
}
#endif


}
