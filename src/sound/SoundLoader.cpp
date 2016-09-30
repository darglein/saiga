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

#include "opus/opusfile.h"


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

void readDecode(std::ifstream &stream, void* dst, int size, int offset){

    std::vector<char> bytes(size);
    stream.read(&bytes[0], size);

    for(char& c : bytes){
        c -= offset;
    }
    std::memcpy(dst,bytes.data(),size);
}

//http://www.dunsanyinteractive.com/blogs/oliver/?p=72
/*
 * Load wave file function. No need for ALUT with this
 */
Sound* SoundLoader::loadWaveFileRaw(const std::string &filename) {
    //    cout << "loadWaveFileRaw " << filename << endl;
    int allowedOffset = 0x42;
    int offset = 0;

    WAVE_Format wave_format;
    RIFF_Header riff_header;
    WAVE_Data wave_data;

    std::ifstream stream (filename,std::ifstream::binary);
    if(!stream.is_open()){
        cout << "Could not open file " << filename << endl;
        return nullptr;
    }

    // Read in the first chunk into the struct
    stream.read( (char*)&riff_header,sizeof(RIFF_Header));

    //check for RIFF and WAVE tag in memeory
    if (riff_header.chunkID[0] == 'R' &&
            riff_header.chunkID[1] == 'I' &&
            riff_header.chunkID[2] == 'F' &&
            riff_header.chunkID[3] == 'F' &&
            riff_header.format[0] == 'W' &&
            riff_header.format[1] == 'A' &&
            riff_header.format[2] == 'V' &&
            riff_header.format[3] == 'E'){
        //normal riff wave header.
        offset = 0;
    }else  if (riff_header.chunkID[0] == 'R' + allowedOffset &&
               riff_header.chunkID[1] == 'I' + allowedOffset &&
               riff_header.chunkID[2] == 'F' + allowedOffset &&
               riff_header.chunkID[3] == 'F' + allowedOffset &&
               riff_header.format[0] == 'W' + allowedOffset &&
               riff_header.format[1] == 'A' + allowedOffset &&
               riff_header.format[2] == 'V' + allowedOffset &&
               riff_header.format[3] == 'E' + allowedOffset){
        //'encoded' wave header.
        offset = 0x42;
//        cout << "found encoded riff wave header! " << endl;
//        return nullptr;
    }else{
//        cout << (int)riff_header.chunkID[0] << " " << " " << (int) 'R' << " " <<  (int)'R' + allowedOffset << " " << (int)(riff_header.chunkID[0] - (char)allowedOffset) <<  endl;
        cout << "Invalid RIFF or WAVE Header" << endl;
        return nullptr;
    }

    //Read in the 2nd chunk for the wave info
//    stream.read( (char*)&wave_format,sizeof(WAVE_Format));
    readDecode(stream,&wave_format,sizeof(WAVE_Format),offset);
    //check for fmt tag in memory
    if (wave_format.subChunkID[0] != 'f' ||
            wave_format.subChunkID[1] != 'm' ||
            wave_format.subChunkID[2] != 't' ||
            wave_format.subChunkID[3] != ' '){
        cout << "Invalid Wave Format" << endl;
        return nullptr;
    }

    //check for extra parameters;
    if (wave_format.subChunkSize > 16){
        short bla; //ignore them
        stream.read( (char*)&bla,sizeof(short));
    }

    //Read in the the last byte of data before the sound file
//    stream.read( (char*)&wave_data,sizeof(WAVE_Data));

    readDecode(stream,&wave_data,sizeof(WAVE_Data),offset);

    //check for data tag in memory
    if (wave_data.subChunkID[0] != 'd' ||
            wave_data.subChunkID[1] != 'a' ||
            wave_data.subChunkID[2] != 't' ||
            wave_data.subChunkID[3] != 'a'){
        cout << "Invalid data header" << endl;
        return nullptr;
    }
    //        cout << "size of data: " << wave_data.subChunk2Size << endl;

    std::vector<unsigned char> data(wave_data.subChunk2Size);

    // Read in the sound data into the soundData variable
//    stream.read( (char*)data.data(),wave_data.subChunk2Size);

    readDecode(stream,data.data(),wave_data.subChunk2Size,offset);

    //Now we set the variables that we passed in with the
    //data from the structs

    Sound* sound = new Sound();
    sound->name = filename;
    sound->setFormat(wave_format.numChannels,wave_format.bitsPerSample,wave_format.sampleRate);

	if (!sound->checkFirstSample(data.data())) {
		int bytes = wave_format.bitsPerSample / 8 * wave_format.numChannels;
		std::vector<unsigned char> zerobytes(bytes, 0);
		data.insert(data.begin(), zerobytes.begin(), zerobytes.end());
#if defined(SAIGA_DEBUG)
		std::cerr << "Inserting " << bytes << " zero padding bytes at the beginning of sound " << sound->name << std::endl;
#endif
	}

    sound->createBuffer(data.data(),wave_data.subChunk2Size);

    return sound;
}

#ifdef USE_OPUS
Sound *SoundLoader::loadOpusFile(const std::string &filename)
{

    // The <tt>libopusfile</tt> API always decodes files to 48kHz.
    // The original sample rate is not preserved by the lossy compression.
    const int sampleRate = 48000;

    int error;
    OggOpusFile * file = op_open_file(filename.c_str(), &error);
    if(error){
        cout<<"could not open file: "<<filename<<endl;
        return nullptr;
    }

    int linkCount = op_link_count(file);
    assert(linkCount==1);
    int currentLink = op_current_link(file);
//    int bitRate = op_bitrate(file,currentLink); //TODO
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
    sound->name = filename;
    sound->setFormat(channels,16,sampleRate);

    sound->createBuffer(data.data(),data.size()*sizeof(opus_int16));


//     cout<<"Loaded opus file: "<<filename<<" ( "<<"bitRate="<<bitRate<<" samplestotal="<<data.size() <<" channels="<<channels<<" )"<<endl;
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
