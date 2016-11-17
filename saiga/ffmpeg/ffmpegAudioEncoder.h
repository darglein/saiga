#pragma once

#include "saiga/config.h"
#include "saiga/opengl/texture/image.h"
#include "saiga/time/timer.h"
#include "saiga/util/synchronizedBuffer.h"
#include <string>
#include <iostream>
#include <fstream>
#include <thread>

//ffmpeg is compiled with a pure c compiler, so all includes need an 'extern "C"'.
extern "C"{
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libswscale/swscale.h>
}

class SAIGA_GLOBAL FFMPEGAudioEncoder{
private:


    float t, tincr;
    AVCodec *codec;
    AVCodecContext *c= NULL;
    AVFrame *frame;
    AVPacket pkt;
    int i, j, k, ret, got_output;
    int buffer_size;
    int bytesPerSample;
//    uint16_t *samples;
    int currentSamples = 0;
    std::vector<unsigned char> frameBuffer;

public:
    std::ofstream  outputStream;

    FFMPEGAudioEncoder();

    void startEncoding(const std::string &filename);

    void addFrame(std::vector<unsigned char> &soundSamples, int samples);
    void finishEncoding();
    void addFrame();
};
