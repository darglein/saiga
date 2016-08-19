#pragma once

#include "saiga/config.h"
#include "saiga/opengl/texture/image.h"
#include <string>
#include <iostream>
#include <fstream>

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

class SAIGA_GLOBAL FFMPEGEncoder{
private:
    int currentFrame = 0;
    AVCodec *codec;
    AVCodecContext *c= NULL;
    int ret, got_output;
    AVFrame *frame;
    AVPacket pkt;
    SwsContext * ctx = nullptr;
public:
    std::ofstream  outputStream;

    FFMPEGEncoder();

    void startEncoding(const std::string &filename, int outWidth, int outHeight, int outFps, int bitRate);
    void addFrame( Image& image);
    void finishEncoding();
};
