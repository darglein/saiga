/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/image/image.h"
#include "saiga/time/timer.h"
#include "saiga/util/synchronizedBuffer.h"


#include <fstream>
#include <thread>


//ffmpeg is compiled with a pure c compiler, so all includes need an 'extern "C"'.
extern "C"{
struct AVFrame;
struct AVCodecContext;
struct AVFormatContext;
struct SwsContext;
#include <libavcodec/avcodec.h>
}

namespace Saiga {

class SAIGA_GLOBAL FFMPEGEncoder
{
public:
    using EncoderImageType = TemplatedImage<ucvec4>;

    //Recommended codecs and container formats:
    //  .mp4    AV_CODEC_ID_H264
    //  .mpeg   AV_CODEC_ID_MPEG2VIDEO or AV_CODEC_ID_MPEG4
    //  .avi    AV_CODEC_ID_RAWVIDEO
    FFMPEGEncoder(const std::string &filename, int outWidth, int outHeight, int inWidth, int inHeight, int outFps, int bitRate,AVCodecID videoCodecId=AV_CODEC_ID_NONE, int bufferSize = 50);
    ~FFMPEGEncoder();


    void addFrame(std::shared_ptr<EncoderImageType> image);
    std::shared_ptr<EncoderImageType> getFrameBuffer();


private:


    void startEncoding(const std::string &filename, int outWidth, int outHeight, int inWidth, int inHeight, int outFps, int bitRate,AVCodecID videoCodecId=AV_CODEC_ID_NONE);
    void finishEncoding();


    int outWidth, outHeight, inWidth, inHeight;

    SynchronizedBuffer<std::shared_ptr<EncoderImageType>> imageStorage;
    SynchronizedBuffer<std::shared_ptr<EncoderImageType>> imageQueue;

    SynchronizedBuffer<AVFrame*> frameStorage;
    SynchronizedBuffer<AVFrame*> frameQueue;

    std::thread scaleThread; //scales and converts the image to the correct size and color format
    std::thread encodeThread; //does the actual encoding

    volatile int currentFrame = 0;
    volatile int finishedFrames = 0;

    volatile bool running = false;
    volatile bool finishScale = false;
    volatile bool finishEncode = false;
    //    AVCodec *codec;
    //    AVCodecContext *c= NULL;

    AVCodecContext *m_codecContext;
    AVFormatContext* m_formatCtx;
    int ticksPerFrame;
    //AVPacket pkt;
    SwsContext * ctx = nullptr;
    void scaleFrame(std::shared_ptr<EncoderImageType> image, AVFrame *frame);
    bool encodeFrame(AVFrame *frame);
    bool encodeFrame();
    bool scaleFrame();
    void scaleThreadFunc();
    void encodeThreadFunc();
    int64_t getNextFramePts();
    void createBuffers();
};

}

