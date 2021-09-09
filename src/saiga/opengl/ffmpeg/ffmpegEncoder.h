/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/image/image.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Thread/SynchronizedBuffer.h"

#include <fstream>
#include <thread>


// ffmpeg is compiled with a pure c compiler, so all includes need an 'extern "C"'.
extern "C"
{
    struct AVFrame;
    struct AVCodecContext;
    struct AVFormatContext;
    struct SwsContext;
#include <libavcodec/avcodec.h>
}

namespace Saiga
{
class SAIGA_OPENGL_API FFMPEGEncoder
{
   public:

    // Recommended codecs and container formats:
    //  .mp4    AV_CODEC_ID_H264
    //  .mpeg   AV_CODEC_ID_MPEG2VIDEO or AV_CODEC_ID_MPEG4
    //  .avi    AV_CODEC_ID_RAWVIDEO
    FFMPEGEncoder(const std::string& filename, int outWidth, int outHeight, int inWidth, int inHeight, int outFps = 60,
                  int bitRate = 4000000, AVCodecID videoCodecId = AV_CODEC_ID_NONE, int bufferSize = 50);
    ~FFMPEGEncoder();


    void addFrame(ImageView<ucvec4> image);


    bool isRunning() { return running; }
    void startEncoding();
    void finishEncoding();


    std::string filename;
    int outWidth, outHeight, inWidth, inHeight;
    int outFps;
    int bitRate;
    AVCodecID videoCodecId;

   private:
    SynchronizedBuffer<std::shared_ptr<TemplatedImage<ucvec4>>> imageQueue;

    SynchronizedBuffer<AVFrame*> frameStorage;
    SynchronizedBuffer<AVFrame*> frameQueue;

    std::thread scaleThread;   // scales and converts the image to the correct size and color format
    std::thread encodeThread;  // does the actual encoding

    volatile int currentFrame   = 0;
    volatile int finishedFrames = 0;

    volatile bool running = false;
    volatile bool finishScale;
    volatile bool finishEncode;
    //    AVCodec *codec;
    //    AVCodecContext *c= NULL;

    AVCodecContext* m_codecContext;
    AVFormatContext* m_formatCtx;
    int ticksPerFrame;
    // AVPacket pkt;
    SwsContext* ctx = nullptr;
    void scaleFrame(std::shared_ptr<TemplatedImage<ucvec4>> image, AVFrame* frame);
    bool encodeFrame(AVFrame* frame);
    bool encodeFrame();
    bool scaleFrame();
    void scaleThreadFunc();
    void encodeThreadFunc();
    int64_t getNextFramePts();
    void createBuffers();
};

}  // namespace Saiga
