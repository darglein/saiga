#pragma once

#include "saiga/config.h"
#include "saiga/opengl/texture/image.h"
#include "saiga/util/timer2.h"
#include "saiga/util/synchronizedBuffer.h"
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <memory>

//ffmpeg is compiled with a pure c compiler, so all includes need an 'extern "C"'.
extern "C"{
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
}

class SAIGA_GLOBAL FFMPEGEncoder{
private:
    int outWidth, outHeight, inWidth, inHeight;



    RingBuffer<std::shared_ptr<Image>> imageStorage;
    SynchronizedBuffer<std::shared_ptr<Image>> imageQueue;

    RingBuffer<AVFrame*> frameStorage;
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
    AVPacket pkt;
    SwsContext * ctx = nullptr;
    Timer2 timer1, timer2, timer3;
    void scaleFrame(std::shared_ptr<Image> image, AVFrame *frame);
    bool encodeFrame(AVFrame *frame, AVPacket &pkt);
    void writeFrame(AVPacket &pkt);
    bool encodeFrame();
    bool scaleFrame();
    void scaleThreadFunc();
    void encodeThreadFunc();
    int64_t getNextFramePts();
public:
//    std::ofstream  outputStream;
	//FILE *f;

    FFMPEGEncoder(int bufferSize = 50);

    //Recommended codecs and container formats:
    //.mp4 AV_CODEC_ID_H264
    //.mpeg AV_CODEC_ID_MPEG2VIDEO or AV_CODEC_ID_MPEG4
    //.avi AV_CODEC_ID_RAWVIDEO
    void startEncoding(const std::string &filename, int outWidth, int outHeight, int inWidth, int inHeight, int outFps, int bitRate,AVCodecID videoCodecId=AV_CODEC_ID_NONE);
    void createBuffers();
    void addFrame(std::shared_ptr<Image> image);
    std::shared_ptr<Image> getFrameBuffer();
    void finishEncoding();
};
