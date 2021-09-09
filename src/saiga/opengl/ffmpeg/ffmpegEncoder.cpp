/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/config.h"

#if defined(SAIGA_USE_FFMPEG)
#    include "saiga/core/util/assert.h"

#    include "ffmpegEncoder.h"

extern "C"
{
#    include "libavformat/avformat.h"
#    include "libswscale/swscale.h"

#    include <libavcodec/avcodec.h>
#    include <libavutil/common.h>
#    include <libavutil/imgutils.h>
#    include <libavutil/mathematics.h>
#    include <libavutil/opt.h>
#    include <libavutil/samplefmt.h>

#    include <libavutil/channel_layout.h>
}



namespace Saiga
{
static bool ffmpegInitialized = false;

FFMPEGEncoder::FFMPEGEncoder(const std::string& filename, int outWidth, int outHeight, int inWidth, int inHeight,
                             int outFps, int bitRate, AVCodecID videoCodecId, int bufferSize)
    : filename(filename),
      outWidth(outWidth),
      outHeight(outHeight),
      inWidth(inWidth),
      inHeight(inHeight),
      outFps(outFps),
      bitRate(bitRate),
      videoCodecId(videoCodecId),
      imageQueue(bufferSize),
      frameStorage(bufferSize),
      frameQueue(bufferSize)
{
    if (!ffmpegInitialized)
    {
        std::cout << "Initializing FFMPEG... ";
        av_log_set_level(AV_LOG_DEBUG);



        // These functions are deprecated since version 58, but required before that.
#    if LIBAVCODEC_VERSION_MAJOR < 58
        avcodec_register_all();
#    endif
#    if LIBAVFORMAT_VERSION_MAJOR < 58
        av_register_all();
#    endif
        ffmpegInitialized = true;
        std::cout << "done" << std::endl;
    }
}

FFMPEGEncoder::~FFMPEGEncoder()
{
    finishEncoding();
}

void FFMPEGEncoder::scaleThreadFunc()
{
    while (true)
    {
        bool hadWork = scaleFrame();

        if (!running && !hadWork) break;


        if (!hadWork)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    std::cout << "Scale Thread done." << std::endl;
    finishScale = true;
}

bool FFMPEGEncoder::scaleFrame()
{
    std::shared_ptr<TemplatedImage<ucvec4>> image;
    if (!imageQueue.tryGet(image))
    {
        return false;
    }
    AVFrame* frame = frameStorage.get();
    scaleFrame(image, frame);
    frameQueue.add(frame);
    return true;
}

void FFMPEGEncoder::scaleFrame(std::shared_ptr<TemplatedImage<ucvec4>> image, AVFrame* frame)
{
    uint8_t* inData[1] = {image->data8()};          // RGB24 have one plane
    int inLinesize[1]  = {(int)image->pitchBytes};  // RGB stride

    SAIGA_ASSERT(image);
    SAIGA_ASSERT(frame);

    sws_scale(ctx, inData, inLinesize, 0, image->height, frame->data, frame->linesize);
    frame->pts = getNextFramePts();
}

int64_t FFMPEGEncoder::getNextFramePts()
{
    return currentFrame++ * ticksPerFrame;
}

void FFMPEGEncoder::encodeThreadFunc()
{
    while (!finishScale)
    {
        bool hadWork = encodeFrame();
        if (!hadWork)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    while (encodeFrame())
    {
    }
    std::cout << "Encode Thread done." << std::endl;
    finishEncode = true;
}

bool FFMPEGEncoder::encodeFrame()
{
    AVFrame* frame;
    if (!frameQueue.tryGet(frame))
    {
        return false;
    }

    encodeFrame(frame);

    frameStorage.add(frame);
    finishedFrames++;
    return true;
}

bool FFMPEGEncoder::encodeFrame(AVFrame* frame)
{
    int ret = avcodec_send_frame(m_codecContext, frame);
    if (ret < 0)
    {
        std::cout << "Error encoding frame" << std::endl;
        exit(1);
    }

    AVPacket* pkt = av_packet_alloc();
    while (ret >= 0)
    {
        ret = avcodec_receive_packet(m_codecContext, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return true;
        else if (ret < 0)
        {
            fprintf(stderr, "Error during encoding\n");
            exit(1);
        }
        av_interleaved_write_frame(m_formatCtx, pkt);
        av_packet_unref(pkt);
    }

    av_packet_free(&pkt);

    return true;
}


void FFMPEGEncoder::addFrame(ImageView<ucvec4> view)
{
    SAIGA_ASSERT(view.w == inWidth);
    SAIGA_ASSERT(view.h == inHeight);

    auto image = std::make_shared<TemplatedImage<ucvec4>>(view);


    imageQueue.add(image);
}


void FFMPEGEncoder::finishEncoding()
{
    if (!running) return;

    std::cout << "finishing ffmpeg encoding..." << std::endl;
    running = false;

    scaleThread.join();
    encodeThread.join();

    // Flush the encoder.
    // This will encode all remaining frames and write them to the output.
    encodeFrame(nullptr);

    av_write_trailer(m_formatCtx);

    avcodec_close(m_codecContext);
    sws_freeContext(ctx);
    ctx = nullptr;

    avcodec_free_context(&m_codecContext);

    frameStorage.clear();

    std::cout << "encoding done." << std::endl;
}

void FFMPEGEncoder::startEncoding()
{
    std::cout << "FFMPEGEncoder start encoding to file " << filename << std::endl;

    //    this->outWidth = outWidth;
    //    this->outHeight = outHeight;
    //    this->inWidth = inWidth;
    //    this->inHeight = inHeight;
    int timeBase = outFps * 1000;

    AVOutputFormat* oformat = av_guess_format(NULL, filename.c_str(), NULL);
    if (oformat == NULL)
    {
        oformat = av_guess_format("mpeg", NULL, NULL);
    }

    SAIGA_ASSERT(oformat, "av_guess_format failed!");

    if (videoCodecId == AV_CODEC_ID_NONE)
    {
        // use the default codec given by the format
        videoCodecId = oformat->video_codec;
    }
    else
    {
        oformat->video_codec = videoCodecId;
    }

    AVCodec* codec = avcodec_find_encoder(oformat->video_codec);
    if (codec == NULL)
    {
        std::cerr << "Could not find encoder. " << std::endl;
        exit(1);
    }

    m_codecContext = avcodec_alloc_context3(codec);
    if (m_codecContext == NULL)
    {
        std::cerr << "Could allocate codec context. " << std::endl;
        exit(1);
    }
    m_codecContext->codec_id     = oformat->video_codec;
    m_codecContext->codec_type   = AVMEDIA_TYPE_VIDEO;
    m_codecContext->gop_size     = 10;
    m_codecContext->bit_rate     = bitRate;
    m_codecContext->width        = outWidth;
    m_codecContext->height       = outHeight;
    m_codecContext->max_b_frames = 1;
    m_codecContext->pix_fmt      = AV_PIX_FMT_YUV420P;
    m_codecContext->framerate    = {1, outFps};
    m_codecContext->time_base    = {1, outFps};

    m_formatCtx                 = avformat_alloc_context();
    m_formatCtx->oformat        = oformat;
    m_formatCtx->video_codec_id = oformat->video_codec;


    AVCodecParameters* codecpar = avcodec_parameters_alloc();
    avcodec_parameters_from_context(codecpar, m_codecContext);

    AVStream* videoStream = avformat_new_stream(m_formatCtx, codec);
    if (!videoStream)
    {
        printf("Could not allocate stream\n");
    }
    //    videoStream->codecpar
    videoStream->codec = m_codecContext;
    //    videoStream->codecpar= codecpar;
    videoStream->time_base = {1, timeBase};
    if (m_formatCtx->oformat->flags & AVFMT_GLOBALHEADER)
    {
#    if LIBAVUTIL_VERSION_MAJOR >= 56
        m_codecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
#    else
        m_codecContext->flags |= CODEC_FLAG_GLOBAL_HEADER;
#    endif
    }
    //    1 = 1;
    if (avcodec_open2(m_codecContext, codec, NULL) < 0)
    {
        std::cerr << "Failed to open codec. " << std::endl;
        exit(1);
    }

    if (avio_open(&m_formatCtx->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0)
    {
        std::cerr << "Failed to open output file. " << std::endl;
        exit(1);
    }

    if (avformat_write_header(m_formatCtx, NULL) < 0)
    {
        std::cout << "avformat_write_header error" << std::endl;
        exit(1);
    }


    av_dump_format(m_formatCtx, 0, filename.c_str(), 1);

    if (videoStream->time_base.den != timeBase)
    {
        std::cerr << "Warning: Stream time base different to desired time base. " << videoStream->time_base.den
                  << " instead of " << timeBase << std::endl;
        timeBase = videoStream->time_base.den;
    }
    // SAIGA_ASSERT(videoStream->time_base.num == 1);
    ticksPerFrame = videoStream->time_base.den / outFps;



    SAIGA_ASSERT(ctx == nullptr);
    ctx = sws_getContext(inWidth, inHeight, AV_PIX_FMT_RGBA, m_codecContext->width, m_codecContext->height,
                         AV_PIX_FMT_YUV420P, 0, 0, 0, 0);
    SAIGA_ASSERT(ctx);

    running        = true;
    finishScale    = false;
    finishEncode   = false;
    currentFrame   = 0;
    finishedFrames = 0;
    createBuffers();

    scaleThread  = std::thread(&FFMPEGEncoder::scaleThreadFunc, this);
    encodeThread = std::thread(&FFMPEGEncoder::encodeThreadFunc, this);
}

void FFMPEGEncoder::createBuffers()
{
    for (int i = 0; i < (int)frameStorage.capacity(); ++i)
    {
        AVFrame* frame = av_frame_alloc();
        if (!frame)
        {
            fprintf(stderr, "Could not allocate video frame\n");
            exit(1);
        }
        frame->format = m_codecContext->pix_fmt;
        frame->width  = m_codecContext->width;
        frame->height = m_codecContext->height;

        /* the image can be allocated by any means and av_image_alloc() is
         * just the most convenient way if av_malloc() is to be used */
        int ret = av_image_alloc(frame->data, frame->linesize, m_codecContext->width, m_codecContext->height,
                                 m_codecContext->pix_fmt, 32);
        if (ret < 0)
        {
            fprintf(stderr, "Could not allocate raw picture buffer\n");
            exit(1);
        }
        frameStorage.add(frame);
    }
}

}  // namespace Saiga
#endif
