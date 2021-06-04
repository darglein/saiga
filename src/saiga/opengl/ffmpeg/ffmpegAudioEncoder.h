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
#include <libavcodec/avcodec.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswscale/swscale.h>

#include <libavutil/channel_layout.h>
}

namespace Saiga
{
class SAIGA_OPENGL_API FFMPEGAudioEncoder
{
   private:
    //    float t, tincr;
    AVCodec* codec;
    AVCodecContext* c = NULL;
    AVFrame* frame;
    AVPacket pkt;
    int i, ret, got_output;
    int buffer_size;
    int bytesPerSample;
    //    uint16_t *samples;
    int currentSamples = 0;
    std::vector<unsigned char> frameBuffer;

   public:
    std::ofstream outputStream;

    FFMPEGAudioEncoder();

    void startEncoding(const std::string& filename);

    void addFrame(std::vector<unsigned char>& soundSamples, int samples);
    void finishEncoding();
    void addFrame();
};

}  // namespace Saiga
