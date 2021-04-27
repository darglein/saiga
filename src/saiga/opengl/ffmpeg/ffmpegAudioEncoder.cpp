/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/config.h"

#if defined(SAIGA_USE_FFMPEG)

#    include "saiga/core/math/math.h"
#    include "saiga/core/util/assert.h"

#    include "ffmpegAudioEncoder.h"

namespace Saiga
{
FFMPEGAudioEncoder::FFMPEGAudioEncoder()
{
    // Can be removed after ffmpeg 4
    // avcodec_register_all();
}

void FFMPEGAudioEncoder::addFrame()
{
    //    std::cout << "Adding audio frame." << std::endl;
    SAIGA_ASSERT(currentSamples == buffer_size);

    pkt.data = NULL;  // packet data will be allocated by the encoder
    pkt.size = 0;
    /* encode the samples */
    //    ret = avcodec_encode_audio2(c, &pkt, frame, &got_output);

    auto ret = avcodec_send_frame(c, frame);
    avcodec_receive_packet(c, &pkt);
    SAIGA_ASSERT(0, "check");

    if (ret < 0)
    {
        fprintf(stderr, "Error encoding audio frame\n");
        exit(1);
    }
    if (got_output)
    {
        outputStream.write((const char*)pkt.data, pkt.size);
        av_packet_unref(&pkt);
    }
    currentSamples = 0;
}


void FFMPEGAudioEncoder::addFrame(std::vector<unsigned char>& soundSamples, int samples)
{
    int newSamples      = samples * bytesPerSample;
    int newSampleOffset = 0;
    int requiredSamples = frameBuffer.size() - currentSamples;

    while (newSamples > requiredSamples)
    {
        // copy samples to buffer
        std::copy(soundSamples.begin() + newSampleOffset, soundSamples.begin() + newSampleOffset + requiredSamples,
                  frameBuffer.begin() + currentSamples);
        currentSamples += requiredSamples;

        // buffer is full now
        addFrame();
        newSampleOffset += requiredSamples;
        newSamples -= requiredSamples;
        requiredSamples = frameBuffer.size();
    }

    if (newSamples <= requiredSamples)
    {
        // copy samples to buffer
        std::copy(soundSamples.begin() + newSampleOffset, soundSamples.begin() + newSampleOffset + newSamples,
                  frameBuffer.begin() + currentSamples);
        currentSamples += newSamples;

        if (currentSamples == buffer_size)
        {
            // buffer is full write frame!
            addFrame();
        }
        return;
    }
}



void FFMPEGAudioEncoder::finishEncoding()
{
    /* get the delayed frames */
    for (got_output = 1; got_output; i++)
    {
        // ret = avcodec_encode_audio2(c, &pkt, NULL, &got_output);
        SAIGA_ASSERT(0, "check");
        if (ret < 0)
        {
            fprintf(stderr, "Error encoding frame\n");
            exit(1);
        }
        if (got_output)
        {
            outputStream.write((const char*)pkt.data, pkt.size);
            av_packet_unref(&pkt);
        }
    }
    outputStream.close();
    //    av_freep(&samples);
    av_frame_free(&frame);
    avcodec_close(c);
    av_free(c);
}

/* check that a given sample format is supported by the encoder */
static int check_sample_fmt(AVCodec* codec, enum AVSampleFormat sample_fmt)
{
    const enum AVSampleFormat* p = codec->sample_fmts;
    while (*p != AV_SAMPLE_FMT_NONE)
    {
        if (*p == sample_fmt) return 1;
        p++;
    }
    return 0;
}

#    if 0
/* just pick the highest supported samplerate */
static int select_sample_rate(AVCodec *codec)
{
    const int *p;
    int best_samplerate = 0;
    if (!codec->supported_samplerates)
        return 44100;
    p = codec->supported_samplerates;
    while (*p) {
        best_samplerate = FFMAX(*p, best_samplerate);
        p++;
    }
    return best_samplerate;
}


/* select layout with the highest channel count */
static int select_channel_layout(AVCodec *codec)
{
    const uint64_t *p;
    uint64_t best_ch_layout = 0;
    int best_nb_channels   = 0;
    if (!codec->channel_layouts)
        return AV_CH_LAYOUT_STEREO;
    p = codec->channel_layouts;
    while (*p) {
        int nb_channels = av_get_channel_layout_nb_channels(*p);
        if (nb_channels > best_nb_channels) {
            best_ch_layout    = *p;
            best_nb_channels = nb_channels;
        }
        p++;
    }
    return best_ch_layout;
}
#    endif

void FFMPEGAudioEncoder::startEncoding(const std::string& filename)
{
    std::cout << "Encode audio file " << filename << std::endl;
    /* find the MP2 encoder */
    codec = avcodec_find_encoder(AV_CODEC_ID_MP2);
    if (!codec)
    {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }
    c = avcodec_alloc_context3(codec);
    if (!c)
    {
        fprintf(stderr, "Could not allocate audio codec context\n");
        exit(1);
    }
    /* put sample parameters */
    c->bit_rate = 64000;
    /* check that the encoder supports s16 pcm input */
    c->sample_fmt = AV_SAMPLE_FMT_S16;
    if (!check_sample_fmt(codec, c->sample_fmt))
    {
        fprintf(stderr, "Encoder does not support sample format %s", av_get_sample_fmt_name(c->sample_fmt));
        exit(1);
    }
    /* select other audio parameters supported by the encoder */
    //    c->sample_rate    = select_sample_rate(codec);
    //    c->channel_layout = select_channel_layout(codec);
    //    c->channels       = av_get_channel_layout_nb_channels(c->channel_layout);
    c->sample_rate    = 44100;
    c->channel_layout = AV_CH_LAYOUT_STEREO;
    c->channels       = 2;

    bytesPerSample = 2 * 2;  // 16 bit stereo

    /* open it */
    if (avcodec_open2(c, codec, NULL) < 0)
    {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

    std::cout << "Audio encoding. "
            "c->sample_rate="
         << c->sample_rate << " c->channel_layout=" << c->channel_layout << " c->channels=" << c->channels
         << " c->frame_size=" << c->frame_size << std::endl;

    outputStream.open(filename, std::ios::out | std::ios::binary);

    /* frame containing input raw audio */
    frame = av_frame_alloc();
    if (!frame)
    {
        fprintf(stderr, "Could not allocate audio frame\n");
        exit(1);
    }
    frame->nb_samples     = c->frame_size;
    frame->format         = c->sample_fmt;
    frame->channel_layout = c->channel_layout;



    /* the codec gives us the frame size, in samples,
     * we calculate the size of the samples buffer in bytes */
    buffer_size = av_samples_get_buffer_size(NULL, c->channels, c->frame_size, c->sample_fmt, 0);
    SAIGA_ASSERT(buffer_size == bytesPerSample * c->frame_size);
    std::cout << "buffer size: " << buffer_size << " test " << 2 * 2 * c->frame_size << std::endl;
    if (buffer_size < 0)
    {
        fprintf(stderr, "Could not get sample buffer size\n");
        exit(1);
    }
    frameBuffer.resize(buffer_size);
    /* setup the data pointers in the AVFrame */
    ret =
        avcodec_fill_audio_frame(frame, c->channels, c->sample_fmt, (const uint8_t*)frameBuffer.data(), buffer_size, 0);
    if (ret < 0)
    {
        fprintf(stderr, "Could not setup audio frame\n");
        exit(1);
    }

    av_init_packet(&pkt);
}

}  // namespace Saiga
#endif
