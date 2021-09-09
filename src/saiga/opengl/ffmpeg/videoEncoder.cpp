/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"

#if defined(SAIGA_USE_OPENGL) && defined(SAIGA_USE_FFMPEG)

#    include "saiga/core/imgui/imgui.h"
#    include "saiga/core/util/assert.h"
#    include "saiga/opengl/ffmpeg/ffmpegEncoder.h"
#    include "saiga/opengl/window/OpenGLWindow.h"

#    include "videoEncoder.h"


namespace Saiga
{
VideoEncoder::VideoEncoder(int w, int h)
{
    resize(w, h);
    main_menu.AddItem(
        "Saiga", "VideoEncoder", [this]() { should_render_imgui = !should_render_imgui; }, 298, "F9");
}

VideoEncoder::~VideoEncoder()
{
    main_menu.EraseItem("Saiga", "VideoEncoder");
}

void VideoEncoder::resize(int w, int h)
{
    SAIGA_ASSERT(!encoder || !encoder->isRunning());
    encoder = std::make_shared<FFMPEGEncoder>(file, w, h, w, h, 60);
}

void VideoEncoder::frame(ImageView<ucvec4> image)
{
    if (encoder->isRunning())
    {
        SAIGA_ASSERT(image.w == encoder->inWidth);
        SAIGA_ASSERT(image.h == encoder->inHeight);
//        auto img = encoder->getFrameBuffer();

//        image.copyTo(img->getImageView());
        encoder->addFrame(image);
    }
}


void VideoEncoder::renderGUI()
{
    if (!should_render_imgui) return;

    if (ImGui::Begin("VideoEncoder", &should_render_imgui))
    {
        ImGui::InputText("Output File", &file);
        encoder->filename = file;


        ImGui::InputInt("Output Width", &encoder->outWidth);
        ImGui::InputInt("Output Height", &encoder->outHeight);
        ImGui::InputInt("Output FPS", &encoder->outFps);
        ImGui::InputInt("Output Bitrate", &encoder->bitRate);


        static const char* codecitems[4] = {"AV_CODEC_ID_H264", "AV_CODEC_ID_MPEG2VIDEO", "AV_CODEC_ID_MPEG4",
                                            "AV_CODEC_ID_RAWVIDEO"};
        ImGui::Combo("Codec", &codecId, codecitems, 4);

        AVCodecID codec = AV_CODEC_ID_H264;
        switch (codecId)
        {
            case 0:
                codec = AV_CODEC_ID_H264;
                break;
            case 1:
                codec = AV_CODEC_ID_MPEG2VIDEO;
                break;
            case 2:
                codec = AV_CODEC_ID_MPEG4;
                break;
            case 3:
                codec = AV_CODEC_ID_RAWVIDEO;
                break;
        }
        encoder->videoCodecId = codec;



        if (!encoder->isRunning() && ImGui::Button("Start Recording"))
        {
            startRecording();
        }
        if (encoder->isRunning() && ImGui::Button("Stop Recording"))
        {
            stopRecording();
        }
    }
    ImGui::End();
}

void VideoEncoder::startRecording()
{
    encoder->startEncoding();
}

void VideoEncoder::stopRecording()
{
    encoder->finishEncoding();
}

bool VideoEncoder::isEncoding()
{
    return encoder->isRunning();
}

ivec2 VideoEncoder::Size()
{
    return ivec2(encoder->inWidth, encoder->inHeight);
}

}  // namespace Saiga

#endif
