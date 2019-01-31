/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"

#if defined(SAIGA_USE_OPENGL) && defined(SAIGA_USE_FFMPEG)

#    include "saiga/imgui/imgui.h"
#    include "saiga/opengl/window/OpenGLWindow.h"
#    include "saiga/util/assert.h"

#    include "videoEncoder.h"


namespace Saiga
{
VideoEncoder::VideoEncoder(OpenGLWindow* window)
    : encoder(file, window->getWidth(), window->getHeight(), window->getWidth(), window->getHeight(), 60),
      window(window)
{
}

void VideoEncoder::update()
{
    if (encoder.isRunning())
    {
        auto img = encoder.getFrameBuffer();
        // read the current framebuffer to the buffer
        window->readToExistingImage(*img);
        // add an image to the video stream
        encoder.addFrame(img);
    }
}

void VideoEncoder::renderGUI()
{
    {
        ImGui::PushID(346436);

        ImGui::InputText("Output File", file, 256);
        encoder.filename = file;


        ImGui::InputInt("Output Width", &encoder.outWidth);
        ImGui::InputInt("Output Height", &encoder.outHeight);
        ImGui::InputInt("Output FPS", &encoder.outFps);
        ImGui::InputInt("Output Bitrate", &encoder.bitRate);


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

        encoder.videoCodecId = codec;



        if (!encoder.isRunning() && ImGui::Button("Start Recording"))
        {
            encoder.inWidth  = window->getWidth();
            encoder.inHeight = window->getHeight();
            encoder.startEncoding();
        }
        if (encoder.isRunning() && ImGui::Button("Stop Recording"))
        {
            encoder.finishEncoding();
        }

        ImGui::PopID();
    }
}

}  // namespace Saiga

#endif
