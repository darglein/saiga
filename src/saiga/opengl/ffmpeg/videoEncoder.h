/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#ifndef SAIGA_USE_FFMPEG
#    error Saiga was compiled without FFMPEG
#endif
#include "saiga/config.h"
#include "saiga/core/image/all.h"

#include <memory>

namespace Saiga
{
class FFMPEGEncoder;

/**
 * A wrapper class for the ffmpeg encoder.
 * This includes a GUI and convenience functions for easy integration into the saiga main loop.
 */
class SAIGA_OPENGL_API VideoEncoder
{
   public:
    std::shared_ptr<FFMPEGEncoder> encoder;
    VideoEncoder(int w, int h);
    ~VideoEncoder();

    void resize(int w, int h);

    void frame(ImageView<ucvec4> image);


    void renderGUI();

    // Can be used externally, for example, by mapping it to a key.
    void startRecording();
    void stopRecording();
    bool isEncoding();


    ivec2 Size();

   private:
    int codecId              = 0;
    std::string file         = "out.mp4";
    bool should_render_imgui = false;
};

}  // namespace Saiga
