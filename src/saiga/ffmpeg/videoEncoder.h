/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/ffmpeg/ffmpegEncoder.h"



namespace Saiga
{
class OpenGLWindow;

/**
 * A wrapper class for the ffmpeg encoder.
 * This includes a GUI and convenience functions for easy integration into the saiga main loop.
 */
class SAIGA_GLOBAL VideoEncoder
{
   public:
    FFMPEGEncoder encoder;
    OpenGLWindow* window;
    VideoEncoder(OpenGLWindow* window);

    void update();
    void renderGUI();

   private:
    int codecId    = 0;
    char file[256] = "out.mp4";
};

}  // namespace Saiga
