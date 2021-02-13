/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#ifndef SAIGA_USE_FFMPEG
#    error Saiga was compiled without FFMPEG
#endif
#include "saiga/config.h"

#include <memory>

namespace Saiga
{
class OpenGLWindow;
class FFMPEGEncoder;

/**
 * A wrapper class for the ffmpeg encoder.
 * This includes a GUI and convenience functions for easy integration into the saiga main loop.
 */
class SAIGA_OPENGL_API VideoEncoder
{
   public:
    std::shared_ptr<FFMPEGEncoder> encoder;
    OpenGLWindow* window;
    VideoEncoder(OpenGLWindow* window);

    void update();
    void renderGUI();

    // Can be used externally, for example, by mapping it to a key.
    void startRecording();
    void stopRecording();

   private:
    int codecId    = 0;
    char file[256] = "out.mp4";
};

}  // namespace Saiga
