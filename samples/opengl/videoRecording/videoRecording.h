/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/animation/cameraAnimation.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/ffmpeg/ffmpegEncoder.h"
#include "saiga/opengl/ffmpeg/videoEncoder.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"

using namespace Saiga;


class Sample : public SampleWindowDeferred
{
    using Base = SampleWindowDeferred;

   public:
    SimpleAssetObject cube1, cube2;
    SimpleAssetObject sphere;

    int remainingFrames;
    bool rotateCamera = false;
    int frame         = 0;
    int frameSkip     = 0;
    std::shared_ptr<FFMPEGEncoder> encoder;

    VideoEncoder enc;



    Interpolation cameraInterpolation;

    Sample();
    void testBspline();

    void update(float dt) override;
    void render(Camera* cam) override;
    void renderDepth(Camera* cam) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;
};
