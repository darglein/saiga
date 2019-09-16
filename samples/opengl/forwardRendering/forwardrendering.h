/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/opengl/window/SampleWindowForward.h"

using namespace Saiga;



class Sample : public SampleWindowForward
{
    using Base = SampleWindowForward;

   public:
    Sample();

    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;

   private:
    GLPointCloud pointCloud;
    LineSoup lineSoup;
    LineVertexColoredAsset frustum;
};
