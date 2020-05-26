/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/opengl/world/skybox.h"

using namespace Saiga;



class Sample : public SampleWindowForward
{
    using Base = SampleWindowForward;

   public:
    Sample();

    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;
    Skybox skybox;

   private:
    bool add_values_to_console = true;
    GLPointCloud pointCloud;
    LineSoup lineSoup;
    LineVertexColoredAsset frustum;
};
