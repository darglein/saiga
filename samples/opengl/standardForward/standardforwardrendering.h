/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/window/SampleWindowStandardForward.h"
#include "saiga/opengl/world/skybox.h"

using namespace Saiga;



class Sample : public SampleWindowStandardForward
{
    using Base = SampleWindowStandardForward;

   public:
    Sample();

    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;
    Skybox skybox;

   private:
    SimpleAssetObject box;
    GLPointCloud pointCloud;
};
