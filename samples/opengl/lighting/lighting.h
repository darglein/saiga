/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/window/SampleWindowDeferred.h"

using namespace Saiga;

class Sample : public SampleWindowDeferred
{
    using Base = SampleWindowDeferred;

   public:
    SimpleAssetObject cube1, cube2;
    SimpleAssetObject sphere;


    std::shared_ptr<BoxLight> boxLight;
    std::shared_ptr<PointLight> pointLight;
    std::shared_ptr<SpotLight> spotLight;

    float rotationSpeed    = 0.1;
    bool showimguidemo     = false;
    bool lightDebug        = false;
    bool pointLightShadows = false;


    Sample();

    void render(Camera* cam) override;
    void renderDepth(Camera* cam) override;
};
