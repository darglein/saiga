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
    SimpleAssetObject object;


    vec3 up               = vec3(0, 1, 0);
    bool autoRotate       = false;
    float autoRotateSpeed = 0.5;

    std::array<char, 512> file = {0};


    bool renderObject    = true;
    bool renderWireframe = false;
    bool renderGeometry  = false;
    std::shared_ptr<MVPTextureShader> normalShader, textureShader;

    Sample();

    void update(float dt) override;
    void renderDepth(Camera* cam) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;
};
