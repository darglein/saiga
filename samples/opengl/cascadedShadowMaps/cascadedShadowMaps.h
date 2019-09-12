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
    SimpleAssetObject groundPlane;
    std::vector<SimpleAssetObject> cubes;

    AABB sceneBB;
    ProceduralSkybox skybox;


    bool debugLightShader    = false;
    bool fitShadowToCamera   = true;
    bool fitNearPlaneToScene = true;

    std::shared_ptr<DirectionalLight> sun;

    Sample();
    ~Sample();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void render(Camera* cam) override;
    void renderDepth(Camera* cam) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;
};
