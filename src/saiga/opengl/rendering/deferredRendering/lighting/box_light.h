/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/rendering/deferredRendering/lighting/directional_light.h"

namespace Saiga
{
class SAIGA_OPENGL_API BoxLightShader : public LightShader
{
   public:
    virtual void checkUniforms();
};

class SAIGA_OPENGL_API BoxLight : public Light
{
    friend class DeferredLighting;

   protected:
    std::shared_ptr<SimpleShadowmap> shadowmap;

   public:
    OrthographicCamera shadowCamera;
    BoxLight();
    ~BoxLight() {}

    void bindUniforms(std::shared_ptr<BoxLightShader> shader, Camera* cam);

    void setView(vec3 pos, vec3 target, vec3 up);

    void createShadowMap(int w, int h, ShadowQuality quality = ShadowQuality::LOW);

    void calculateCamera();

    bool cullLight(Camera* cam);
    bool renderShadowmap(DepthFunction f, UniformBuffer& shadowCameraBuffer);
    void renderImGui();
};

}  // namespace Saiga
