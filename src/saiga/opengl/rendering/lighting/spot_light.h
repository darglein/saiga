/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/rendering/lighting/deferred_light_shader.h"
#include "saiga/opengl/rendering/lighting/point_light.h"

namespace Saiga
{
class SAIGA_OPENGL_API SpotLight : public LightBase, public LightDistanceAttenuation
{
    friend class DeferredLighting;

   protected:
    float angle = 60.0f;
    std::unique_ptr<SimpleShadowmap> shadowmap;

   public:
    float shadowNearPlane = 0.1f;
    PerspectiveCamera shadowCamera;
    vec3 direction = vec3(0, -1, 0);
    vec3 position  = vec3(0, 0, 0);
    vec3 getPosition() { return position; }
    void setPosition(const vec3& p) { position = p; }

    /**
     * The default direction of the mesh is negative y
     */

    SpotLight();
    virtual ~SpotLight() {}
    void bindUniforms(std::shared_ptr<SpotLightShader> shader, Camera* shadowCamera);



    void createShadowMap(int w, int h, ShadowQuality quality = ShadowQuality::LOW);


    mat4 ModelMatrix();

    void setAngle(float value);
    float getAngle() const { return angle; }

    void setDirection(vec3 dir);

    void calculateCamera();

    bool cullLight(Camera* cam);
    bool renderShadowmap(DepthFunction f, UniformBuffer& shadowCameraBuffer);
    void renderImGui();
};

}  // namespace Saiga
