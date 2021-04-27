/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/core/util/Align.h"
#include "saiga/opengl/rendering/lighting/attenuated_light.h"
#include "saiga/opengl/shader/basic_shaders.h"

namespace Saiga
{
class SAIGA_OPENGL_API LightShader : public DeferredShader
{
   public:
    GLint location_lightColorDiffuse, location_lightColorSpecular;  // rgba, rgb=color, a=intensity [0,1]
    GLint location_depthBiasMV, location_depthTex, location_readShadowMap;
    GLint location_shadowMapSize;  // vec4(w,h,1/w,1/h)
    GLint location_invProj;        // required to compute the viewspace position from the gbuffer
    GLint location_volumetricDensity;

    virtual void checkUniforms();

    void uploadVolumetricDensity(float density);

    void uploadColorDiffuse(vec3& color, float intensity);
    void uploadColorSpecular(vec3& color, float intensity);

    void uploadDepthBiasMV(const mat4& mat);
    void uploadDepthTexture(std::shared_ptr<TextureBase> texture);
    void uploadShadow(float shadow);
    void uploadShadowMapSize(ivec2 s);
    void uploadInvProj(const mat4& mat);
};



class SAIGA_OPENGL_API AttenuatedLightShader : public LightShader
{
   public:
    GLint location_attenuation;

    virtual void checkUniforms();
    virtual void uploadA(vec3& attenuation, float cutoffRadius);
};

class PointLight;
class SpotLight;
class DirectionalLight;

class SAIGA_OPENGL_API PointLightShader : public AttenuatedLightShader
{
   public:
    GLint location_shadowPlanes;

    virtual void checkUniforms();

    void uploadShadowPlanes(float f, float n);

    void SetUniforms(PointLight* light, Camera* shadow_camera);
};


class SAIGA_OPENGL_API SpotLightShader : public AttenuatedLightShader
{
   public:
    GLint location_angle;
    GLint location_shadowPlanes;
    virtual void checkUniforms();
    void uploadAngle(float angle);
    void uploadShadowPlanes(float f, float n);

    void SetUniforms(SpotLight* light, Camera* shadow_camera);
};


class SAIGA_OPENGL_API DirectionalLightShader : public LightShader
{
   public:
    GLint location_direction, location_ambientIntensity;
    GLint location_ssaoTexture;
    GLint location_depthTexures;
    GLint location_viewToLightTransforms;
    GLint location_depthCuts;
    GLint location_numCascades;
    GLint location_cascadeInterpolateRange;

    virtual void checkUniforms();
    void uploadDirection(vec3& direction);
    void uploadAmbientIntensity(float i);
    void uploadSsaoTexture(std::shared_ptr<TextureBase> texture);

    void uploadDepthTextures(std::vector<std::shared_ptr<TextureBase>>& textures);
    void uploadViewToLightTransforms(AlignedVector<mat4>& transforms);
    void uploadDepthCuts(std::vector<float>& depthCuts);
    void uploadNumCascades(int n);
    void uploadCascadeInterpolateRange(float r);

    void SetUniforms(DirectionalLight* light, Camera* shadow_camera);
};


}  // namespace Saiga
