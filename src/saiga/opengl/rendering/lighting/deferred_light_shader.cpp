/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "deferred_light_shader.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/lighting/all.h"
#include "saiga/opengl/rendering/lighting/deferred_lighting.h"
namespace Saiga
{
void LightShader::checkUniforms()
{
    DeferredShader::checkUniforms();
    location_lightColorDiffuse  = getUniformLocation("lightColorDiffuse");
    location_lightColorSpecular = getUniformLocation("lightColorSpecular");
    location_depthBiasMV        = getUniformLocation("depthBiasMV");
    location_depthTex           = getUniformLocation("depthTex");
    location_readShadowMap      = getUniformLocation("readShadowMap");
    location_shadowMapSize      = getUniformLocation("shadowMapSize");
    location_invProj            = getUniformLocation("invProj");
    location_volumetricDensity  = getUniformLocation("volumetricDensity");
}

void LightShader::uploadVolumetricDensity(float density)
{
    Shader::upload(location_volumetricDensity, density);
}



void LightShader::uploadColorDiffuse(vec3& color, float intensity)
{
    vec4 c = make_vec4(color, intensity);
    Shader::upload(location_lightColorDiffuse, c);
}


void LightShader::uploadColorSpecular(vec3& color, float intensity)
{
    vec4 c = make_vec4(color, intensity);
    Shader::upload(location_lightColorSpecular, c);
}

void LightShader::uploadDepthBiasMV(const mat4& mat)
{
    Shader::upload(location_depthBiasMV, mat);
}

void LightShader::uploadInvProj(const mat4& mat)
{
    Shader::upload(location_invProj, mat);
}

void LightShader::uploadDepthTexture(std::shared_ptr<TextureBase> texture)
{
    texture->bind(5);
    Shader::upload(location_depthTex, 5);
}

void LightShader::uploadShadow(float shadow)
{
    Shader::upload(location_readShadowMap, shadow);
}

void LightShader::uploadShadowMapSize(ivec2 s)
{
    auto w = s[0];
    auto h = s[1];
    Shader::upload(location_shadowMapSize, vec4(w, h, 1.0f / w, 1.0f / h));
}


void AttenuatedLightShader::checkUniforms()
{
    LightShader::checkUniforms();
    location_attenuation = getUniformLocation("attenuation");
}


void AttenuatedLightShader::uploadA(vec3& attenuation, float cutoffRadius)
{
    Shader::upload(location_attenuation, make_vec4(attenuation, cutoffRadius));
}


void PointLightShader::checkUniforms()
{
    AttenuatedLightShader::checkUniforms();
    location_shadowPlanes = getUniformLocation("shadowPlanes");
}



void PointLightShader::uploadShadowPlanes(float f, float n)
{
    Shader::upload(location_shadowPlanes, vec2(f, n));
}

void PointLightShader::SetUniforms(PointLight* light, Camera* shadow_camera)
{
    if (light->volumetric) this->uploadVolumetricDensity(light->volumetricDensity);
    this->upload(10, light->active_light_id);
    this->uploadModel(light->ModelMatrix());
    this->uploadInvProj(inverse(shadow_camera->proj));
    if (light->castShadows)
    {
        this->uploadShadowPlanes(light->shadowCamera.zFar, light->shadowCamera.zNear);
        this->upload(8, light->shadow_id);
    }
}

void SpotLightShader::checkUniforms()
{
    AttenuatedLightShader::checkUniforms();
    location_angle        = getUniformLocation("angle");
    location_shadowPlanes = getUniformLocation("shadowPlanes");
}


void SpotLightShader::uploadAngle(float angle)
{
    Shader::upload(location_angle, angle);
}

void SpotLightShader::uploadShadowPlanes(float f, float n)
{
    Shader::upload(location_shadowPlanes, vec2(f, n));
}

void SpotLightShader::SetUniforms(SpotLight* light, Camera* shadow_camera)
{
    if (light->volumetric) this->uploadVolumetricDensity(light->volumetricDensity);
    this->upload(10, light->active_light_id);
    this->uploadModel(light->ModelMatrix());
    this->uploadShadowPlanes(light->shadowCamera.zFar, light->shadowCamera.zNear);
    this->uploadInvProj(inverse(shadow_camera->proj));
    if (light->castShadows)
    {
        uploadShadowMapSize(ivec2(512, 512));
        this->upload(8, light->shadow_id);
    }
}

void DirectionalLightShader::checkUniforms()
{
    LightShader::checkUniforms();
    location_direction               = getUniformLocation("direction");
    location_ambientIntensity        = getUniformLocation("ambientIntensity");
    location_ssaoTexture             = getUniformLocation("ssaoTex");
    location_depthTexures            = getUniformLocation("depthTexures");
    location_viewToLightTransforms   = getUniformLocation("viewToLightTransforms");
    location_depthCuts               = getUniformLocation("depthCuts");
    location_numCascades             = getUniformLocation("numCascades");
    location_cascadeInterpolateRange = getUniformLocation("cascadeInterpolateRange");
}



void DirectionalLightShader::uploadDirection(vec3& direction)
{
    Shader::upload(location_direction, direction);
}

void DirectionalLightShader::uploadAmbientIntensity(float i)
{
    Shader::upload(location_ambientIntensity, i);
}


void DirectionalLightShader::uploadNumCascades(int n)
{
    Shader::upload(location_numCascades, n);
}

void DirectionalLightShader::uploadCascadeInterpolateRange(float r)
{
    Shader::upload(location_cascadeInterpolateRange, r);
}

void DirectionalLightShader::uploadSsaoTexture(std::shared_ptr<TextureBase> texture)
{
    texture->bind(5);
    Shader::upload(location_ssaoTexture, 5);
}


void DirectionalLightShader::SetUniforms(DirectionalLight* light, Camera* shadow_camera)
{
    this->uploadAmbientIntensity(light->ambientIntensity);

    vec3 viewd = -normalize(make_vec3(shadow_camera->view * make_vec4(light->direction, 0)));
    this->uploadDirection(viewd);

    mat4 ip = inverse(shadow_camera->proj);
    this->uploadInvProj(ip);
    this->upload(10, light->active_light_id);

    if (light->castShadows)
    {
        uploadShadowMapSize(ivec2(2048, 2048));

        this->upload(8, light->shadow_id);
        this->uploadDepthCuts(light->depthCuts);
        this->uploadNumCascades(light->numCascades);
        this->uploadCascadeInterpolateRange(light->cascadeInterpolateRange);
    }
}

void DirectionalLightShader::uploadViewToLightTransforms(AlignedVector<mat4>& transforms)
{
    Shader::upload(location_viewToLightTransforms, transforms.size(), transforms.data());
}

void DirectionalLightShader::uploadDepthCuts(std::vector<float>& depthCuts)
{
    Shader::upload(location_depthCuts, depthCuts.size(), depthCuts.data());
}


}  // namespace Saiga
