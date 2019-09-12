/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/core/util/tostring.h"

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


void LightShader::uploadColorDiffuse(vec4& color)
{
    Shader::upload(location_lightColorDiffuse, color);
}

void LightShader::uploadColorDiffuse(vec3& color, float intensity)
{
    vec4 c = make_vec4(color, intensity);
    Shader::upload(location_lightColorDiffuse, c);
}

void LightShader::uploadColorSpecular(vec4& color)
{
    Shader::upload(location_lightColorSpecular, color);
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

// void Light::createShadowMap(int resX, int resY){

////    std::cout<<"Light::createShadowMap"<<endl;

//    shadowmap.createFlat(resX,resY);

//}


void Light::bindUniformsStencil(MVPShader& shader)
{
    shader.uploadModel(model);
}

mat4 Light::viewToLightTransform(const Camera& camera, const Camera& shadowCamera)
{
    // glm like glsl is column major!
    const mat4 biasMatrix = make_mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 1.0);
    // We could also use inverse(camera.view) but using the model matrix is faster
    return biasMatrix * shadowCamera.proj * shadowCamera.view * camera.model;
}

void Light::renderImGui()
{
    ImGui::Checkbox("active", &active);
    ImGui::Checkbox("castShadows", &castShadows);
    ImGui::Checkbox("volumetric", &volumetric);
    ImGui::InputFloat("intensity", &colorDiffuse[3], 0.1, 1);
    ImGui::InputFloat("specular intensity", &colorSpecular[3], 0.1, 1);
    ImGui::SliderFloat("volumetricDensity", &volumetricDensity, 0.0f, 0.5f);
    // todo: check srgb
    ImGui::ColorEdit3("colorDiffuse", &colorDiffuse[0]);
    ImGui::ColorEdit3("colorSpecular", &colorSpecular[0]);
    auto str = to_string(visible) + "/" + to_string(selected) + "/" + to_string(culled);
    ImGui::Text("visible/selected/culled: %s", str.c_str());
}

}  // namespace Saiga
