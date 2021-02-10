/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"

namespace Saiga
{
PointLight::PointLight()
{
    polygon_offset = vec2(2.0, 100.0);
}


mat4 PointLight::ModelMatrix()
{
    vec3 scale    = make_vec3(radius);
    vec3 position = (this->position);

    return createTRSmatrix(position, quat::Identity(), scale);
}

void PointLight::bindUniforms(std::shared_ptr<PointLightShader> shader, Camera* cam)
{
    shader->uploadA(attenuation, radius);

    if (volumetric) shader->uploadVolumetricDensity(volumetricDensity);
    shader->uploadColorDiffuse(colorDiffuse, intensity);
    shader->uploadColorSpecular(colorSpecular, intensity_specular);

    shader->uploadModel(ModelMatrix());
    shader->uploadShadowPlanes(this->shadowCamera.zFar, this->shadowCamera.zNear);
    shader->uploadInvProj(inverse(cam->proj));
    if (this->castShadows)
    {
        shader->uploadDepthBiasMV(viewToLightTransform(*cam, this->shadowCamera));
        shader->uploadDepthTexture(shadowmap->getDepthTexture());
        shader->uploadShadowMapSize(shadowmap->getSize());
    }
    assert_no_glerror();
}



void PointLight::createShadowMap(int w, int h, ShadowQuality quality)
{
    shadowmap   = std::make_unique<CubeShadowmap>(w, h, quality);
    castShadows = true;
}



struct CameraDirection
{
    GLenum CubemapFace;
    vec3 Target;
    vec3 Up;
};

static const CameraDirection gCameraDirections[] = {
    {GL_TEXTURE_CUBE_MAP_POSITIVE_X, vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)},
    {GL_TEXTURE_CUBE_MAP_NEGATIVE_X, vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)},
    {GL_TEXTURE_CUBE_MAP_POSITIVE_Y, vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f)},
    {GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, vec3(0.0f, -1.0f, 0.0f), vec3(0.0f, 0.0f, -1.0f)},
    {GL_TEXTURE_CUBE_MAP_POSITIVE_Z, vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, -1.0f, 0.0f)},
    {GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, vec3(0.0f, 0.0f, -1.0f), vec3(0.0f, -1.0f, 0.0f)}};


void PointLight::bindFace(int face)
{
    shadowmap->bindCubeFace(gCameraDirections[face].CubemapFace);
}

void PointLight::calculateCamera(int face)
{
    vec3 pos(position);
    vec3 dir(gCameraDirections[face].Target);
    vec3 up(gCameraDirections[face].Up);
    shadowCamera.setView(pos, pos + dir, up);
    shadowCamera.setProj(90.0f, 1, shadowNearPlane, radius);
}

bool PointLight::cullLight(Camera* cam)
{
    Sphere s(position, radius);
    this->culled = cam->sphereInFrustum(s) == Camera::OUTSIDE;
    //    this->culled = false;
    //    std::cout<<culled<<endl;
    return culled;
}

bool PointLight::renderShadowmap(DepthFunction f, UniformBuffer& shadowCameraBuffer)
{
    if (shouldCalculateShadowMap())
    {
        for (int i = 0; i < 6; i++)
        {
            bindFace(i);
            calculateCamera(i);
            shadowCamera.recalculatePlanes();
            CameraDataGLSL cd(&shadowCamera);
            shadowCameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
            f(&shadowCamera);
            shadowmap->unbindFramebuffer();
        }
        return true;
    }
    else
    {
        return false;
    }
}

void PointLight::renderImGui()
{
    LightBase::renderImGui();
    LightDistanceAttenuation::renderImGui();
    ImGui::InputFloat("shadowNearPlane", &shadowNearPlane);
}

}  // namespace Saiga
