/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/spot_light.h"

#include "saiga/core/imgui/imgui.h"

namespace Saiga
{
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

SpotLight::SpotLight()
{
    polygon_offset = vec2(2.0, 100.0);
}


void SpotLight::calculateCamera()
{
    mat4 M   = ModelMatrix();
    vec3 pos = vec3(getPosition());


    vec3 dir = make_vec3(M.col(1));
    vec3 up  = make_vec3(M.col(0));

    shadowCamera.setView(pos, pos - dir, up);
    shadowCamera.setProj(2 * angle, 1, shadowNearPlane, radius);
}

void SpotLight::bindUniforms(std::shared_ptr<SpotLightShader> shader, Camera* cam)
{
    shader->uploadA(attenuation, radius);

    if (volumetric) shader->uploadVolumetricDensity(volumetricDensity);
    shader->uploadColorDiffuse(colorDiffuse, intensity);
    shader->uploadColorSpecular(colorSpecular, intensity_specular);

    float cosa = cos(radians(angle * 0.95f));  // make border smoother
    shader->uploadAngle(cosa);
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



void SpotLight::createShadowMap(int w, int h, ShadowQuality quality)
{
    shadowmap   = std::make_unique<SimpleShadowmap>(w, h, quality);
    castShadows = true;
}

mat4 SpotLight::ModelMatrix()
{
    float l = tan(radians(angle)) * radius;
    vec3 scale(l, radius, l);
    quat rot = rotation(normalize(direction), vec3(0, -1, 0));
    return createTRSmatrix((position), rot, scale);
}

void SpotLight::setAngle(float value)
{
    this->angle = value;
}

void SpotLight::setDirection(vec3 dir)
{
    direction = dir;
}

bool SpotLight::cullLight(Camera* cam)
{
    // do an exact frustum-frustum intersection if this light casts shadows, else do only a quick check.
    if (this->castShadows)
        this->culled = !this->shadowCamera.intersectSAT(*cam);
    else
        this->culled = cam->sphereInFrustum(this->shadowCamera.boundingSphere) == Camera::OUTSIDE;

    return culled;
}

bool SpotLight::renderShadowmap(DepthFunction f, UniformBuffer& shadowCameraBuffer)
{
    if (shouldCalculateShadowMap())
    {
        shadowmap->bindFramebuffer();
        shadowCamera.recalculatePlanes();
        CameraDataGLSL cd(&shadowCamera);
        shadowCameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
        f(&shadowCamera);
        shadowmap->unbindFramebuffer();
        return true;
    }
    else
    {
        return false;
    }
}

void SpotLight::renderImGui()
{
    LightBase::renderImGui();
    LightDistanceAttenuation::renderImGui();
    ImGui::SliderFloat("Angle", &angle, 0, 85);

    ImGui::InputFloat("shadowNearPlane", &shadowNearPlane);
}

}  // namespace Saiga
