/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/rendering/deferredRendering/lighting/spot_light.h"

#include "saiga/imgui/imgui.h"

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

SpotLight::SpotLight() {}


void SpotLight::calculateCamera()
{
    vec3 dir = vec3(this->getUpVector());
    vec3 pos = vec3(getPosition());
    vec3 up  = vec3(getRightVector());
    shadowCamera.setView(pos, pos - dir, up);
    shadowCamera.setProj(2 * angle, 1, shadowNearPlane, cutoffRadius);
}

void SpotLight::bindUniforms(std::shared_ptr<SpotLightShader> shader, Camera* cam)
{
    AttenuatedLight::bindUniforms(shader, cam);
    float cosa = cos(radians(angle * 0.95f));  // make border smoother
    shader->uploadAngle(cosa);
    shader->uploadShadowPlanes(this->shadowCamera.zFar, this->shadowCamera.zNear);
    shader->uploadInvProj(inverse(cam->proj));
    if (this->hasShadows())
    {
        shader->uploadDepthBiasMV(viewToLightTransform(*cam, this->shadowCamera));
        shader->uploadDepthTexture(shadowmap->getDepthTexture());
        shader->uploadShadowMapSize(shadowmap->getSize());
    }
    assert_no_glerror();
}


void SpotLight::recalculateScale()
{
    float l = tan(radians(angle)) * cutoffRadius;
    vec3 scale(l, cutoffRadius, l);
    this->setScale(scale);
}

void SpotLight::setRadius(float value)
{
    cutoffRadius = value;
    recalculateScale();
}

void SpotLight::createShadowMap(int w, int h, ShadowQuality quality)
{
    //    Light::createShadowMap(resX,resY);
    //    float farplane = 50.0f;
    shadowmap = std::make_shared<SimpleShadowmap>(w, h, quality);
    //    shadowmap->createFlat(w,h);
}

void SpotLight::setAngle(float value)
{
    this->angle = value;
    recalculateScale();
}

void SpotLight::setDirection(vec3 dir)
{
    rot = rotation(normalize(dir), vec3(0, -1, 0));
}

bool SpotLight::cullLight(Camera* cam)
{
    // do an exact frustum-frustum intersection if this light casts shadows, else do only a quick check.
    if (this->hasShadows())
        this->culled = !this->shadowCamera.intersectSAT(cam);
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
    AttenuatedLight::renderImGui();
    if (ImGui::SliderFloat("Angle", &angle, 0, 85))
    {
        recalculateScale();
        calculateModel();
    }
    ImGui::InputFloat("shadowNearPlane", &shadowNearPlane);
}

}  // namespace Saiga
