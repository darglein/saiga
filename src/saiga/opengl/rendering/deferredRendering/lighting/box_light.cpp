/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/deferredRendering/lighting/box_light.h"

#include "saiga/core/imgui/imgui.h"

namespace Saiga
{
void BoxLightShader::checkUniforms()
{
    LightShader::checkUniforms();
}


//==================================


BoxLight::BoxLight() {}

void BoxLight::createShadowMap(int w, int h, ShadowQuality quality)
{
    shadowmap = std::make_shared<SimpleShadowmap>(w, h, quality);
}

void BoxLight::bindUniforms(std::shared_ptr<BoxLightShader> shader, Camera* cam)
{
    if (isVolumetric()) shader->uploadVolumetricDensity(volumetricDensity);
    shader->uploadColorDiffuse(colorDiffuse);
    shader->uploadColorSpecular(colorSpecular);
    shader->uploadModel(model);
    shader->uploadInvProj(inverse(cam->proj));
    shader->uploadDepthBiasMV(viewToLightTransform(*cam, this->shadowCamera));
    if (this->hasShadows())
    {
        shader->uploadDepthTexture(shadowmap->getDepthTexture());
        shader->uploadShadowMapSize(shadowmap->getSize());
    }
}

void BoxLight::setView(vec3 pos, vec3 target, vec3 up)
{
    //    this->setViewMatrix(lookAt(pos,pos + (pos-target),up));
    this->setViewMatrix(lookAt(pos, target, up));
}

void BoxLight::calculateCamera()
{
    // the camera is centred at the centre of the shadow volume.
    // we define the box only by the sides of the orthographic projection
    calculateModel();
    // trs matrix without scale
    //(scale is applied through projection matrix
    mat4 T = translate(make_vec3(position));
    mat4 R = make_mat4(rot);
    mat4 m = T * R;
    shadowCamera.setView(inverse(m));
    //    shadowCamera.setProj(-scale[0], scale[0], -scale[1], scale[1], -scale[2], scale[2]);
    shadowCamera.setProj(-scale[0], scale[0], -scale[1], scale[1], -scale[2], scale[2]);
}

bool BoxLight::cullLight(Camera* cam)
{
    // do an exact frustum-frustum intersection if this light casts shadows, else do only a quick check.
    if (this->hasShadows())
        this->culled = !this->shadowCamera.intersectSAT(cam);
    else
        this->culled = cam->sphereInFrustum(this->shadowCamera.boundingSphere) == Camera::OUTSIDE;
    return culled;
}

bool BoxLight::renderShadowmap(DepthFunction f, UniformBuffer& shadowCameraBuffer)
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

void BoxLight::renderImGui()
{
    Light::renderImGui();
}

}  // namespace Saiga
