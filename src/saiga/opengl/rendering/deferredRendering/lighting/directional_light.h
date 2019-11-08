/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/camera/camera.h"
#include "saiga/core/util/Align.h"
#include "saiga/opengl/rendering/deferredRendering/lighting/light.h"
#include "saiga/opengl/uniformBuffer.h"

namespace Saiga
{
#define MAX_CASCADES 5


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
    void uploadDepthTextures(std::shared_ptr<ArrayTexture2D> textures);
};

class SAIGA_OPENGL_API DirectionalLight : public Light
{
   protected:
    std::shared_ptr<CascadedShadowmap> shadowmap;

    // bounding box of every cascade frustum
    std::vector<AABB> orthoBoxes;



    // relative split planes to the camera near and far plane
    // must be of size numCascades + 1
    // should start with 0 and end with 1
    // values between the first and the last indicate the split planes
    std::vector<float> depthCutsRelative;

    // actual split planes in view space depth
    // will be calculated from depthCutsRelative
    std::vector<float> depthCuts;

    // shadow camera for depth map rendering
    // is different for every cascade and is set in bindCascade
    OrthographicCamera shadowCamera;

    // The size in world space units how big the interpolation region between two cascades is.
    // Larger values mean a smoother transition, but decreases performance, because more shadow samples need to be
    // fetched. Larger values also increase the size of each shadow frustum and therefore the quality may be reduceds.
    float cascadeInterpolateRange = 3.0f;


    // direction of the light in world space
    vec3 direction = vec3(0, -1, 0);

    // relative intensity to the diffuse light in ambiend regions
    float ambientIntensity = 0.2f;

    // number of cascades for cascaded shadow mapping
    // 1 means normal shadow mapping
    int numCascades = 1;

   public:
    DirectionalLight() {}
    ~DirectionalLight() {}

    /**
     * Creates the shadow map with the given number of cascades, and initializes depthCutsRelative
     * to a uniform range.
     * If this function is called when a shadow map was already created before,
     * the old shadow map is deleted and overwritten by the new one.
     */
    void createShadowMap(int w, int h, int numCascades = 1, ShadowQuality quality = ShadowQuality::LOW);

    /**
     * Sets the light direction in world coordinates.
     * Computes the view matrix for the shadow camera.
     */
    void setDirection(const vec3& dir);
    vec3 getDirection() const { return direction; }

    /**
     * Computes the left/right, bottom/top and near/far planes of the shadow volume so that,
     * it fits the given camera. This should be called every time the camera is translated
     * or rotated to be sure, that all visible objects have shadows.
     */
    void fitShadowToCamera(Camera* shadowCamera);

    /**
     * Computes the near plane of the shadow frustum so that all objects in the scene cast shadows.
     * Objects that do not lay in the camera frustum can cast shadows into it. That's why the method above
     * is not enough. This will overwrite the near plane, so it should be called after fitShadowToCamera.
     */
    void fitNearPlaneToScene(AABB sceneBB);

    /**
     * Binds the shadow framebuffer with the correct texture for this cascade
     * and sets the light camera correctly.
     */
    void bindCascade(int n);

    // Bind the uniforms for light rendering
    void bindUniforms(DirectionalLightShader& shader, Camera* shadowCamera);

    // see description for depthCutsRelative for more info
    void setDepthCutsRelative(const std::vector<float>& value);
    std::vector<float> getDepthCutsRelative() const;

    int getNumCascades() const { return numCascades; }

    float getCascadeInterpolateRange() const { return cascadeInterpolateRange; }
    void setCascadeInterpolateRange(float value) { cascadeInterpolateRange = value; }

    void setAmbientIntensity(float ai) { ambientIntensity = ai; }
    float getAmbientIntensity() { return ambientIntensity; }

    // the directional light is always visible
    bool cullLight(Camera*)
    {
        culled = false;
        return culled;
    }
    bool renderShadowmap(DepthFunction f, UniformBuffer& shadowCameraBuffer);
    void renderImGui();
};

}  // namespace Saiga
