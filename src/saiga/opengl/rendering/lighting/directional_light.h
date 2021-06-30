/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/camera/camera.h"
#include "saiga/core/util/Align.h"
#include "saiga/opengl/rendering/lighting/light.h"
namespace Saiga
{

class SAIGA_OPENGL_API DirectionalLight : public LightBase
{
   public:
    static constexpr int shadow_map_size = 2048;

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
    float ambientIntensity = 0.1f;

    // number of cascades for cascaded shadow mapping
    // 1 means normal shadow mapping
    int numCascades = 1;

   public:
    DirectionalLight() : LightBase(LightColorPresets::DirectSunlight, 1)
    {
        setDirection(vec3(-1, -3, -2));
        BuildCascades(1);
        polygon_offset = vec2(2.0, 50.0);
    }
    ~DirectionalLight() {}

    struct ShaderData
    {
        vec4 colorDiffuse;   // rgb intensity
        vec4 colorSpecular;  // rgb specular intensity
        vec4 direction;      // xyz, w ambient intensity
    };

    inline ShaderData GetShaderData()
    {
        ShaderData data;
        data.colorDiffuse  = make_vec4(colorDiffuse, intensity);
        data.colorSpecular = make_vec4(colorSpecular, intensity_specular);
        data.direction     = make_vec4(direction, ambientIntensity);
        return data;
    }

    inline ShadowData GetShadowData(Camera* view_point)
    {
        ShadowData sd;
        sd.view_to_light = viewToLightTransform(*view_point, shadowCamera);
        return sd;
    }

    /**
     * Creates the shadow map with the given number of cascades, and initializes depthCutsRelative
     * to a uniform range.
     * If this function is called when a shadow map was already created before,
     * the old shadow map is deleted and overwritten by the new one.
     */
    void BuildCascades(int numCascades = 1);

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


    // see description for depthCutsRelative for more info
    void setDepthCutsRelative(const std::vector<float>& value);
    std::vector<float> getDepthCutsRelative() const;

    int getNumCascades() const { return numCascades; }

    float getCascadeInterpolateRange() const { return cascadeInterpolateRange; }
    void setCascadeInterpolateRange(float value) { cascadeInterpolateRange = value; }

    void setAmbientIntensity(float ai) { ambientIntensity = ai; }
    float getAmbientIntensity() const { return ambientIntensity; }

    // the directional light is always visible
    bool cullLight(Camera*)
    {
        culled = false;
        return culled;
    }
    void renderImGui();
};

}  // namespace Saiga
