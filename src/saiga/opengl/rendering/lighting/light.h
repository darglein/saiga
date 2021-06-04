/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/object3d.h"
#include "saiga/core/util/color.h"

#include <functional>

namespace Saiga
{
namespace LightColorPresets
{
// some values were taken from here
// http://planetpixelemporium.com/tutorialpages/light.html


// === Basic Lamps ===

static const vec3 Candle       = (Color(255, 147, 41));
static const vec3 Tungsten40W  = (Color(255, 197, 143));
static const vec3 Tungsten100W = (Color(255, 214, 170));
static const vec3 Halogen      = (Color(255, 241, 224));
static const vec3 CarbonArc    = (Color(255, 250, 244));


// === Special Effects ==

static const vec3 MuzzleFlash = (Color(226, 184, 34));

// === Sun Light ==

static const vec3 HighNoonSun    = (Color(255, 255, 251));
static const vec3 DirectSunlight = (Color(255, 255, 255));
static const vec3 OvercastSky    = (Color(201, 226, 255));
static const vec3 ClearBlueSky   = (Color(64, 156, 255));
}  // namespace LightColorPresets

using DepthFunction = std::function<void(Camera*)>;


struct ShadowData
{
    Eigen::Matrix<float,4,4,Eigen::DontAlign> view_to_light;
    vec2 shadow_planes;
    vec2 inv_shadow_map_size;

    ShadowData() { static_assert(sizeof(ShadowData) ==  20 * sizeof(float)); }
};



class SAIGA_OPENGL_API LightBase
{
   public:
    // [R,G,B,Intensity]
    vec3 colorDiffuse = make_vec3(1);
    float intensity   = 1;

    // [R,G,B,Intensity]
    vec3 colorSpecular       = make_vec3(1);
    float intensity_specular = 1;
    // density of the participating media
    float volumetricDensity = 0.025f;

    // glPolygonOffset(slope, units)
    vec2 polygon_offset = vec2(2.0f, 10.0f);


    int shadow_id = -1;
    int active_light_id = -1;

    LightBase() {}
    LightBase(const vec3& color, float intensity)
    {
        setColorDiffuse(color);
        setIntensity(intensity);
    }
    virtual ~LightBase() {}

    void setColorDiffuse(const vec3& color) { this->colorDiffuse = color; }
    void setColorSpecular(const vec3& color) { this->colorSpecular = color; }
    void setIntensity(float f) { intensity = f; }

    vec3 getColorSpecular() const { return colorSpecular; }
    vec3 getColorDiffuse() const { return colorDiffuse; }
    float getIntensity() const { return intensity; }


    bool shouldCalculateShadowMap() { return castShadows && active && !culled; }
    bool shouldRender() { return active && !culled; }


    /**
     * computes the transformation matrix from view space of "camera" to
     * fragment space of "shadowCamera" .
     * This is used in shadow mapping for all light types.
     */
    static mat4 viewToLightTransform(const Camera& camera, const Camera& shadowCamera);

    virtual void renderImGui();

    //   protected:
    bool visible = true, active = true, selected = false, culled = false;
    // shadow map
    bool castShadows = false;
    bool volumetric  = false;
};

}  // namespace Saiga
