/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/object3d.h"
#include "saiga/core/util/color.h"
#include "saiga/opengl/rendering/lighting/shadowmap.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/uniformBuffer.h"

#include <functional>

namespace Saiga
{
namespace LightColorPresets
{
// some values were taken from here
// http://planetpixelemporium.com/tutorialpages/light.html


// === Basic Lamps ===

static const vec3 Candle       = Color::srgb2linearrgb(Color(255, 147, 41));
static const vec3 Tungsten40W  = Color::srgb2linearrgb(Color(255, 197, 143));
static const vec3 Tungsten100W = Color::srgb2linearrgb(Color(255, 214, 170));
static const vec3 Halogen      = Color::srgb2linearrgb(Color(255, 241, 224));
static const vec3 CarbonArc    = Color::srgb2linearrgb(Color(255, 250, 244));


// === Special Effects ==

static const vec3 MuzzleFlash = Color::srgb2linearrgb(Color(226, 184, 34));

// === Sun Light ==

static const vec3 HighNoonSun    = Color::srgb2linearrgb(Color(255, 255, 251));
static const vec3 DirectSunlight = Color::srgb2linearrgb(Color(255, 255, 255));
static const vec3 OvercastSky    = Color::srgb2linearrgb(Color(201, 226, 255));
static const vec3 ClearBlueSky   = Color::srgb2linearrgb(Color(64, 156, 255));
}  // namespace LightColorPresets

using DepthFunction = std::function<void(Camera*)>;



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
    float volumetricDensity = 0.04f;

    // glPolygonOffset(slope, units)
    vec2 polygon_offset = vec2(2.0f, 10.0f);


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
