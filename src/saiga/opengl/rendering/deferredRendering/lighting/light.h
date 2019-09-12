/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/object3d.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/uniformBuffer.h"
#include "saiga/opengl/rendering/deferredRendering/lighting/shadowmap.h"
#include "saiga/core/util/color.h"

#include <functional>

namespace Saiga
{
class SAIGA_OPENGL_API LightShader : public DeferredShader
{
public:
    GLint location_lightColorDiffuse, location_lightColorSpecular;  // rgba, rgb=color, a=intensity [0,1]
    GLint location_depthBiasMV, location_depthTex, location_readShadowMap;
    GLint location_shadowMapSize;  // vec4(w,h,1/w,1/h)
    GLint location_invProj;        // required to compute the viewspace position from the gbuffer
    GLint location_volumetricDensity;

    virtual void checkUniforms();

    void uploadVolumetricDensity(float density);

    void uploadColorDiffuse(vec4& color);
    void uploadColorDiffuse(vec3& color, float intensity);

    void uploadColorSpecular(vec4& color);
    void uploadColorSpecular(vec3& color, float intensity);

    void uploadDepthBiasMV(const mat4& mat);
    void uploadDepthTexture(std::shared_ptr<TextureBase> texture);
    void uploadShadow(float shadow);
    void uploadShadowMapSize(ivec2 s);
    void uploadInvProj(const mat4& mat);
};


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

class SAIGA_OPENGL_API Light : public Object3D
{
public:
    vec4 colorDiffuse  = make_vec4(1);
    vec4 colorSpecular = make_vec4(1);
    // density of the participating media
    float volumetricDensity = 0.02f;



    Light() {}
    Light(const vec3& color, float intensity)
    {
        setColorDiffuse(color);
        setIntensity(intensity);
    }
    Light(const vec4& color) { setColorDiffuse(color); }
    ~Light() {}

    void setColorDiffuse(const vec3& color) { this->colorDiffuse = make_vec4(color, this->colorDiffuse[3]); }
    void setColorDiffuse(const vec4& color) { this->colorDiffuse = color; }
    void setColorSpecular(const vec3& color) { this->colorSpecular = make_vec4(color, this->colorSpecular[3]); }
    void setColorSpecular(const vec4& color) { this->colorSpecular = color; }
    void setIntensity(float f) { this->colorDiffuse[3] = f; }
    void addIntensity(float f) { this->colorDiffuse[3] += f; }

    vec3 getColorSpecular() const { return make_vec3(colorSpecular); }
    vec3 getColorDiffuse() const { return make_vec3(colorDiffuse); }
    float getIntensity() const { return colorDiffuse[3]; }

    void setActive(bool _active) { this->active = _active; }
    bool isActive() const { return active; }
    void setVisible(bool _visible) { this->visible = _visible; }
    bool isVisible() const { return visible; }
    void setSelected(bool _selected) { this->selected = _selected; }
    bool isSelected() const { return selected; }
    void setVolumetric(bool _volumetric) { this->volumetric = _volumetric; }
    bool isVolumetric() const { return volumetric; }


    bool hasShadows() const { return castShadows; }
    void enableShadows() { castShadows = true; }
    void disableShadows() { castShadows = false; }
    void setCastShadows(bool s) { castShadows = s; }

    bool shouldCalculateShadowMap() { return castShadows && active && !culled; }
    bool shouldRender() { return active && !culled; }

    void bindUniformsStencil(MVPShader& shader);

    /**
     * computes the transformation matrix from view space of "camera" to
     * fragment space of "shadowCamera" .
     * This is used in shadow mapping for all light types.
     */
    static mat4 viewToLightTransform(const Camera& camera, const Camera& shadowCamera);

    void renderImGui();

protected:
    bool visible = true, active = true, selected = false, culled = false;
    // shadow map
    bool castShadows = false;
    bool volumetric  = false;

};

}  // namespace Saiga
