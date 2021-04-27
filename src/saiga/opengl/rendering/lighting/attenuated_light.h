/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/opengl/rendering/lighting/light.h"

namespace Saiga
{
namespace AttenuationPresets
{
static const vec3 NONE = vec3(1, 0, 0);

static const vec3 LinearWeak   = vec3(1, 0.5, 0);
static const vec3 Linear       = vec3(1, 1, 0);
static const vec3 LinearStrong = vec3(1, 4, 0);


static const vec3 QuadraticWeak   = vec3(1, 0.5, 0.5);
static const vec3 Quadratic       = vec3(1, 1, 1);
static const vec3 QuadraticStrong = vec3(1, 2, 4);
}  // namespace AttenuationPresets


// Intensity Attenuation based on the distance to the light source.
//   - Normalized by the light radius so that we can use the same parameters for different light sizes
//   - Shifted downwards so that DistanceAttenuation(a, radius, radius) == 0
//   -> Therefore lights have a finite range and can be efficiently rendered
//
//   Used by PointLight and SpotLight
//     - Implemented in the shader light_models.glsl
inline float DistanceAttenuation(vec3 attenuation, float radius, float distance)
{
    float x         = distance / radius;
    float cutoff    = 1.f / (attenuation[0] + attenuation[1] + attenuation[2]);
    float intensity = 1.f / (attenuation[0] + attenuation[1] * x + attenuation[2] * x * x) - cutoff;
    return std::max(0.f, intensity);
}

class SAIGA_OPENGL_API LightDistanceAttenuation
{
   public:
    /**
     * Quadratic attenuation of the form:
     * I = i/(a*x*x+b*x+c)
     *
     * It is stored in the format:
     * vec3(c,b,a)
     *
     * Note: The attenuation is independent of the radius.
     * x = d / r, where d is the distance to the light
     *
     * This normalized attenuation makes it easy to scale lights without having to change the attenuation
     *
     */

    vec3 attenuation = AttenuationPresets::Quadratic;


    /**
     * Distance after which the light intensity is clamped to 0.
     * The shadow volumes should be constructed so that they closely contain
     * all points up to the cutoffradius.
     */
    float radius;

   public:
    // evaluates the attenuation formula at a given radius
    float Evaluate(float distance)
    {
        float x = distance / radius;
        return 1.0f / (attenuation(0) + attenuation[1] * x + attenuation[2] * x * x);
    }

    float getRadius() const { return radius; }
    void setRadius(float value) { radius = value; }

    //    vec3 getAttenuation() const { return attenuation; }
    //    void setAttenuation(const vec3& value) { attenuation = value; }

    void renderImGui();
};

}  // namespace Saiga
