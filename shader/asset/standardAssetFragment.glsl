/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#if defined(DEFERRED)
#include "geometry/geometry_helper_fs.glsl"
#elif defined(DEPTH)
#else
layout(location=0) out vec4 out_color;
#include "lighting/light_models.glsl"
#include "camera.glsl"
#endif


struct AssetMaterial
{
    vec4 color;
    vec4 data;
};


void render(AssetMaterial material, vec3 position, vec3 normal)
{
#if defined(DEFERRED)
    setGbufferData(vec3(material.color),normal,material.data);
#elif defined(DEPTH)
#else
    vec3 light_direction = (view * vec4(1.0, 1.25, 2.0, 0.0)).rgb;
    float intensity       = 0.5;
    float Iamb = 0.05;
    float Idiff = intensity * intensityDiffuse(normal, light_direction);

    vec4 color = material.color * min(Idiff + Iamb, 1.0);

    out_color = vec4(color.rgb, 1);
#endif
}

