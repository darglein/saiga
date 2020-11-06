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

struct PointLight
{
    vec3 position;
    vec4 color; // (w is intensity)
    vec4 attenuation;
};

void render(AssetMaterial material, vec3 position, vec3 normal, PointLight pl)
{
#if defined(DEFERRED)
    setGbufferData(vec3(material.color), normal, material.data);
#elif defined(DEPTH)
#else
    float Iamb = 0.002;

    vec3 fragmentLightDir = normalize(pl.position - position);
    float intensity = pl.color.w;

    float visibility = 1.0;

    float att = getAttenuation(pl.attenuation, distance(position, pl.position));
    float localIntensity = intensity * att * visibility;

    float Idiff = localIntensity * intensityDiffuse(normal, fragmentLightDir);
    float Ispec = 0;
    if(Idiff > 0)
        Ispec = localIntensity * material.data.x  * intensitySpecular(position, normal, fragmentLightDir, 40);


    vec3 color = pl.color.rgb * (
                Iamb  * material.color.rgb +
                Idiff * material.color.rgb +
                Ispec * pl.color.w * pl.color.rgb);

    out_color = vec4(color, 1);
#endif
}

