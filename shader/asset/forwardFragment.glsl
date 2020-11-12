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

#define MAX_PL_COUNT 1024

layout (std140) uniform lightDataBlock
{
    vec4 plPositions[MAX_PL_COUNT];
    vec4 plColors[MAX_PL_COUNT];
    vec4 plAttenuations[MAX_PL_COUNT];
    int plCount;
};

void render(AssetMaterial material, vec3 position, vec3 normal)
{
#if defined(DEFERRED)
    setGbufferData(vec3(material.color), normal, material.data);
#elif defined(DEPTH)
#else
    vec3 lighting = vec3(0);

    for(int c = 0; c < plCount; ++c)
    {
        vec3 plPosition = (view * vec4(plPositions[c].rgb, 0.0)).rgb;
        vec4 plColor = plColors[c];
        vec4 plAttenuation = plAttenuations[c];


        vec3 fragmentLightDir = normalize(plPosition - position);
        float intensity = plColor.w;

        float visibility = 1.0;

        float att = getAttenuation(plAttenuation, distance(position, plPosition));
        float localIntensity = intensity * att * visibility;

        float Idiff = localIntensity * intensityDiffuse(normal, fragmentLightDir);
        float Ispec = 0;
        if(Idiff > 0)
            Ispec = localIntensity * material.data.x  * intensitySpecular(position, normal, fragmentLightDir, 40);


        vec3 color = plColor.rgb * (
                    Idiff * material.color.rgb +
                    Ispec * plColor.w * plColor.rgb);

        lighting += color;
    }

    float Iamb = 0.02;
    lighting += Iamb * material.color.rgb;

    out_color = vec4(lighting, 1);
#endif
}

