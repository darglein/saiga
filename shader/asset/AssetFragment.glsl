/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#if defined(DEFERRED)
#include "geometry/geometry_helper_fs.glsl"
#elif defined(DEPTH)
#elif defined(FORWARD_LIT)
layout(location=0) out vec4 out_color;
#include "lighting/uber_lighting_helpers.glsl"
#else
layout(location=0) out vec4 out_color;
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
#elif defined(FORWARD_LIT)
    vec3 lighting = vec3(0);

    lighting += calculatePointLights(material, position, normal, gl_FragCoord.z);
    lighting += calculateSpotLights(material, position, normal, gl_FragCoord.z);
    lighting += calculateDirectionalLights(material, position, normal);

    out_color = vec4(lighting, 1);
#else
    out_color = material.color;
#endif
}

