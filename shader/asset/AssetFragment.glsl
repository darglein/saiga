/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#if defined(DEFERRED)
#include "geometry/geometry_helper_fs.glsl"
#elif defined(DEPTH)
#else
layout(location=0) out vec4 out_color;
#endif


struct AssetMaterial
{
    vec4 color;
    vec4 data;
};

void render(AssetMaterial material, vec3 normal)
{
#if defined(DEFERRED)
    setGbufferData(vec3(material.color),normal,material.data);
#elif defined(DEPTH)
#else
    out_color = material.color;
#endif
}

