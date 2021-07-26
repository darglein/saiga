/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef SHADOW_SAMPLES_X
#define SHADOW_SAMPLES_X 16
#endif

struct ShadowData
{
    mat4 view_to_light;
    vec2 shadow_planes;
    vec2 inv_shadow_map_size;
};

layout (std430, binding = 10) buffer shadowDataBlock
{
    ShadowData shadow_data[];
};


float offset_lookup(sampler2DShadow map, vec4 loc, vec2 offset)
{
    vec2 texmapscale = shadowMapSize.zw;
    vec4 pos = vec4(loc.xy + offset * texmapscale * loc.w, loc.z, loc.w);
    pos = pos/pos.w;

    return texture(map,pos.xyz);
}


float offset_lookup_array(sampler2DArrayShadow map, vec4 loc, vec2 offset, int layer)
{
    vec2 texmapscale = shadowMapSize.zw;
    vec4 pos = vec4(loc.xy + offset * texmapscale * loc.w, loc.z, loc.w);
    pos = pos/pos.w;
    return texture(map,vec4(pos.x,pos.y,layer,pos.z));
}


//shadow map filtering from here: http://http.developer.nvidia.com/GPUGems/gpugems_ch11.html
//classic pcf for cascaded shadow mapping
float calculateShadowPCFArray(ShadowData sd, sampler2DArrayShadow shadowmap, int layer, vec3 position){
    vec4 shadowPos = sd.view_to_light * vec4(position,1);
    float visibility = 1.0f;

    float sum = 0;

    float s = SHADOW_SAMPLES_X * 0.5f - 0.5f;
    float samples = SHADOW_SAMPLES_X * SHADOW_SAMPLES_X;

    for (float y = -s; y <= s; y += 1.0f)
        for (float x = -s; x <= s; x += 1.0f)
            sum += offset_lookup_array(shadowmap, shadowPos, vec2(x, y),layer);
    visibility = sum / samples;

    return visibility;
}


float VectorToDepth (vec3 Vec, float farplane, float nearplane)
{
    vec3 AbsVec = abs(Vec);
    float LocalZcomp = max(AbsVec.x, max(AbsVec.y, AbsVec.z));

    float f = farplane;
    float n = nearplane;

    float NormZComp = (f+n) / (f-n) - (2*f*n)/(f-n)/LocalZcomp;
    return (NormZComp + 1.0) * 0.5;
}


float calculateShadowCube(ShadowData sd, samplerCubeArrayShadow tex, vec3 lightW, vec3 position_view, int layer){

    float farplane = sd.shadow_planes.x;
    float nearplane = sd.shadow_planes.y;

    vec3 fragW = vec3(sd.view_to_light*vec4(position_view,1));
    vec3 direction =  fragW-lightW;
    float visibility = 1.0f;


    float d = VectorToDepth(direction,farplane,nearplane);
    direction = normalize(direction);

    // visibility = texture(tex, vec4(direction,layer), d);
    visibility = texture(tex, vec4(direction,layer), d);
    return visibility;
}
