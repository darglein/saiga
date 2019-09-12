/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef SHADOW_SAMPLES_X
#define SHADOW_SAMPLES_X 16
#endif

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

float calculateShadow(sampler2DShadow tex, vec3 position){
    vec4 shadowPos = depthBiasMV * vec4(position,1);
    shadowPos = shadowPos/shadowPos.w;
    float visibility = texture(tex, shadowPos.xyz);
    return visibility ;
}

//shadow map filtering from here: http://http.developer.nvidia.com/GPUGems/gpugems_ch11.html


//classic pcf
float calculateShadowPCF2(mat4 viewToLight, sampler2DShadow shadowmap, vec3 position){
    vec4 shadowPos = viewToLight * vec4(position,1);
    float visibility = 1.0f;
    float sum = 0;
    float s = SHADOW_SAMPLES_X * 0.5f - 0.5f;
    float samples = SHADOW_SAMPLES_X * SHADOW_SAMPLES_X;
    for (float y = -s; y <= s; y += 1.0f)
        for (float x = -s; x <= s; x += 1.0f)
            sum += offset_lookup(shadowmap, shadowPos, vec2(x, y));
    visibility = sum / samples;
    return visibility;
}

//classic pcf for cascaded shadow mapping
float calculateShadowPCFArray(mat4 viewToLight, sampler2DArrayShadow shadowmap, int layer, vec3 position){
    vec4 shadowPos = viewToLight * vec4(position,1);
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


//4 sample pcf with dithering
float calculateShadowPCFdither4(sampler2DShadow shadowmap, vec3 position){
    vec4 shadowPos = depthBiasMV * vec4(position,1);

    vec2 offset = fract(gl_FragCoord.xy * 0.5);
    offset.x = offset.x > 0.25 ? 1.0f : 0.0f;
    offset.y = offset.y > 0.25 ? 1.0f : 0.0f;
    offset.y += offset.x;
    if (offset.y > 1.1)
      offset.y = 0;


    //this is equivalent to the above floating point code
//    vec2 pixel = gl_FragCoord.xy;
//    int y = int(pixel.y) % 2;
//    int x = int(pixel.x) % 2;
//    y ^= x;
//    vec2 offset = vec2(x,y);

     float visibility =
             offset_lookup(shadowmap, shadowPos, offset + vec2(-1.5, 0.5)) +
             offset_lookup(shadowmap, shadowPos, offset + vec2(0.5, 0.5)) +
             offset_lookup(shadowmap, shadowPos, offset + vec2(-1.5, -1.5)) +
             offset_lookup(shadowmap, shadowPos, offset + vec2(0.5, -1.5));
     visibility *= 0.25f;
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

float calculateShadowCube(samplerCubeShadow tex, vec3 lightW, vec3 fragW, float farplane, float nearplane){
    vec3 direction =  fragW-lightW;
    float visibility = 1.0f;


    float d = VectorToDepth(direction,farplane,nearplane);
    direction = normalize(direction);

    //a quick bias is applied with glpolygon offset.
    //
    const float bias = 1e-3;
    visibility = texture(tex, vec4(direction,d-bias));
    return visibility;
}

//doesn't really work (has artifacts)
float calculateShadowCubePCF(samplerCubeShadow tex, vec3 lightW, vec3 fragW, float farplane, float nearplane){
    vec3 direction =  fragW-lightW;
    float d = VectorToDepth(direction,farplane,nearplane);
    direction = normalize(direction);

    float visibility = 0.0f;

    vec3 right = normalize(cross(direction,vec3(0.14,0.68,-0.4)));
    vec3 up = normalize(cross(right,direction));

    float radius = 0.01f;

    float s = 1.5f;

        for (float y = -s; y <= s; y += 1.0f)
            for (float x = -s; x <= s; x += 1.0f){
                vec3 samplePos =  normalize(direction + right * x * radius + up * y * radius);
                visibility += texture(tex, vec4(samplePos,d));
            }

//    float s = 0.5f;
//    for (float z = -s; z <= s; z += 1.0f)
//        for (float y = -s; y <= s; y += 1.0f)
//            for (float x = -s; x <= s; x += 1.0f){
//                vec3 samplePos = direction + vec3(x,y,z) * radius;
//                visibility += texture(tex, vec4(samplePos,d));
//            }

    return visibility / 16.0f;
//    return  texture(tex, vec4(direction,d));
}
