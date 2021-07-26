/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#ifdef SINGLE_PASS_LIGHTING
#include "uber_lighting_helpers.glsl"
#else
#include "camera.glsl"
#include "light_models.glsl"
#endif

uniform mat4 model;

uniform mat4 invProj;

uniform sampler2D deferred_diffuse;
uniform sampler2D deferred_normal;
uniform sampler2D deferred_depth;
uniform sampler2D deferred_position;
uniform sampler2D deferred_data;

uniform vec4 viewPort; // x,y,width,height

//uniform vec4 lightColorDiffuse; //rgba, rgb=color, a=intensity [0,1]
//uniform vec4 lightColorSpecular; //rgba, rgb=color, a=intensity [0,1]

uniform vec4 shadowMapSize = vec4(2048,2048,1./2048,1./2048);  //vec4(w,h,1/w,1/h)


float random(vec4 seed4){
    float dot_product = dot(seed4, vec4(12.9898,78.233,45.164,94.673));
    return fract(sin(dot_product) * 43758.5453);
}

vec2 CalcTexCoord()
{
//   return gl_FragCoord.xy / viewPort.zw;
       vec2 tc = (gl_FragCoord.xy -viewPort.xy)/ viewPort.zw;
       return tc;
}




vec3 unpackNormal2 (vec2 enc)
{
    vec3 n;
    n.z=length(enc)*2-1;

    n.xy= normalize(enc)*sqrt(1-n.z*n.z);
    return n;
}

vec3 unpackNormal3 (vec2 enc)
{
    vec2 fenc = enc*4-vec2(2);
    float f = dot(fenc,fenc);
    float g = sqrt(1-f/4);
    vec3 n;
    n.xy = fenc*g;
    n.z = 1-f/2;
    return n;
}

float linearizeDepth(in float depth, in mat4 projMatrix) {
        return projMatrix[3][2] / (depth - projMatrix[2][2]);
}

float linearDepth(float depth, float farplane, float nearplane)
{
    return (2 * nearplane) / (farplane + nearplane - depth * (farplane - nearplane));
}

float nonlinearDepth(float linearDepth, float farplane, float nearplane)
{
    float A = -(farplane+nearplane) / (farplane - nearplane);
    float B = (-2.f * farplane * nearplane) / (farplane - nearplane);
    return 0.5f * ( -A * linearDepth + B) / linearDepth + 0.5f;
}

vec3 reconstructPosition(float d, vec2 tc){
    vec4 p = vec4(tc.x,tc.y,d,1)*2.0f - 1.0f;
    p = invProj * p;
    return p.xyz/p.w;
}



void getGbufferData(out vec3 color,out  vec3 position, out float depth, out vec3 normal, out vec3 data, int sampleId){
    vec2 tc = CalcTexCoord();
    ivec2 tci = ivec2(gl_FragCoord.xy);

    color = texelFetch( deferred_diffuse, tci ,sampleId).rgb;

    depth = texelFetch( deferred_depth, tci ,sampleId).r;
    position = reconstructPosition(depth,tc);

    normal = texelFetch( deferred_normal, tci,sampleId).xyz;
    normal = unpackNormal3(normal.xy);

    data = texelFetch(deferred_data,tci,sampleId).xyz;
}


void getGbufferData(vec2 tc, out vec3 color,out  vec3 position, out float depth, out vec3 normal, out vec3 data){

    color = texture( deferred_diffuse, tc).rgb;

    depth = texture( deferred_depth, tc).r;
    position = reconstructPosition(depth,tc);

    normal = texture( deferred_normal, tc).xyz;
    normal = unpackNormal3(normal.xy);

    data = texture(deferred_data,tc).xyz;
}

#include "shadows.glsl"
