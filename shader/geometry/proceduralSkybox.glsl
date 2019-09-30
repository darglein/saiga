/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
#extension GL_ARB_explicit_uniform_location : enable

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_tex;

#include "camera.glsl"
uniform mat4 model = mat4(1);


out vec2 texCoord;
out vec4 pos;

out vec3 viewDir;
out vec4 eyePos;

void main()
{
    texCoord    = in_tex;
    pos         = vec4(in_position, 1);
    gl_Position = vec4(in_position, 1);


    mat4 invView  = model * inverse(view) ;
    vec4 worldPos = inverse(proj) * pos;
    worldPos /= worldPos.w;
    worldPos = invView * worldPos;


    eyePos  = invView[3];
    viewDir = vec3(worldPos - eyePos);
}



##GL_FRAGMENT_SHADER

#version 330
#extension GL_ARB_explicit_uniform_location : enable

#include "camera.glsl"
uniform mat4 model;
layout(location = 0) uniform vec4 params;
layout(location = 1) uniform vec3 lightDir = vec3(0, -1, 0);

in vec2 texCoord;
in vec4 pos;
in vec3 viewDir;
in vec4 eyePos;

layout(location = 0) out vec4 out_color;


vec3 blueSkyAndSun(vec3 viewDir2, vec3 cameraPos2, vec3 lightDir, bool linearRGB, float sunIntensity, float sunSize,
                   float horizonHeight, float skyboxDistance)
{
    vec3 darkBlueSky = vec3(43, 99, 192) / 255.0f;
    vec3 blueSky     = vec3(97, 161, 248) / 255.0f;

    if (linearRGB)
    {
        darkBlueSky = pow(darkBlueSky, vec3(2.2f));
        blueSky     = pow(blueSky, vec3(2.2f));
    }

    // direction of current viewing ray
    vec3 dir = normalize(viewDir2);

    if (length(vec2(dir.x, dir.z)) < 0.001) return vec3(0);

    // intersection point of viewing ray with cylinder around viewer with radius=skyboxDistance
    vec3 skyboxPos = vec3(cameraPos2) + dir * (skyboxDistance / length(vec2(dir.x, dir.z))) - horizonHeight;

    // this gives the tangens of the viewing ray towards the ground
    float h = skyboxPos.y / skyboxDistance;

    // exponential gradient
    float a = -exp(-h * 3) + 1;

    vec3 col = mix(blueSky, darkBlueSky, a);

    // fade out bottom border to black
    col = mix(col, vec3(0), smoothstep(0.0, -0.2, h));
    //    col = mix(col,vec3(0),step(0,-h));

    if (h < -10) return vec3(0);



    // fake sun
    vec3 ray_dir     = viewDir2;
    float middayperc = 0.015;

    float costheta = max(dot(ray_dir, normalize(-lightDir)), 0);


    float sunperc;

    {
        float x = costheta;
        float t = 0.9995;
        //        float t = 1.1;
        float b  = 1000 / sunSize;
        float ae = 1.0f / exp(t * b);
        sunperc  = exp(b * x - (t * b));
    }

    {
        float x  = costheta;
        float t  = 1;
        float b  = 200 / sunSize;
        float ae = 1.0f / exp(t * b);
        sunperc += exp(b * x - (t * b));
    }
    sunperc = max(sunperc, 0);

    vec3 suncolor = (1.0 - max(0.0, middayperc)) * vec3(1.5, 1.2, middayperc + 0.5) +
                    max(0.0, middayperc) * vec3(1.0, 1.0, 1.0) * 2.0;
    vec3 color = suncolor * sunperc;
    return sunIntensity * (col * 1.0 + 0.8 * color);
}

void main()
{
    float horizonHeight  = params.x;
    float skyboxDistance = params.y;
    float sunIntensity   = params.z;
    float sunSize        = params.w;

    vec3 vdir = normalize(viewDir);
    vec3 vpos = vec3(eyePos);

    out_color =
        vec4(blueSkyAndSun(vdir, vpos, lightDir, true, sunIntensity, sunSize, horizonHeight, skyboxDistance), 1);
}
