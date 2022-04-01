/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER
#version 330
#extension GL_ARB_explicit_uniform_location : enable

layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec4 in_color;

#include "camera.glsl"
uniform mat4 model;
layout(location=0) uniform float point_radius;
out vec4 color;
out float radius;

void main() {
    color = in_color;
    gl_Position = model * vec4(in_position.xyz,1);
    radius = point_radius;
}



##GL_GEOMETRY_SHADER
#version 400
#extension GL_ARB_explicit_uniform_location : enable
layout(points) in;
in vec4[1] color;
in float[1] radius;
layout(triangle_strip, max_vertices=4) out;

#include "camera.glsl"
uniform mat4 model;
layout(location=1) uniform int cull_backface;
out vec2 tc;
out vec4 color2;
out vec4 centerV;
out float r;
out vec3 dir;
out vec4 posW;
out vec4 posV;


void main() {
    if (color[0].a == 0.0) {
        return;
    }
    //create a billboard with the given radius
    vec4 centerWorld = gl_in[0].gl_Position;


    vec3 eyePos = -view[3].xyz * mat3(view);

    vec3 up = vec3(0, 1, 0);

    #ifdef SHADOW
    dir = normalize(transpose(view)[2].xyz);
    #else
    dir = normalize(eyePos-vec3(centerWorld));
    #endif

    vec3 right = normalize(cross(up, dir));
    up = normalize(cross(dir, right));

    centerV = view*centerWorld;
    r = radius[0];
    color2 = color[0];

    float dx=radius[0];
    float dy=radius[0];

    vec4 ix=vec4(-1, 1, -1, 1);
    vec4 iy=vec4(-1, -1, 1, 1);
    vec4 tx=vec4(0, 1, 0, 1);
    vec4 ty=vec4(0, 0, 1, 1);


    for (int i =0; i<4;i++){
        tc.x = tx[i];
        tc.y = ty[i];
        posW = vec4(ix[i]*dx * right + iy[i]*dy * up, 0) + centerWorld;
        posV = view * posW;
        gl_Position = proj*posV;
        EmitVertex();
    }
}


##GL_FRAGMENT_SHADER
#version 400


in float r;
in vec4 color2;
in vec2 tc;
in vec4 centerV;
in vec4 posW;
in vec4 posV;
in vec3 dir;

#include "camera.glsl"
uniform mat4 model;


#include "geometry/geometry_helper_fs.glsl"



void main() {
    vec2 reltc = tc*2-vec2(1);
    reltc *= r;
    float lensqr = dot(reltc, reltc);
    if(lensqr > r*r)
    discard;

    //solving x^2 + y^2 + z^2 = r^2 for z
    float z = sqrt(r*r - lensqr);

    vec4 vertexMV = posV;
    vertexMV.z += z;

    vec3 n = normalize(vec3(vertexMV)-vec3(centerV));
    vec4 fragPosP = proj * vertexMV;
    fragPosP /= fragPosP.w;

    #ifdef WRITE_DEPTH
    float out_d = fragPosP.z * 0.5f + 0.5f;

    #ifdef SHADOW
    out_d += 0.001f;
    #endif
    gl_FragDepth = out_d;
    #endif

    #ifndef SHADOW
    vec3 data = vec3(1, 0, 0);
    setGbufferData(vec3(color2), n, vec4(data.xy, 0, 0));
    #endif
}

