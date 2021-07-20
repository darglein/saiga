/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
#extension GL_ARB_explicit_uniform_location : enable
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_color;

#include "camera.glsl"
uniform mat4 model;


layout(location = 0) uniform float world_radius = 0.1;

out vec3 color;
out float radius;

void main() {
    color = in_color.rgb;
    radius = world_radius;
    gl_Position = model* vec4(in_position.xyz,1);
}




##GL_GEOMETRY_SHADER
#version 400

layout(points) in;
in vec3[1] color;
in float[1] radius;
layout(triangle_strip, max_vertices=4) out;

#include "camera.glsl"
uniform mat4 model;

out vec2 tc;
out vec3 color2;
out vec4 centerV;
out float r;
out vec3 dir;
out vec4 posW;
out vec4 posV;


void main() {
//    if (color[0].a == 0.0) {
//        return;
//    }
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

#version 330

in vec3 color2;
in vec2 tc;


layout(location=0) out vec4 out_color;

void main() {
    vec2 ctc = (vec2(0.5) - tc) * 2;
    float d = dot(ctc, ctc);
//    out_color = vec4(d,d,d,1);
    if(d > 0.5) discard;

    out_color = vec4(color2,1);
}


