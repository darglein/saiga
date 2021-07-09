/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_VERTEX_SHADER
#version 330

layout(location=0) in vec3 in_position;
#include "camera.glsl"
uniform mat4 model;

out vec2 tc;

void main() {
    tc = vec2(in_position.x,in_position.y);
    tc = tc * 0.5f + 0.5f;

    vec3 pos = in_position;
    pos.z = 0.99999;
    gl_Position =  vec4(pos,1);
}



##GL_FRAGMENT_SHADER
#version 330

#include "camera.glsl"
uniform mat4 model;
uniform sampler2D image;

in vec2 tc;

layout(location=0) out vec3 out_color;

uniform bool cylindricMapping;
const float PI = 3.1415927;

vec2 SphericalCoordinates(vec3 r)
{
    vec2 lookup_coord = r.xy;
    // Note, coordinate system is (mathematical [x,y,z]) => (here: [x,-z,y])
    // also Note, r is required to be normalized for theta.
    // spheric
    float phi = atan(-r.z, r.x);    // angle in [-PI, PI]
    float x = phi / (2.0 * PI);     // in [-.5,.5]
    x = fract(x);                   // uv-coord in [0,1]    // is not needed. just for convenience (but it changes seam-position)

    float theta = acos(r.y);    // [1,-1] ->  [0,Pi]    // acos(r.y/length(r)), if r not normalized
    float y = theta / PI;       // uv in [0,1]
    y = 1 - y;                  // texture-coordinate-y is flipped in opengl

    lookup_coord = vec2(x, y);

    return lookup_coord;
}

vec3 reconstructPosition(mat4 invProj, float d, vec2 tc){
    vec4 p = vec4(tc.x,tc.y,d,1)*2.0f - 1.0f;
    p = inverse(view) * inverse(proj) * p;
    return p.xyz/p.w;
}

void main() {

    vec3 pos = reconstructPosition(inverse(proj), 0.5, tc);
    vec3 camera_pos = inverse(view)[3].rgb;

    vec3 view_dir = normalize(camera_pos - pos);

    vec2 spher = SphericalCoordinates(view_dir);
    out_color = texture(image,spher).rgb;
}


