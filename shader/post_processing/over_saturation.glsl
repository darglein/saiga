/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 330

#include "post_processing_helper_fs.glsl"


vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {

    ivec2 tci = ivec2(gl_FragCoord.xy);
    vec3 rgbM = texelFetch( image, tci ,0).rgb;


//    vec3 rgbM = texture(image, tc).xyz;

    vec3 hsvM = rgb2hsv(rgbM);

    hsvM.y = clamp(hsvM.y*2.0f, 0 , 1);

    out_color = vec4(hsv2rgb(hsvM),1);
    return;


}


