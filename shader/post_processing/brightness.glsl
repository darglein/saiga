/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 330

uniform sampler2D image;

uniform float brightness = 0.5f;
in vec2 tc;

layout(location=0) out vec4 out_color;


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


vec3 adjustBrightness(vec3 color, float brightness){
    brightness = 1.0f - brightness;
    brightness *= 2.0f;
    vec3 ret;

    vec3 hsv = rgb2hsv(color);
    hsv.z = pow(hsv.z,brightness);
    ret = hsv2rgb(hsv);

    return ret;
}

vec3 adjustBrightness2(vec3 color, float brightness){
    //polynomial fit with
    //(x,f(x))
    //(0,high)
    //(0.5,1)
    //(1,low)

        float high = 4.0f;
        float low = 0.4f;

    float d = 1;
    float e = 1;

    float a = mix(high-1,low-1,step(0.5f,brightness));
    float x = 2 * brightness;
    brightness = a*(x-d)*(x-d) + e;



//    float t = 2*brightness;
//    if(t < 1){
//        brightness = (2*t*t*t-3*t*t+1)*high + (-2*t*t*t+3*t*t)*mid;
//    }else{
//        t = t-1;
//        brightness = (2*t*t*t-3*t*t+1)*mid + (-2*t*t*t+3*t*t)*low;
//    }

//    brightness = 1.0f - brightness;
//    brightness *= 2.0f;
    vec3 ret;
    ret = pow(color,vec3(brightness));
    return ret;
}

void main() {
    ivec2 tci = ivec2(gl_FragCoord.xy);
    vec4 rgbM = texelFetch( image, tci ,0);

    rgbM.rgb = adjustBrightness2(rgbM.rgb,brightness);
    out_color = rgbM;
    return;
}


