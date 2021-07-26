/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 330

#include "post_processing_helper_fs.glsl"


// FXAA shader, GLSL code adapted from:
// http://horde3d.org/wiki/index.php5?title=Shading_Technique_-_FXAA
//https://code.google.com/p/processing/source/browse/trunk/processing
// Whitepaper describing the technique:
// http://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf


void main() {
    // The parameters are hardcoded for now, but could be
    // made into uniforms to control fromt he program.
    float FXAA_SPAN_MAX = 8.0;
    float FXAA_REDUCE_MUL = 1.0/8.0;
    float FXAA_REDUCE_MIN = (1.0/128.0);

    vec3 rgbNW = texture2D(image, tc + (vec2(-1.0, -1.0) * screenSize.zw)).xyz;
    vec3 rgbNE = texture2D(image, tc + (vec2(+1.0, -1.0) * screenSize.zw)).xyz;
    vec3 rgbSW = texture2D(image, tc + (vec2(-1.0, +1.0) * screenSize.zw)).xyz;
    vec3 rgbSE = texture2D(image, tc + (vec2(+1.0, +1.0) * screenSize.zw)).xyz;
    vec3 rgbM = texture2D(image, tc).xyz;

    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM = dot( rgbM, luma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

    float rcpDirMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);

    dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
	      max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX), dir * rcpDirMin)) * screenSize.zw;

    vec3 rgbA = (1.0/2.0) * (
		texture2D(image, tc + dir * (1.0/3.0 - 0.5)).xyz +
		texture2D(image, tc + dir * (2.0/3.0 - 0.5)).xyz);
    vec3 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (
		texture2D(image, tc + dir * (0.0/3.0 - 0.5)).xyz +
		texture2D(image, tc + dir * (3.0/3.0 - 0.5)).xyz);
    float lumaB = dot(rgbB, luma);

    if((lumaB < lumaMin) || (lumaB > lumaMax)){
	out_color.xyz=rgbA;
    } else {
	out_color.xyz=rgbB;
    }

}


