/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 330

//enables texture gather extension if present. Required for mesa 11.2 to work.
#extension GL_ARB_texture_gather : enable
#extension GL_ARB_gpu_shader5 : enable

#include "post_processing_helper_fs.glsl"


#define FXAA_PC 1
#define FXAA_GLSL_130 1
#define FXAA_QUALITY__PRESET 12
//#define FXAA_QUALITY__PRESET 23
//#define FXAA_QUALITY__PRESET 39
//#define FXAA_GREEN_AS_LUMA 1
#include "NVIDIA_Fxaa3_11_impl.glsl"

void main() {
    vec4 c = FxaaPixelShader(
                tc, //{xy} = center of pixel
                vec4(0), //// Used only for FXAA Console
                image, // Input color texture.
                image, // Only used on the optimized 360 version
                image, // Only used on the optimized 360 version
                screenSize.zw,// {x_} = 1.0/screenWidthInPixels {_y} = 1.0/screenHeightInPixels
                vec4(0), // Only used on FXAA Console.
                vec4(0), // Only used on FXAA Console.
                vec4(0), // Only used on FXAA Console.
                0.75, // Choose the amount of sub-pixel aliasing removal.
                0.166, // The minimum amount of local contrast required to apply algorithm.
                0.0833, // Trims the algorithm from processing darks.
                0,0,0,vec4(0)  // Only used on FXAA Console.
                );

    out_color = vec4(c);
}


