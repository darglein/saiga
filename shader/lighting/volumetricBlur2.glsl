/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER

#version 430


layout(binding=0, rgba16) uniform image2D destTex;

layout(local_size_x = 16, local_size_y = 16) in;


 void main() {

     ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);


     //Store operations to any texel that is outside the boundaries of the bound image will do nothing.
     vec4 color = vec4(1,0,0,0);
     imageStore(destTex, storePos, color);
 }
