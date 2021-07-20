/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 330

#include "post_processing_helper_fs.glsl"



//simple pass through shader
void main() {
    ivec2 tci = ivec2(gl_FragCoord.xy);
    out_color = texelFetch( image, tci ,0);


//    out_color = texture( image, tc );
}


