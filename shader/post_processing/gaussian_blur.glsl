/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 330

#include "post_processing_helper_fs.glsl"


const float gauss[9] = float[](0.077847,	0.123317,	0.077847,
				 0.123317,	0.195346,	0.123317,
				  0.077847,	0.123317,	0.077847);

const float sobelx[9] = float[](1,0,-1,
				 2,0,-2,
				 1,0,-1);

const float sobely[9] = float[](1,2,1,
				 0,0,0,
				 -1,-2,-1);




vec4 filter3x3(const float kernel[9]){
    vec4 color;

    color += kernel[0]*texture( image, tc + vec2(-screenSize.z,-screenSize.w) );
    color += kernel[1]*texture( image, tc + vec2(0,-screenSize.w) ).rgb;
    color += kernel[2]*texture( image, tc + vec2(screenSize.z,-screenSize.w) );

    color += kernel[3]*texture( image, tc + vec2(-screenSize.z,0) );
    color += kernel[4]*texture( image, tc + vec2(0,0) ).rgb;
    color += kernel[5]*texture( image, tc + vec2(screenSize.z,0) );

    color += kernel[6]*texture( image, tc + vec2(-screenSize.z,screenSize.w) );
    color += kernel[7]*texture( image, tc + vec2(0,screenSize.w) );
    color += kernel[8]*texture( image, tc + vec2(screenSize.z,screenSize.w) );

    return color;
}

void main() {



//    out_color = texture( image, tc  ).rgb;
//     out_color = filter3x3(sobelx) + filter3x3(sobely);
    out_color = filter3x3(gauss);
}


