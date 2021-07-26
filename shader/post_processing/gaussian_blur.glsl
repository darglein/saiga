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




vec4 filter3x3(const float filter[9]){
    vec4 color;

    color += filter[0]*texture( image, tc + vec2(-screenSize.z,-screenSize.w) );
    color += filter[1]*texture( image, tc + vec2(0,-screenSize.w) ).rgb;
    color += filter[2]*texture( image, tc + vec2(screenSize.z,-screenSize.w) );

    color += filter[3]*texture( image, tc + vec2(-screenSize.z,0) );
    color += filter[4]*texture( image, tc + vec2(0,0) ).rgb;
    color += filter[5]*texture( image, tc + vec2(screenSize.z,0) );

    color += filter[6]*texture( image, tc + vec2(-screenSize.z,screenSize.w) );
    color += filter[7]*texture( image, tc + vec2(0,screenSize.w) );
    color += filter[8]*texture( image, tc + vec2(screenSize.z,screenSize.w) );

    return color;
}

void main() {



//    out_color = texture( image, tc  ).rgb;
//     out_color = filter3x3(sobelx) + filter3x3(sobely);
    out_color = filter3x3(gauss);
}


