/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_VERTEX_SHADER

#version 430
layout(location=0) in vec3 in_position;


out vec2 tc;


void main() {
    tc = vec2(in_position.x,in_position.y);
    tc = tc * 0.5f + 0.5f;
    gl_Position = vec4(in_position.x,in_position.y,0,1);
}


##GL_FRAGMENT_SHADER
#version 430

layout(location=0) uniform sampler2D imageLeft;
layout(location=1) uniform sampler2D imageRight;
layout(location=2) uniform vec2 size;

in vec2 tc;

layout(location=0) out vec4 out_color;

//simple pass through shader
void main() {
    ivec2 leftSize = textureSize(imageLeft,0);
    ivec2 leftRight = textureSize(imageRight,0);

    vec4 col;
    vec2 tc2 = tc;
    tc2.x = tc.x * 2;
    if(tc2.x <= 1)
    {
        col = texture(imageLeft,tc2);
    }else{
        tc2.x -= 1;
        col = texture(imageRight,tc2);
    }
    col.w = 1;


    out_color = col;
    //    out_color= vec4(1);


    //    out_color = texture( image, tc );
}


