#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 400

#include "post_processing_helper_fs.glsl"



void main() {
    vec4 color = texture( image, tc );
    //fxaa needs luma in non linear color space
    //because srgb is linear we need sqrt here
    color.a = sqrt(dot(color.rgb, vec3(0.299, 0.587, 0.114))); // compute luma
//    color.a = dot(color.rgb, vec3(0.299, 0.587, 0.114)); // compute luma
    out_color = color;
}


