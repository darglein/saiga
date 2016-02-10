
#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 400

#include "post_processing_helper_fs.glsl"



void main() {
    float n = 1;
    float f = 30;
    float distanceThreshold = 0.01f;

    //TODO:
    //check normals or colors


    float dNW = texture(gbufferDepth, tc + (vec2(-1.0, -1.0) * screenSize.zw)).r;
    float dNE = texture(gbufferDepth, tc + (vec2(+1.0, -1.0) * screenSize.zw)).r;
    float dSW = texture(gbufferDepth, tc + (vec2(-1.0, +1.0) * screenSize.zw)).r;
    float dSE = texture(gbufferDepth, tc + (vec2(+1.0, +1.0) * screenSize.zw)).r;



    dNW = linearDepth(dNW,n,f);
    dNE = linearDepth(dNE,n,f);
    dSW = linearDepth(dSW,n,f);
    dSE = linearDepth(dSE,n,f);

    float dMin = min(min(dNW, dNE), min(dSW, dSE));
    float dMax = max(max(dNW, dNE), max(dSW, dSE));



    out_color = texture(image,tc).rgb;

    if(dMax-dMin > distanceThreshold)
        out_color = vec3(0);
//    out_color = vec3(dMax-dMin);

}


