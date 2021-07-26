/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;


out vec2 tc;


void main() {
    tc = vec2(in_position.x,in_position.y);
    tc = tc*0.5f+0.5f;
    gl_Position = vec4(in_position.x,in_position.y,0,1);
}





##GL_FRAGMENT_SHADER

#version 330


uniform sampler2D image;

uniform vec4 screenSize;
uniform vec4 ssrData; //vec4(stride,jitter,zThickness,maxSteps)
uniform float useBinarySearch;

uniform vec2 filterRadius;
uniform float distanceThreshold;

in vec2 tc;

#include "../lighting/lighting_helper_fs.glsl"

layout(location=0) out float out_color;


const int sample_count = 16;
const vec2 poisson16[] = vec2[](    // These are the Poisson Disk Samples
                                vec2( -0.94201624,  -0.39906216 ),
                                vec2(  0.94558609,  -0.76890725 ),
                                vec2( -0.094184101, -0.92938870 ),
                                vec2(  0.34495938,   0.29387760 ),
                                vec2( -0.91588581,   0.45771432 ),
                                vec2( -0.81544232,  -0.87912464 ),
                                vec2( -0.38277543,   0.27676845 ),
                                vec2(  0.97484398,   0.75648379 ),
                                vec2(  0.44323325,  -0.97511554 ),
                                vec2(  0.53742981,  -0.47373420 ),
                                vec2( -0.26496911,  -0.41893023 ),
                                vec2(  0.79197514,   0.19090188 ),
                                vec2( -0.24188840,   0.99706507 ),
                                vec2( -0.81409955,   0.91437590 ),
                                vec2(  0.19984126,   0.78641367 ),
                                vec2(  0.14383161,  -0.14100790 )
                               );


//http://blog.evoserv.at/index.php/2012/12/hemispherical-screen-space-ambient-occlusion-ssao-for-deferred-renderers-using-openglglsl/
void main()
{
    vec3 diffColor,position,normal,data;
    float depth;
    getGbufferData(diffColor,position,depth,normal,data,0);



    vec3 viewPos = position;
    vec3 viewNormal = normal;

//    vec2 filterRadius = vec2(20.0f) / screen_size.x;


    float ambientOcclusion = 0;
    // perform AO
    for (int i = 0; i < sample_count; ++i)
    {
        // sample at an offset specified by the current Poisson-Disk sample and scale it by a radius (has to be in Texture-Space)
        vec2 sampleTexCoord = tc + (poisson16[i] * (filterRadius));

        float sampleDepth = texture( deferred_depth, sampleTexCoord ).r;
        vec3 samplePos = reconstructPosition(sampleDepth,sampleTexCoord);


        vec3 sampleDir = normalize(samplePos - viewPos);

        // angle between SURFACE-NORMAL and SAMPLE-DIRECTION (vector from SURFACE-POSITION to SAMPLE-POSITION)
        float NdotS = max(dot(viewNormal, sampleDir), 0);
        // distance between SURFACE-POSITION and SAMPLE-POSITION
        float VPdistSP = distance(viewPos, samplePos);

        // a = distance function
        float a = 1.0 - smoothstep(distanceThreshold, distanceThreshold * 2, VPdistSP);
        // b = dot-Product
        float b = NdotS;

        ambientOcclusion += (a*b);
    }

    out_color = 1.0 - (ambientOcclusion / float(sample_count));
}


