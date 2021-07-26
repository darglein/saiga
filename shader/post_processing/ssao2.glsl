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
    tc = in_position.xy * 0.5f + 0.5f;
    gl_Position = vec4(in_position.x,in_position.y,0,1);
}





##GL_FRAGMENT_SHADER
#version 430


uniform sampler2D randomImage;

in vec2 tc;

//	ssao uniforms:
const int MAX_KERNEL_SIZE = 128;
uniform int uKernelSize;
uniform vec3 uKernelOffsets[MAX_KERNEL_SIZE];
uniform float radius = 1.5;
uniform float power = 1.0;

#include "../lighting/lighting_helper_fs.glsl"

layout(location=0) out float out_color;


float ssao(in mat3 kernelBasis, in vec3 originPos, in float radius) {
        float occlusion = 0.0;
        int kernelSize = uKernelSize;
        for (int i = 0; i < kernelSize; ++i) {
        //	get sample position:
                vec3 samplePos = kernelBasis * uKernelOffsets[i];
                samplePos = samplePos * radius + originPos;

//                samplePos = originPos;
        //	project sample position:
                vec4 offset = proj * vec4(samplePos, 1.0);
                offset.xy /= offset.w; // only need xy
                offset.xy = offset.xy * 0.5 + 0.5; // scale/bias to texcoords

        //	get sample depth:
                float sampleDepth = texture(deferred_depth, offset.xy).r;

//                return sampleDepth;
//                sampleDepth = linearDepth(sampleDepth, 0.1f,60.0f);
//                sampleDepth = linearizeDepth(sampleDepth, proj);

                vec3 recPos = reconstructPosition(sampleDepth, offset.xy);
                sampleDepth = recPos.z;


                float rangeCheck = smoothstep(0.0, 1.0, radius / abs(originPos.z - sampleDepth));
//                occlusion += rangeCheck * step(sampleDepth, samplePos.z);
                occlusion += rangeCheck * step(samplePos.z,sampleDepth);
        }

        occlusion = (occlusion / float(kernelSize));
        return pow(occlusion, power);
}


void main() {
//	get noise texture coords:
        vec2 noiseTexCoords = vec2(textureSize(deferred_depth, 0)) / vec2(textureSize(randomImage, 0));
        noiseTexCoords *= tc;
        vec3 rvec = texture(randomImage, noiseTexCoords).rgb * 2.0 - 1.0;

//	get view space origin:
        vec3 diffColor,vposition,normal,data;
        float depth;
        getGbufferData(tc,diffColor,vposition,depth,normal,data);

//	construct kernel basis matrix:

        vec3 tangent = normalize(rvec - normal * dot(rvec, normal));
        vec3 bitangent = cross(tangent, normal);
        mat3 kernelBasis = mat3(tangent, bitangent, normal);


        out_color = ssao(kernelBasis, vposition, radius);
}
