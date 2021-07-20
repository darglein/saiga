/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330

#define BONE_COUNT 15
#define BONES_PER_VERTEX 4


layout(location=0) in ivec4 in_boneIndices;
layout(location=1) in vec4 in_boneWeights;
layout(location=2) in vec3 in_position;
layout(location=3) in vec3 in_normal;
layout(location=4) in vec3 in_color;
layout(location=5) in vec3 in_data;

//layout(location=0) in vec3 in_position;
//layout(location=1) in vec3 in_normal;
//layout(location=2) in vec3 in_color;
//layout(location=3) in vec3 in_data;
//layout(location=4) in vec4 in_boneIndices;
//layout(location=5) in vec4 in_boneWeights;

#include "camera.glsl"
uniform mat4 model;

uniform mat4 boneMatrices[BONE_COUNT];

layout (std140) uniform boneMatricesBlock {
    mat4 boneMatrices2[BONE_COUNT];
};


out vec3 normal;
out vec3 color;
out vec3 data;


mat4 calculateBoneMatrix(){
    mat4 boneMatrix = mat4(0);
    for(int i=0;i<BONES_PER_VERTEX;++i){
        int index = int(in_boneIndices[i]);
        boneMatrix += boneMatrices2[index] * in_boneWeights[i];
    }

    return boneMatrix;
}


void main() {
    mat4 boneMatrix = calculateBoneMatrix();
    mat4 newModel = model * boneMatrix;
    color = in_color;
    data = in_data;

    normal = normalize(vec3(view*newModel * vec4( in_normal, 0 )));
    gl_Position = viewProj *newModel* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 330

uniform float userData; //blue channel of data texture in gbuffer. Not used in lighting.

in vec3 normal;
in vec3 color;
in vec3 data;

#include "geometry_helper_fs.glsl"



void main() {
    setGbufferData(color,normal,vec4(data.xy,userData,0));
}


