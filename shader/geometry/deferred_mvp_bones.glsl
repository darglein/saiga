
##GL_VERTEX_SHADER

#version 400

#define BONE_COUNT 15
#define BONES_PER_VERTEX 4
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec3 in_color;
layout(location=3) in vec3 in_data;
layout(location=4) in vec4 in_boneIndices;
layout(location=5) in vec4 in_boneWeights;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform mat4 MV;
uniform mat4 MVP;

uniform mat4 boneMatrices[BONE_COUNT];

layout (std140) uniform boneMatricesBlock {
    mat4 boneMatrices2[BONE_COUNT];
};


out vec3 normal;
out vec3 normalW;
out vec3 vertexMV;
out vec3 vertex;
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
//    gl_Position = vec4( in_position, 1 );

    mat4 boneMatrix = calculateBoneMatrix();
    mat4 newModel = model * boneMatrix;
    color = in_color;
    normal = normalize(vec3(view*newModel * vec4( in_normal, 0 )));
    normalW = normalize(vec3(newModel * vec4( in_normal, 0 )));
    vertexMV = vec3(view * newModel * vec4( in_position, 1 ));
    vertex = vec3(newModel * vec4( in_position, 1 ));
    data = in_data;
    gl_Position = proj*view *newModel* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 400
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

in vec3 normal;
in vec3 normalW;
in vec3 vertexMV;
in vec3 vertex;
in vec3 color;
in vec3 data;

#include "geometry/deferred_fs.glsl"



void main() {
    setGbufferData(color,vertexMV,normal,data);
}


