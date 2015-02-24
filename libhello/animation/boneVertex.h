#pragma once

#include <libhello/util/glm.h>
#include <libhello/opengl/vertexBuffer.h>


struct BoneVertex{
    vec3 position;
    vec3 normal;
    vec2 texture;

    //every vertex has a maximum of 2 bones
#define BONES_PER_VERTEX 4
    float boneIndices[BONES_PER_VERTEX] = {0};
    float boneWeights[BONES_PER_VERTEX] = {0};
//    vec2 boneWeights = vec2(0);

    BoneVertex(){}

    void addBone(int index, float weight){
        for(int i=0;i<BONES_PER_VERTEX;i++){
            if(boneWeights[i] == 0){
                boneIndices[i] = index;
                boneWeights[i] = weight;
                return;
            }
        }

        //to many weights
        assert(0);
    }
};

template<>
void VertexBuffer<BoneVertex>::setVertexAttributes();


