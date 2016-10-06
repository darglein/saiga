#include "saiga/animation/boneVertex.h"
#include "saiga/util/assert.h"


BoneVertex::BoneVertex(){
    for (int i = 0; i < MAX_BONES_PER_VERTEX; ++i){
        boneIndices[i] = 0;
        boneWeights[i] = 0;
    }
}

void BoneVertex::addBone(int32_t index, float weight){
    for(int i=0;i<MAX_BONES_PER_VERTEX;i++){
        if(boneWeights[i] == 0){
            boneIndices[i] = index;
            boneWeights[i] = weight;
            return;
        }
    }

    //to many weights
    assert(0);
}

void BoneVertex::apply(const std::vector<mat4> &boneMatrices){
    mat4 boneMatrix(0.0f);
    for(int i=0;i<MAX_BONES_PER_VERTEX;++i){
        int index = (int)boneIndices[i];
        assert(index>=0 && index<(int)boneMatrices.size());
        boneMatrix += boneMatrices[index] * boneWeights[i];
    }

    position = vec3(boneMatrix*vec4(position,1));
    normal = vec3(boneMatrix*vec4(normal,0));
    normal = glm::normalize(normal);

}

void BoneVertex::normalizeWeights()
{
    float weightSum = 0;
    for(int i=0;i<MAX_BONES_PER_VERTEX;++i){
        weightSum += boneWeights[i];
    }

    for(int i=0;i<MAX_BONES_PER_VERTEX;++i){
        boneWeights[i] = boneWeights[i] / weightSum;
    }

}

int BoneVertex::activeBones(){
    int count = 0;
    for(int i=0;i<MAX_BONES_PER_VERTEX;++i){
        if(boneWeights[i]>0){
            count++;
        }
    }
    return count;
}

template<>
void VertexBuffer<BoneVertexCD>::setVertexAttributes(){
    glEnableVertexAttribArray( 0 );
    glEnableVertexAttribArray( 1 );
    glEnableVertexAttribArray( 2 );
    glEnableVertexAttribArray( 3 );
    glEnableVertexAttribArray( 4 );
    glEnableVertexAttribArray( 5 );


    //bone indices + weights
    glVertexAttribIPointer(0,4, GL_INT, sizeof(BoneVertexCD), (void*) (0 * sizeof(GLfloat)) );
    glVertexAttribPointer(1,4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*) (4 * sizeof(GLfloat)) );

    //position normal
    glVertexAttribPointer(2,3, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*) (8 * sizeof(GLfloat))  );
    glVertexAttribPointer(3,3, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*) (11 * sizeof(GLfloat)) );

    //color data
    glVertexAttribPointer(4,3, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*) (14 * sizeof(GLfloat)) );
    glVertexAttribPointer(5,3, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*) (17 * sizeof(GLfloat)) );

}

