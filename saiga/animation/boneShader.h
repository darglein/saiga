#pragma once

#include <saiga/config.h>
#include <saiga/opengl/shader/basic_shaders.h>

#include <saiga/opengl/uniformBuffer.h>

class SAIGA_GLOBAL BoneShader : public MVPShader{
public:
    GLint location_boneMatrices;
    GLint location_boneMatricesBlock, binding_boneMatricesBlock;


    virtual void checkUniforms();

    void uploadBoneMatrices(mat4* matrices, int count);
};



