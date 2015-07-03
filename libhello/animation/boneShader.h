#pragma once

#include <libhello/config.h>
#include <libhello/opengl/basic_shaders.h>

#include <libhello/opengl/uniformBuffer.h>

class SAIGA_GLOBAL BoneShader : public MVPShader{
public:
    GLuint location_boneMatrices;
    GLuint location_boneMatricesBlock, binding_boneMatricesBlock;


    BoneShader(const std::string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();

    void uploadBoneMatrices(mat4* matrices, int count);
};



