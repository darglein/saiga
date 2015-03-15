#pragma once

#include <libhello/opengl/basic_shaders.h>



class BoneShader : public MVPShader{
public:
    GLuint location_boneMatrices;
    BoneShader(const string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();

    void uploadBoneMatrices(mat4* matrices, int count);
};



