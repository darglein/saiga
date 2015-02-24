#include "animation/boneShader.h"

void BoneShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_boneMatrices = getUniformLocation("boneMatrices");
}

void BoneShader::uploadBoneMatrices(mat4 *matrices, int count)
{
    Shader::upload(location_boneMatrices,count,matrices);
}


