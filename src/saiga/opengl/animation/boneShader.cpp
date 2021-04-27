/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_OPENGL
#    include "saiga/opengl/animation/boneShader.h"

namespace Saiga
{
void BoneShader::checkUniforms()
{
    MVPShader::checkUniforms();
    location_boneMatrices      = getUniformLocation("boneMatrices");
    location_boneMatricesBlock = getUniformBlockLocation("boneMatricesBlock");

    setUniformBlockBinding(location_boneMatricesBlock, BONE_MATRICES_BINDING_POINT);
    //    std::cout<<"uniform block: "<<location_boneMatricesBlock<<endl;

    //        GLint ret;

    //        glGetActiveUniformBlockiv(program,location_boneMatricesBlock,GL_UNIFORM_BLOCK_DATA_SIZE,&ret);
    //        std::cout<<"GL_UNIFORM_BLOCK_DATA_SIZE "<<ret<<endl;
    //        glGetActiveUniformBlockiv(program,location_boneMatricesBlock,GL_UNIFORM_BLOCK_NAME_LENGTH,&ret);
    //        std::cout<<"GL_UNIFORM_BLOCK_NAME_LENGTH "<<ret<<endl;
    //        glGetActiveUniformBlockiv(program,location_boneMatricesBlock,GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS,&ret);
    //        std::cout<<"GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS "<<ret<<endl;

    //    //    std::vector<GLint> indices(ret);
    //    //
    //    glGetActiveUniformBlockiv(program,location_boneMatricesBlock,GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES,&indices[0]);
    //    ////    std::cout<<"GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES "<<ret<<endl;
    //    //    for(GLint i : indices){
    //    //        std::cout<<i<<endl;
    //    //    }

    //    std::vector<GLint> indices = getUniformBlockIndices(location_boneMatricesBlock);
    //    std::cout<<"Uniform block indices: "<<indices.size()<<endl;
    //    for(GLint i : indices){
    //        std::cout<<i<<endl;
    //    }

    //    std::vector<GLint> data = getUniformBlockSize(location_boneMatricesBlock,indices);
    //    std::cout<<"Uniform block size: "<<data.size()<<endl;
    //    for(GLint i : data){
    //        std::cout<<i<<endl;
    //    }

    //    getUniformInfo(location_boneMatrices);
    //    getUniformInfo(indices[0]);

    //    test.init(this,location_boneMatricesBlock);

    //    std::cout<<test<<endl;
}

void BoneShader::uploadBoneMatrices(mat4* matrices, int count)
{
    Shader::upload(location_boneMatrices, count, matrices);
    SAIGA_ASSERT(0);
}

}  // namespace Saiga
#endif
