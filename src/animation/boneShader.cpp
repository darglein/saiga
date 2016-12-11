#include "saiga/animation/boneShader.h"

void BoneShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_boneMatrices = getUniformLocation("boneMatrices");
    location_boneMatricesBlock = getUniformBlockLocation("boneMatricesBlock");

    binding_boneMatricesBlock = 0;
    setUniformBlockBinding(location_boneMatricesBlock,binding_boneMatricesBlock);
//    cout<<"uniform block: "<<location_boneMatricesBlock<<endl;

//        GLint ret;

//        glGetActiveUniformBlockiv(program,location_boneMatricesBlock,GL_UNIFORM_BLOCK_DATA_SIZE,&ret);
//        cout<<"GL_UNIFORM_BLOCK_DATA_SIZE "<<ret<<endl;
//        glGetActiveUniformBlockiv(program,location_boneMatricesBlock,GL_UNIFORM_BLOCK_NAME_LENGTH,&ret);
//        cout<<"GL_UNIFORM_BLOCK_NAME_LENGTH "<<ret<<endl;
//        glGetActiveUniformBlockiv(program,location_boneMatricesBlock,GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS,&ret);
//        cout<<"GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS "<<ret<<endl;

//    //    std::vector<GLint> indices(ret);
//    //    glGetActiveUniformBlockiv(program,location_boneMatricesBlock,GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES,&indices[0]);
//    ////    cout<<"GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES "<<ret<<endl;
//    //    for(GLint i : indices){
//    //        cout<<i<<endl;
//    //    }

//    std::vector<GLint> indices = getUniformBlockIndices(location_boneMatricesBlock);
//    cout<<"Uniform block indices: "<<indices.size()<<endl;
//    for(GLint i : indices){
//        cout<<i<<endl;
//    }

//    std::vector<GLint> data = getUniformBlockSize(location_boneMatricesBlock,indices);
//    cout<<"Uniform block size: "<<data.size()<<endl;
//    for(GLint i : data){
//        cout<<i<<endl;
//    }

//    getUniformInfo(location_boneMatrices);
//    getUniformInfo(indices[0]);

//    test.init(this,location_boneMatricesBlock);

//    cout<<test<<endl;

}

void BoneShader::uploadBoneMatrices(mat4 *matrices, int count)
{
    Shader::upload(location_boneMatrices,count,matrices);
}


