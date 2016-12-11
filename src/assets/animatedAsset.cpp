#include "saiga/assets/animatedAsset.h"
#include "saiga/animation/boneShader.h"




void AnimatedAsset::render(Camera *cam, const mat4 &model, UniformBuffer& boneMatrices)
{
    BoneShader* bs = static_cast<BoneShader*>(this->shader);


    bs->bind();
    bs->bindCamera(cam);
    bs->uploadModel(model);
//    boneMatrices.bind(0);
    boneMatrices.bind(BONE_MATRICES_BINDING_POINT);
    buffer.bindAndDraw();
    bs->unbind();
}

void AnimatedAsset::renderDepth(Camera *cam, const mat4 &model, UniformBuffer &boneMatrices)
{

    BoneShader* bs = static_cast<BoneShader*>(this->depthshader);

    bs->bind();
    bs->bindCamera(cam);
    bs->uploadModel(model);
    boneMatrices.bind(BONE_MATRICES_BINDING_POINT);
    buffer.bindAndDraw();
    bs->unbind();
}
