#include "saiga/assets/animatedAsset.h"
#include "saiga/animation/boneShader.h"




void AnimatedAsset::render(Camera *cam, const mat4 &model, UniformBuffer& boneMatrices)
{
    BoneShader* bs = static_cast<BoneShader*>(this->shader);


    bs->bind();
    bs->uploadAll(model,cam->view,cam->proj);
    boneMatrices.bind(bs->binding_boneMatricesBlock);
    buffer.bindAndDraw();
    bs->unbind();
}

void AnimatedAsset::renderDepth(Camera *cam, const mat4 &model, UniformBuffer &boneMatrices)
{

    BoneShader* bs = static_cast<BoneShader*>(this->depthshader);

    bs->bind();
    bs->uploadAll(model,cam->view,cam->proj);
    boneMatrices.bind(bs->binding_boneMatricesBlock);
    buffer.bindAndDraw();
    bs->unbind();
}
