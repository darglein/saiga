#include "saiga/assets/animatedAsset.h"
#include "saiga/animation/boneShader.h"




void AnimatedAsset::render(Camera *cam, const mat4 &model, UniformBuffer& boneMatrices)
{
    BoneShader* bs = static_cast<BoneShader*>(this->shader);

    BoneShader* bshader = static_cast<BoneShader*>(this->shader);
	bshader->bind();

//    std::vector<mat4> boneMatrices(boneCount);
    boneMatricesBuffer.updateBuffer(boneMatrices.data(),boneMatrices.size()*sizeof(mat4),0);
    boneMatricesBuffer.bind(bshader->binding_boneMatricesBlock);

	bshader->uploadAll(model,cam->view,cam->proj);
    buffer.bindAndDraw();
	bshader->unbind();

}

void AnimatedAsset::renderDepth(Camera *cam, const mat4 &model, UniformBuffer &boneMatrices)
{
    BoneShader* bdepthshader = static_cast<BoneShader*>(this->depthshader);
	bdepthshader->bind();
	bdepthshader->uploadAll(model,cam->view,cam->proj);
    buffer.bindAndDraw();
	bdepthshader->unbind();

}
