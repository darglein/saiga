#include "saiga/assets/animatedAsset.h"
#include "saiga/animation/boneShader.h"



void AnimatedAsset::render(Camera *cam, const mat4 &model)
{
    AnimationFrame af;
    Animation &a = animations[0];
    std::vector<mat4> boneMatrices;// = a.restPosition.boneMatrices;
//    cout<<"bonematrices "<<boneMatrices.size()<<endl;

//    a.getFrame(test,af);
    a.getFrameNormalized(glm::fract(test),af);
    af.calculateFromTree();
//	boneMatrices = a.getKeyFrame(glm::fract(test)*5).boneMatrices;
    boneMatrices = af.boneMatrices;

//    cout<<"bonematrices "<<boneMatrices.size()<<endl;

    test += 0.01f;


    BoneShader* bshader = static_cast<BoneShader*>(this->shader);
	bshader->bind();

//    std::vector<mat4> boneMatrices(boneCount);
    boneMatricesBuffer.updateBuffer(boneMatrices.data(),boneMatrices.size()*sizeof(mat4),0);
    boneMatricesBuffer.bind(bshader->binding_boneMatricesBlock);

	bshader->uploadAll(model,cam->view,cam->proj);
    buffer.bindAndDraw();
	bshader->unbind();
}

void AnimatedAsset::renderDepth(Camera *cam, const mat4 &model)
{
    BoneShader* bdepthshader = static_cast<BoneShader*>(this->depthshader);
	bdepthshader->bind();
	bdepthshader->uploadAll(model,cam->view,cam->proj);
    buffer.bindAndDraw();
	bdepthshader->unbind();
}
