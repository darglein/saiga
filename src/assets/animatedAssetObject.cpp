#include "saiga/assets/animatedAssetObject.h"

#include "saiga/assets/asset.h"
#include "saiga/camera/camera.h"

#include "saiga/animation/boneShader.h"

void AnimatedAssetObject::init(AnimatedAsset *_asset)
{
    this->asset = _asset;

    BoneShader* bs = static_cast<BoneShader*>(asset->shader);
    boneMatricesBuffer.init(bs,bs->location_boneMatricesBlock);
}

void AnimatedAssetObject::updateAnimation(float dt)
{
    animationTimeAtUpdate += dt / animationTotalTime;
    //loop animation constantly
    if(animationTimeAtUpdate >= 1)
        animationTimeAtUpdate -= 1;
}

void AnimatedAssetObject::interpolateAnimation(float dt, float alpha)
{
    animationTimeAtRender = animationTimeAtUpdate + dt * alpha / animationTotalTime;

    asset->animations[activeAnimation].getFrameNormalized(animationTimeAtRender,currentFrame);//Note: 5% CPU Time

    currentFrame.calculateFromTree();

    boneMatricesBuffer.updateBuffer(currentFrame.boneMatrices.data(),currentFrame.boneMatrices.size()*sizeof(mat4),0);
}


void AnimatedAssetObject::render(Camera *cam)
{


    asset->render(cam,model,boneMatricesBuffer);
}

void AnimatedAssetObject::renderDepth(Camera *cam)
{
    asset->renderDepth(cam,model,boneMatricesBuffer);
}

void AnimatedAssetObject::renderWireframe(Camera *cam)
{
    asset->renderWireframe(cam,model);
}

void AnimatedAssetObject::renderRaw()
{
    asset->renderRaw();
}

