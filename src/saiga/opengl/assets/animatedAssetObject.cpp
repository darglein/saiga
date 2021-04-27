/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "animatedAssetObject.h"

#include "saiga/core/camera/camera.h"
#include "saiga/opengl/animation/boneShader.h"
#include "saiga/opengl/assets/asset.h"

namespace Saiga
{
#if 0
void AnimatedAssetObject::setAnimation(int id)
{
    SAIGA_ASSERT(id >= 0 && id < (int)asset->animations.size());
    activeAnimation    = id;
    animationTotalTime = asset->animations[id].duration;
}

void AnimatedAssetObject::init(std::shared_ptr<AnimatedAsset> _asset)
{
    SAIGA_ASSERT(_asset);
    this->asset                    = _asset;
    std::shared_ptr<BoneShader> bs = std::static_pointer_cast<BoneShader>(asset->deferredShader);

    boneMatricesBuffer.init(bs, bs->location_boneMatricesBlock);

    setAnimation(0);
}

void AnimatedAssetObject::updateAnimation(float dt)
{
    animationTimeAtUpdate += animationtime_t(dt);
    // loop animation constantly
    if (animationTimeAtUpdate >= animationTotalTime) animationTimeAtUpdate -= animationTotalTime;
}

void AnimatedAssetObject::interpolateAnimation(float dt, float alpha)
{
    animationTimeAtRender = animationTimeAtUpdate + animationtime_t(dt * alpha);
    if (animationTimeAtRender >= animationTotalTime) animationTimeAtRender -= animationTotalTime;
    asset->animations[activeAnimation].getFrame(animationTimeAtRender, currentFrame);
    boneMatricesBuffer.updateBuffer(
        currentFrame.getBoneMatrices(asset->animations[activeAnimation]).data(),
        currentFrame.getBoneMatrices(asset->animations[activeAnimation]).size() * sizeof(mat4), 0);
}


void AnimatedAssetObject::render(Camera* cam)
{
    asset->render(cam, model, boneMatricesBuffer);
}

void AnimatedAssetObject::renderDepth(Camera* cam)
{
    asset->renderDepth(cam, model, boneMatricesBuffer);
}

void AnimatedAssetObject::renderWireframe(Camera* cam)
{
    asset->renderWireframe(cam, model);
}

void AnimatedAssetObject::renderRaw()
{
    asset->renderRaw();
}
#endif
}  // namespace Saiga
