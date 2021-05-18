/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/geometry/object3d.h"
#include "saiga/opengl/assets/animatedAsset.h"

namespace Saiga
{
class SAIGA_OPENGL_API AnimatedAssetObject : public Object3D
{
   private:
    animationtime_t animationTotalTime    = animationtime_t(0);
    animationtime_t animationTimeAtUpdate = animationtime_t(0);
    animationtime_t animationTimeAtRender = animationtime_t(0);
    int activeAnimation                   = 0;

    AnimationKeyframe currentFrame;


    // it's better to have the buffer here instead of in the asset, because otherwise it has to be uploaded for every
    // render call (multiple times per frame)
    UniformBuffer boneMatricesBuffer;

   public:
    void setAnimation(int id);

    void init(std::shared_ptr<AnimatedAsset> _asset);


    void updateAnimation(float dt);
    void interpolateAnimation(float dt, float alpha);

    void render(Camera* cam);
    void renderDepth(Camera* cam);
    void renderWireframe(Camera* cam);
    void renderRaw();

   private:
    std::shared_ptr<AnimatedAsset> asset = nullptr;
};

}  // namespace Saiga
