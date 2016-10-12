#pragma once

#include "saiga/config.h"
#include "saiga/rendering/object3d.h"
#include "saiga/assets/animatedAsset.h"


class SAIGA_GLOBAL AnimatedAssetObject : public Object3D{
private:
    float test = 0;
    animationtime_t animationTotalTime = animationtime_t(0);
    animationtime_t animationTimeAtUpdate = animationtime_t(0);
    animationtime_t animationTimeAtRender = animationtime_t(0);
    int activeAnimation = 0;

    AnimationFrame currentFrame;


    //it's better to have the buffer here instead of in the asset, because otherwise it has to be uploaded for every render call (multiple times per frame)
    UniformBuffer boneMatricesBuffer;

    public:
    void setAnimation(int id);

    void init(AnimatedAsset* _asset);


    void updateAnimation(float dt);
    void interpolateAnimation(float dt, float alpha);

    void render(Camera *cam);
    void renderDepth(Camera *cam);
    void renderWireframe(Camera *cam);
    void renderRaw();

private:
    AnimatedAsset* asset = nullptr;

};
