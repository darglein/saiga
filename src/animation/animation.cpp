#include "animation/animation.h"


Animation::Animation()
{
}




void Animation::setKeyFrame(int i){
    i = i%animationFrames.size();

    AnimationFrame &k = animationFrames[i];


    for(int m =0;m<boneMatrices.size();++m){
        boneMatrices[m] = k.boneMatrices[m];
    }
}

void Animation::update(){

    animtick = animtick + (30.0f/1000.0f) * animfps;
    setKeyFrame((int)animtick);

}
