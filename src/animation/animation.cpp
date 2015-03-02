#include "animation/animation.h"

using std::cout;
using std::endl;

Animation::Animation()
{
}


void Animation::setKeyFrame(float f){
    //get before and after that time
    int frame = floor(f);
    int nextFrame = frame+1;
    float t = f - frame;

    frame = frame%animationFrames.size();
    nextFrame = nextFrame%animationFrames.size();

    //don't interpolate between last and first frame
    if(nextFrame==0){
        frame = 0;
        nextFrame = 1;
    }

    AnimationFrame &k0 = animationFrames[frame];
    AnimationFrame &k1 = animationFrames[nextFrame];

    AnimationFrame::interpolate(k0,k1,t,boneMatrices);

//    for(int m =0;m<boneMatrices.size();++m){
//        boneMatrices[m] = k.boneMatrices[m];
//        cout<<boneMatrices[m]<<endl;
//    }
}


void Animation::setKeyFrame(int i){
    i = i%animationFrames.size();

    AnimationFrame &k = animationFrames[i];


    for(int m =0;m<boneMatrices.size();++m){
        boneMatrices[m] = k.boneMatrices[m];
//        cout<<boneMatrices[m]<<endl;
    }
//    cout<<"========================================"<<endl;
}

void Animation::update(){

    animtick = animtick + (3.0f/1000.0f) * animfps;
    setKeyFrame(animtick);

}
