#include "libhello/animation/animation.h"

using std::cout;
using std::endl;

Animation::Animation()
{
}


void Animation::getFrame(float f, AnimationFrame &out){


    //get before and after that time
    int frame = floor(f);
    int nextFrame = frame+1;
    float t = f - frame;

    frame = frame%animationFrames.size();
    nextFrame = nextFrame%animationFrames.size();



    AnimationFrame &k0 = animationFrames[frame];
    AnimationFrame &k1 = animationFrames[nextFrame];


    AnimationFrame::interpolate(k0,k1,out,t);

}


void Animation::setKeyFrame(float f){
    f = f - floor(f);
    f = f*(frameCount-1);

    //get before and after that time
    int frame = floor(f);
    int nextFrame = frame+1;
    float t = f - frame;

    frame = frame%animationFrames.size();
    nextFrame = nextFrame%animationFrames.size();



    AnimationFrame &k0 = animationFrames[frame];
    AnimationFrame &k1 = animationFrames[nextFrame];

    AnimationFrame out;

    AnimationFrame::interpolate(k0,k1,out,t);

    for(unsigned int m =0;m<out.boneMatrices.size();++m){
        boneMatrices[m] = out.boneMatrices[m];
    }
}


void Animation::setKeyFrame(int i){
    i = i%animationFrames.size();

    AnimationFrame &k = animationFrames[i];


    for(unsigned int m =0;m<boneMatrices.size();++m){
        boneMatrices[m] = k.boneMatrices[m];
//        cout<<boneMatrices[m]<<endl;
    }
//    cout<<"========================================"<<endl;
}

void Animation::update(){

    animtick = animtick + (3.0f/1000.0f) * animfps;
    setKeyFrame(animtick);

}
