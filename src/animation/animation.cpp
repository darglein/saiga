#include "animation/animation.h"

using std::cout;
using std::endl;

Animation::Animation()
{
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
