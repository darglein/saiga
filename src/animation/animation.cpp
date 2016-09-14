#include "saiga/animation/animation.h"
#include "saiga/util/assert.h"



const AnimationFrame &Animation::getKeyFrame(int frameIndex)
{
    assert(frameIndex>=0 && frameIndex<frameCount);
    return keyFrames[frameIndex];
}

void Animation::getFrame(float time, AnimationFrame &out){


    //here time is given in animation time base
    time = glm::clamp(time,0.0f,duration);
//    cout << "time " << time << " " << duration << endl;
    assert(time >= 0 && time <= duration);
    //seach for correct frame interval
    int frame = 0;
    for(AnimationFrame& af : keyFrames){
        if(af.time >= time){
            break;
        }
        frame++;
    }
    int prevFrame = glm::max(0,frame - 1);

    AnimationFrame &k0 = keyFrames[prevFrame];
    AnimationFrame &k1 = keyFrames[frame];

    if(frame == prevFrame){
        out = k0;
        return;
    }

    float alpha = (time - k0.time) / (k1.time - k0.time);
    AnimationFrame::interpolate(k0,k1,out,alpha);

}

void Animation::getFrameNormalized(float time, AnimationFrame &out)
{
    time = glm::clamp(time,0.0f,1.0f);
    assert(time >= 0 && time <= 1);
    getFrame(duration*time,out);
}

void Animation::print()
{
    cout << "[Animation] " + name << " Frames="<<frameCount  << " duration="<<duration<<"s"<< endl;
    cout << "\tKeyframes: [";
    for(AnimationFrame& af : keyFrames){
        cout << af.time << ", ";
    }
    cout << "]" << endl;
}


