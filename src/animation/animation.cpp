#include "saiga/animation/animation.h"
#include "saiga/util/assert.h"


int Animation::getActiveFrames()
{
    return skipLastFrame ? frameCount-1 : frameCount;
}

const AnimationFrame &Animation::getKeyFrame(int frameIndex)
{
    assert(frameIndex>=0 && frameIndex<frameCount);
    return keyFrames[frameIndex];
}


void Animation::getFrame2(float time, AnimationFrame &out){

#if 0
    //get frame index before and after
    int frame = floor(time);
    int nextFrame = frame+1;
    float t = time - frame;

    int modulo = getActiveFrames();

    frame = frame%modulo;
    nextFrame = nextFrame%modulo;


    getFrame(frame,nextFrame,t,out);
#else
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

#endif
}



void Animation::getFrameNormalized(float time, AnimationFrame &out)
{
    time = glm::clamp(time,0.0f,1.0f);
    assert(time >= 0 && time <= 1);
    getFrame2(duration*time,out);
}

void Animation::getFrame(int frame0, int frame1, float alpha, AnimationFrame &out)
{
    assert(frame0>=0 && frame0<frameCount);
    assert(frame1>=0 && frame1<frameCount);

    AnimationFrame &k0 = keyFrames[frame0];
    AnimationFrame &k1 = keyFrames[frame1];
    AnimationFrame::interpolate(k0,k1,out,alpha);
}

void Animation::print()
{
    cout << "[Animation] " + name << " Frames="<<frameCount << " tickspersecond="<<ticksPerSecond << " duration="<<duration<< endl;
    cout << "\tKeyframes: [";
    for(AnimationFrame& af : keyFrames){
        cout << af.time << ", ";
    }
    cout << "]" << endl;
}


