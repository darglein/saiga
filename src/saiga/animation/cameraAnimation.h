/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/util/math.h"
#include <saiga/time/time.h>
#include "saiga/assets/assetLoader.h"
#include "saiga/animation/bspline.h"

namespace Saiga {


class SAIGA_GLOBAL Interpolation
{
public:
    struct Keyframe
    {
        quat rot;
        vec3 position;
    };

//    Keyframe interpolate(const Keyframe& f1, const Keyframe& f2, float alpha);

//    Keyframe interpolate(
//            const Keyframe& f1, const Keyframe& f2,
//            const Keyframe& f3, const Keyframe& f4, float alpha);

    std::vector<Keyframe> keyframes;

    void addKeyframe(const quat& R, const vec3& t)
    {
        keyframes.push_back( {R,t} );
    }

//    Keyframe get(double time);
    Keyframe getNormalized(double time);



    bool cubicInterpolation = true;
    int totalTicks = 0;
    int tick = 1;
    float dt = 1/60.0;
    float totalTime = 5;
    int selectedKeyframe = 0;

    Bspline<vec3> positionSpline;
    Bspline<quat> orientationSpline;

    Interpolation() {}


    void start(Camera& cam, float totalTimeS, float dt);
    bool update(Camera& cam);


    void updateCurve();
    bool isRunning() { return tick <= totalTicks; }

    void render();
    void renderGui(Camera& cam);

    // Camera path mesh
    bool visible = true;
    int subSamples = 5;
    float keyframeScale = 0.5;
    std::shared_ptr<Asset> cameraPathAsset;
private:
    void createAsset();
};



}
