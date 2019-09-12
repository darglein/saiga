/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once


#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/vision/recursive/PGORecursive.h"

using namespace Saiga;

class Sample : public SampleWindowForward
{
    using Base = SampleWindowForward;

   public:
    Sample();



    void update(float dt) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;

   private:
    // render objects
    GLPointCloud pointCloud;
    LineSoup lineSoup;
    LineVertexColoredAsset frustum;


    std::vector<vec3> boxOffsets;
    bool change = false;
    //    bool uploadChanges = true;
    float rms  = 0;
    float chi2 = 0;
    Saiga::Object3D teapotTrans;


    //    bool displayModels = true;

    Saiga::PoseGraph scene;
    Saiga::OptimizationOptions baoptions;

    std::vector<std::string> datasets, baldatasets;
};
