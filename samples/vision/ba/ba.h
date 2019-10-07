/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once



#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/vision/recursive/BARecursive.h"
#include "saiga/vision/scene/Scene.h"
#include "saiga/vision/scene/SynteticScene.h"

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
    // bool uploadChanges    = true;
    float rms             = 0;
    int minMatchEdge      = 1000;
    float maxEdgeDistance = 1;
    Saiga::Object3D teapotTrans;

    Saiga::Scene scene;
    Saiga::SynteticScene sscene;

    // bool displayModels = true;
    // bool showImgui     = true;

    Saiga::BAOptions baoptions;
    Saiga::BARec barec;

    std::vector<std::string> datasets;
};
