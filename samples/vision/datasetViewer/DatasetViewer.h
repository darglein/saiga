/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/opengl/world/TextureDisplay.h"
#include "saiga/vision/camera/CameraBase.h"
using namespace Saiga;



class Sample : public SampleWindowForward
{
    using Base = SampleWindowForward;

   public:
    Sample();


    void update(float dt) override;
    void renderOverlay(Camera* cam) override {}
    void renderFinal(Camera* cam) override;

   private:
    std::shared_ptr<Texture> leftTexture, rightTexture;
    TextureDisplay display;


    std::unique_ptr<CameraBase> rgbdcamera;
    std::unique_ptr<CameraBase> monocamera;
    std::unique_ptr<CameraBase> stereocamera;

    RGBAImageType leftImage;
    RGBAImageType rightImage;

    GrayImageType leftImageGray;
    GrayImageType rightImageGray;


    CameraInputType cameraType = CameraInputType::Mono;



    //    char dir[256]    = "recording/";
    char dir[256] = "/ssd2/slam/euroc/MH_01/mav0/";
    //    char dir[256] = "/ssd2/slam/kitti/dataset/sequences/00/";
    //    char dir[256] = "/ssd2/slam/ismar/ismar_test/C1_test/";
    //    char dir[256] = "/ssd2/slam/tum/rgbd_dataset_freiburg3_long_office_household/";
    //    char dir[256] = "/home/dari/Projects/saiga/build/bin/recording/";

    int frameId      = 0;
    bool initTexture = false;


    ImGui::HzTimeGraph tg;
};
