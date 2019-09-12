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
#include "saiga/vision/camera/RGBDCamera.h"
using namespace Saiga;



class Sample : public SampleWindowForward
{
    using Base = SampleWindowForward;

   public:
    Sample();


    void update(float dt) override;
    void renderFinal(Camera* cam) override;

   private:
    std::shared_ptr<Texture> texture, texture2;
    TextureDisplay display;

    Saiga::RGBDFrameData frameData;
    std::unique_ptr<Saiga::RGBDCamera> rgbdcamera;

    Saiga::TemplatedImage<ucvec4> rgbImage;
    Saiga::TemplatedImage<ucvec4> depthmg;



    char dir[256]      = "recording/";
    bool capturing     = false;
    int frameId        = 0;
    bool updateTexture = false;
    bool initTexture   = false;


    ImGui::HzTimeGraph tg;
};
