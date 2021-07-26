/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */



#include "saiga/core/imgui/imgui.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/opengl/world/TextureDisplay.h"
#include "saiga/vision/camera/CameraBase.h"
using namespace Saiga;

#include "saiga/vision/camera/all.h"



class Sample : public SampleWindowForward
{
    using Base = SampleWindowForward;

   public:
    Sample() {}


    void update(float dt) override
    {
        if (cameraType == CameraInputType::RGBD)
        {
            if (!rgbdcamera) return;

            Saiga::FrameData frameData;
            if (!rgbdcamera->getImage(frameData)) return;
            tg.addTime();

            if (!leftTexture)
            {
                leftImage   = frameData.image_rgb;
                leftTexture = std::make_shared<Texture>(leftImage);
            }

            if (!rightTexture)
            {
                rightImage.create(frameData.depth_image.height, frameData.depth_image.width);
                rightTexture = std::make_shared<Texture>(rightImage);
            }


            leftImage = frameData.image_rgb;
            Saiga::ImageTransformation::depthToRGBA(frameData.depth_image, rightImage, 0, 8);

            leftTexture->updateFromImage(leftImage);
            rightTexture->updateFromImage(rightImage);
        }

        if (cameraType == CameraInputType::Stereo)
        {
            if (!stereocamera) return;

            Saiga::FrameData frameData;
            if (!stereocamera->getImage(frameData)) return;
            tg.addTime();


            leftImageGray  = frameData.image;
            rightImageGray = frameData.right_image;

            leftImage.create(frameData.image.dimensions());
            rightImage.create(frameData.right_image.dimensions());

            Saiga::ImageTransformation::Gray8ToRGBA(leftImageGray, leftImage);
            Saiga::ImageTransformation::Gray8ToRGBA(rightImageGray, rightImage);

            if (!leftTexture)
            {
                leftTexture = std::make_shared<Texture>(leftImage, true, false);
            }

            if (!rightTexture)
            {
                rightTexture = std::make_shared<Texture>(rightImage, true, false);
            }


            leftTexture->updateFromImage(leftImage);
            rightTexture->updateFromImage(rightImage);
        }
    }

    void render(RenderInfo render_info) override
    {
        if (render_pass == RenderPass::GUI)
        {
            if (leftImage.valid() && leftTexture)
            {
                display.render(leftTexture.get(), {0, 0}, {leftImage.w, leftImage.h}, true);
            }
            if (rightImage.valid() && rightTexture)
            {
                display.render(rightTexture.get(), {leftImage.w, 0}, {rightImage.w, rightImage.h}, true);
            }

            //            Base::renderFinal(cam);

            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_FirstUseEver);
            ImGui::Begin("DatasetViewer");

            tg.renderImGui();


            ImGui::InputText("Output Dir", dir, 256);

            static int depthWidth  = 320;
            static int depthHeight = 240;
            static int fps         = 30;

            ImGui::InputInt("depthWidth", &depthWidth);
            ImGui::InputInt("depthHeight", &depthHeight);
            ImGui::InputInt("fps", &fps);

            Saiga::RGBDIntrinsics intr;
            //    intr.depthImageSize.w = depthWidth;
            //    intr.depthImageSize.h = depthHeight;
            intr.fps = fps;


            DatasetParameters dparams;
            dparams.playback_fps      = 25;
            dparams.startFrame        = 10;
            dparams.maxFrames         = 100;
            dparams.multiThreadedLoad = true;
            dparams.preload           = true;


            if (ImGui::Button("Load From File Scannet"))
            {
                dparams.dir  = dir;
                rgbdcamera   = std::make_unique<ScannetDataset>(dparams);
                leftTexture  = nullptr;
                rightTexture = nullptr;

                cameraType = rgbdcamera->CameraType();
            }

            if (ImGui::Button("Load From File Saiga"))
            {
                dparams.dir  = dir;
                rgbdcamera   = std::make_unique<SaigaDataset>(dparams);
                leftTexture  = nullptr;
                rightTexture = nullptr;

                cameraType = rgbdcamera->CameraType();
            }


#ifdef SAIGA_USE_YAML_CPP
            if (ImGui::Button("Load From File TUM RGBD"))
            {
                dparams.dir  = dir;
                rgbdcamera   = std::make_unique<TumRGBDDataset>(dparams);
                leftTexture  = nullptr;
                rightTexture = nullptr;

                cameraType = rgbdcamera->CameraType();
            }


            if (ImGui::Button("Load From File Euroc"))
            {
                dparams.dir  = dir;
                stereocamera = std::make_unique<EuRoCDataset>(dparams);
                leftTexture  = nullptr;
                rightTexture = nullptr;
                cameraType   = stereocamera->CameraType();
            }
#endif

            if (ImGui::Button("Load From File Kitti"))
            {
                dparams.dir = dir;
                //        dparams.groundTruth = dir;
                stereocamera = std::make_unique<KittiDataset>(dparams);
                leftTexture  = nullptr;
                rightTexture = nullptr;
                cameraType   = stereocamera->CameraType();
            }

#ifdef SAIGA_USE_OPENNI2
            if (ImGui::Button("Openni"))
            {
                intr.depthFactor = 1000.0;
                rgbdcamera       = std::make_unique<Saiga::RGBDCameraOpenni>(intr);
                initTexture      = true;
            }
            Saiga::RGBDCameraOpenni* cam2 = dynamic_cast<Saiga::RGBDCameraOpenni*>(rgbdcamera.get());
            if (cam2)
            {
                cam2->imgui();
            }
#endif

            if (ImGui::Button("Clear"))
            {
                rgbdcamera = nullptr;
            }

            ImGui::Text("Frame: %d", frameId);


            ImGui::End();
        }
    }

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

    int frameId = 0;
    // bool initTexture = false;


    ImGui::HzTimeGraph tg;
};

int main(const int argc, const char* argv[])
{
    using namespace Saiga;

    {
        Sample example;

        example.run();
    }

    return 0;
}
