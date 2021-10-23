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
#include "saiga/core/util/Thread/all.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/opengl/world/TextureDisplay.h"
#include "saiga/vision/camera/CameraBase.h"
#include "saiga/vision/camera/all.h"

#include "saiga/core/util/FileSystem.h"
using namespace Saiga;
#include <k4a/k4a.h>



class Sample : public SampleWindowForward
{
    using Base = SampleWindowForward;

   public:
    Sample() { createGlobalThreadPool(8); }


    void update(float dt) override
    {
        if (cameraType == CameraInputType::RGBD)
        {
            if (!rgbdcamera) return;

            Saiga::FrameData frameData;
            if (!rgbdcamera->getImage(frameData)) return;
            tg.addTime();

            if (!frameData.image_rgb.valid() && frameData.image.valid())
            {
                frameData.image_rgb.create(frameData.image.dimensions());
                ImageTransformation::Gray8ToRGBA(frameData.image.getImageView(), frameData.image_rgb.getImageView());
            }
            //        else if (frameData.colorImg.valid() && !frameData.grayImg.valid())
            //        {
            //            frameData.grayImg.create(frameData.colorImg.dimensions());
            //            ImageTransformation::RGBAToGray8(frameData.colorImg.getImageView(),
            //            frameData.grayImg.getImageView());
            //            ImageTransformation::Gray8ToRGBA(frameData.grayImg.getImageView(),
            //            frameData.colorImg.getImageView());
            //        }

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

            if (recording)
            {
                auto str       = Saiga::leadingZeroString(frameId, 5);
                auto frame_dir = frame_out_dir + "/" + str + "/";
                std::filesystem::create_directory(frame_dir);
                globalThreadPool->enqueue([=]() { frameData.Save(frame_dir); });
                frameId++;
            }


            leftImage = frameData.image_rgb;
            //        Saiga::ImageTransformation::depthToRGBA_HSV(frameData.depthImg, rightImage, 0, 5);
            Saiga::ImageTransformation::depthToRGBA(frameData.depth_image, rightImage, 0, 7);

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

            // Base::renderFinal(cam);

            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_FirstUseEver);
            ImGui::Begin("DatasetViewer");

            tg.renderImGui();


            ImGui::InputText("Output Dir", dir, 256);


            if (ImGui::Checkbox("Recording", &recording))
            {
                if (recording)
                {
                    std::string out_dir = dir;
                    frame_out_dir       = out_dir + "/frames/";

                    frameId = 0;
                    std::filesystem::remove_all(out_dir);
                    std::filesystem::create_directory(out_dir);
                    std::filesystem::create_directory(frame_out_dir);

                    auto intr = rgbdcamera->intrinsics();
                    intr.fromConfigFile(out_dir + "/camera.ini");
                }
            }

            static KinectCamera::KinectParams params;

            ImGui::Checkbox("color", &params.color);
            ImGui::Checkbox("narrow_depth", &params.narrow_depth);
            ImGui::InputInt("imu_merge_count", &params.imu_merge_count);
            ImGui::InputInt("fps", &params.fps);


            if (ImGui::Button("Open"))
            {
                rgbdcamera   = nullptr;
                rgbdcamera   = std::make_unique<KinectCamera>(params);
                leftTexture  = nullptr;
                rightTexture = nullptr;
                cameraType   = rgbdcamera->CameraType();
            }

            ImGui::Text("Frame: %d", frameId);


            ImGui::End();
        }
    }

   private:
    std::shared_ptr<Texture> leftTexture, rightTexture;
    TextureDisplay display;


    std::unique_ptr<KinectCamera> rgbdcamera;
    std::unique_ptr<CameraBase> monocamera;
    std::unique_ptr<CameraBase> stereocamera;

    RGBAImageType leftImage;
    RGBAImageType rightImage;

    GrayImageType leftImageGray;
    GrayImageType rightImageGray;


    CameraInputType cameraType = CameraInputType::Mono;



    std::string frame_out_dir;
    char dir[256]  = "recording/";
    bool recording = false;
    int frameId    = 0;


    ImGui::HzTimeGraph tg;
};



int main(const int argc, const char* argv[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample example;

    example.run();


    return 0;
}
