/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "KinectAzureViewer.h"

#include "saiga/core/util/Thread/all.h"

#include <filesystem>



Sample::Sample()
{
    createGlobalThreadPool(8);
}


void Sample::update(float dt)
{
    if (cameraType == CameraInputType::RGBD)
    {
        if (!rgbdcamera) return;

        Saiga::RGBDFrameData frameData;
        if (!rgbdcamera->getImage(frameData)) return;
        tg.addTime();

        if (!frameData.colorImg.valid() && frameData.grayImg.valid())
        {
            frameData.colorImg.create(frameData.grayImg.dimensions());
            ImageTransformation::Gray8ToRGBA(frameData.grayImg.getImageView(), frameData.colorImg.getImageView());
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
            leftImage   = frameData.colorImg;
            leftTexture = std::make_shared<Texture>();
            leftTexture->fromImage(leftImage, true, false);
        }

        if (!rightTexture)
        {
            rightTexture = std::make_shared<Texture>();
            rightImage.create(frameData.depthImg.height, frameData.depthImg.width);
            rightTexture->fromImage(rightImage, true, false);
        }

        if (recording)
        {
            auto str       = Saiga::leadingZeroString(frameId, 5);
            auto frame_dir = frame_out_dir + "/" + str + "/";
            std::filesystem::create_directory(frame_dir);
            globalThreadPool->enqueue([=]() { frameData.Save(frame_dir); });
            frameId++;
        }


        leftImage = frameData.colorImg;
        //        Saiga::ImageTransformation::depthToRGBA_HSV(frameData.depthImg, rightImage, 0, 5);
        Saiga::ImageTransformation::depthToRGBA(frameData.depthImg, rightImage, 0, 7);

        leftTexture->updateFromImage(leftImage);
        rightTexture->updateFromImage(rightImage);
    }
}

void Sample::renderFinal(Camera* cam)
{
    if (leftImage.valid() && leftTexture)
    {
        display.render(leftTexture.get(), {0, 0}, {leftImage.w, leftImage.h}, true);
    }
    if (rightImage.valid() && rightTexture)
    {
        display.render(rightTexture.get(), {leftImage.w, 0}, {rightImage.w, rightImage.h}, true);
    }

    Base::renderFinal(cam);

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
        cameraType   = KinectCamera::FrameType::cameraType;
    }

    ImGui::Text("Frame: %d", frameId);


    ImGui::End();
}

int main(const int argc, const char* argv[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample example;

    example.run();


    return 0;
}
