/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "DatasetViewer.h"

#include "saiga/vision/camera/all.h"


Sample::Sample()
{
    std::cout << "init done" << std::endl;
}


void Sample::update(float dt)
{
    if (cameraType == CameraInputType::RGBD)
    {
        if (!rgbdcamera) return;

        Saiga::RGBDFrameData frameData;
        if (!rgbdcamera->getImage(frameData)) return;
        tg.addTime();

        if (!leftTexture)
        {
            leftImage   = frameData.colorImg;
            leftTexture = std::make_shared<Texture>();
            leftTexture->fromImage(leftImage, false, false);
        }

        if (!rightTexture)
        {
            rightTexture = std::make_shared<Texture>();
            rightImage.create(frameData.depthImg.height, frameData.depthImg.width);
            rightTexture->fromImage(rightImage, false, false);
        }


        leftImage = frameData.colorImg;
        Saiga::ImageTransformation::depthToRGBA(frameData.depthImg, rightImage, 0, 8);

        leftTexture->updateFromImage(leftImage);
        rightTexture->updateFromImage(rightImage);
    }

    if (cameraType == CameraInputType::Stereo)
    {
        if (!stereocamera) return;

        Saiga::StereoFrameData frameData;
        if (!stereocamera->getImage(frameData)) return;
        tg.addTime();


        leftImageGray  = frameData.grayImg;
        rightImageGray = frameData.grayImg2;

        leftImage.create(frameData.grayImg.dimensions());
        rightImage.create(frameData.grayImg2.dimensions());

        Saiga::ImageTransformation::Gray8ToRGBA(leftImageGray, leftImage);
        Saiga::ImageTransformation::Gray8ToRGBA(rightImageGray, rightImage);

        if (!leftTexture)
        {
            leftTexture = std::make_shared<Texture>();
            leftTexture->fromImage(leftImage, false, false);
        }

        if (!rightTexture)
        {
            rightTexture = std::make_shared<Texture>();
            rightTexture->fromImage(rightImage, false, false);
        }


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
    ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("Saiga OpenNI");

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
    dparams.fps        = 25;
    dparams.startFrame = 10;
    dparams.maxFrames  = 10000;

    if (ImGui::Button("Load From File"))
    {
        dparams.dir  = "/home/dari/Projects/snake/code/data/tum/rgbd_dataset_freiburg3_long_office_household/";
        rgbdcamera   = std::make_unique<TumRGBDCamera>(dparams, intr);
        leftTexture  = nullptr;
        rightTexture = nullptr;

        cameraType = TumRGBDCamera::FrameType::cameraType;
    }

    if (ImGui::Button("Load From File Euroc"))
    {
        dparams.dir  = "/home/dari/Projects/snake/code/data/euroc/mav0/";
        stereocamera = std::make_unique<EuRoCDataset>(dparams);
        leftTexture  = nullptr;
        rightTexture = nullptr;

        cameraType = EuRoCDataset::FrameType::cameraType;
    }

    if (ImGui::Button("Load From File Kitti"))
    {
        dparams.dir         = "/home/dari/Projects/snake/code/data/kitti/dataset/sequences/00/";
        dparams.groundTruth = "/home/dari/Projects/snake/code/data/kitti/dataset/poses/00.txt";
        stereocamera        = std::make_unique<KittiDataset>(dparams);
        leftTexture         = nullptr;
        rightTexture        = nullptr;

        cameraType = KittiDataset::FrameType::cameraType;
    }

    if (ImGui::Button("Openni"))
    {
        intr.depthFactor = 1000.0;
        rgbdcamera       = std::make_unique<Saiga::RGBDCameraOpenni>(intr);
        initTexture      = true;
    }

    if (ImGui::Button("Clear"))
    {
        rgbdcamera = nullptr;
    }

    ImGui::Text("Frame: %d", frameId);

    Saiga::RGBDCameraOpenni* cam2 = dynamic_cast<Saiga::RGBDCameraOpenni*>(rgbdcamera.get());
    if (cam2)
    {
        cam2->imgui();
    }

    ImGui::End();
}

int main(const int argc, const char* argv[])
{
    using namespace Saiga;

    {
        Sample example;

        example.run();
    }

    return 0;
}
