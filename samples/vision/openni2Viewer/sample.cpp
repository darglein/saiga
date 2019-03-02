/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "sample.h"

#include "saiga/core/image/imageTransformations.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/color.h"
#include "saiga/core/util/threadName.h"
#include "saiga/core/util/threadPool.h"
#include "saiga/core/util/tostring.h"
#include "saiga/extra/network/RGBDCameraNetwork.h"
#include "saiga/vision/FileRGBDCamera.h"
#include "saiga/vision/openni2/RGBDCameraInput.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), renderer(renderer)
{
    Saiga::createGlobalThreadPool(8);
    Saiga::setThreadName("main");
    cout << "init done" << endl;
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    textureDisplay.init(base, renderer.renderPass);
}



void VulkanExample::update(float dt)
{
    if (!rgbdcamera) return;
    auto newFrameData = rgbdcamera->tryGetImage();

    if (newFrameData)
    {
        frameData     = newFrameData;
        updateTexture = true;
        tg.addTime();

        if (capturing)
        {
            auto str = Saiga::leadingZeroString(frameId, 5);
            auto tmp = frameData;
            Saiga::globalThreadPool->enqueue([=]() {
                tmp->colorImg.save(std::string(dir) + str + ".png");
                tmp->depthImg.save(std::string(dir) + str + ".saigai");
            });
        }
        frameId++;
    }
}

void VulkanExample::transfer(vk::CommandBuffer cmd)
{
    if (!rgbdcamera) return;

    if (initTexture)
    {
        while (!rgbdcamera->isOpened())
        {
            cout << "Waiting for camera..." << endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        frameData = rgbdcamera->waitForImage();


        rgbImage.create(frameData->colorImg.h, frameData->colorImg.w);
        //    Saiga::ImageTransformation::addAlphaChannel(frameData->colorImg.getImageView(),rgbImage.getImageView());

        texture = std::make_shared<Saiga::Vulkan::Texture2D>();
        texture->fromImage(renderer.base(), rgbImage);


        texture2 = std::make_shared<Saiga::Vulkan::Texture2D>();
        //    Saiga::TemplatedImage<ucvec4> depthmg(frameData->depthImg.height,frameData->depthImg.width);
        depthmg.create(frameData->depthImg.height, frameData->depthImg.width);
        Saiga::ImageTransformation::depthToRGBA(frameData->depthImg.getImageView(), depthmg.getImageView(), 0, 7000);
        texture2->fromImage(renderer.base(), depthmg);



        textureDes  = textureDisplay.createAndUpdateDescriptorSet(*texture);
        textureDes2 = textureDisplay.createAndUpdateDescriptorSet(*texture2);

        cout << "init done " << endl;
        initTexture = false;
    }

    if (updateTexture)
    {
        texture->uploadImage(frameData->colorImg, true);
        Saiga::ImageTransformation::depthToRGBA(frameData->depthImg, depthmg, 0, 8);
        texture2->uploadImage(depthmg, true);
        updateTexture = false;
    }
}


void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (!rgbdcamera) return;

    if (textureDisplay.bind(cmd))
    {
        textureDisplay.renderTexture(cmd, textureDes, vec2(0, 0),
                                     vec2(frameData->colorImg.width, frameData->colorImg.height));
        textureDisplay.renderTexture(cmd, textureDes2, vec2(frameData->colorImg.width, 0),
                                     vec2(frameData->depthImg.width, frameData->depthImg.height));
    }
}

void VulkanExample::renderGUI()
{
    parentWindow.renderImGui();


    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiSetCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Saiga OpenNI");

    tg.renderImGui();


    ImGui::InputText("Output Dir", dir, 256);
    if (ImGui::Checkbox("Capture", &capturing))
    {
        frameId = 0;
    }

    if (ImGui::Button("Load From File"))
    {
        rgbdcamera  = std::make_unique<Saiga::FileRGBDCamera>("recording/");
        initTexture = true;
    }

    if (ImGui::Button("Openni"))
    {
        Saiga::RGBDCameraInput::CameraOptions co1, co2;
        //        co2.h = 240;
        //        co2.w = 320;

        rgbdcamera  = std::make_unique<Saiga::RGBDCameraInput>(co1, co2);
        initTexture = true;
    }

    if (ImGui::Button("Clear"))
    {
        rgbdcamera = nullptr;
    }

    ImGui::Text("Frame: %d", frameId);

#if 0
    std::shared_ptr<Saiga::RGBDCameraInput> cam = std::dynamic_pointer_cast<Saiga::RGBDCameraInput>(rgbdcamera);
    cam->updateCameraSettings();


    ImGui::Checkbox("autoexposure", &cam->autoexposure);
    ImGui::Checkbox("autoWhiteBalance", &cam->autoWhiteBalance);

    ImGui::InputInt("exposure", &cam->exposure);
    ImGui::InputInt("gain", &cam->gain);
#endif

    ImGui::End();
}


void VulkanExample::keyPressed(SDL_Keysym key)
{
    switch (key.scancode)
    {
        case SDL_SCANCODE_ESCAPE:
            parentWindow.close();
            break;
        default:
            break;
    }
}

void VulkanExample::keyReleased(SDL_Keysym key) {}
