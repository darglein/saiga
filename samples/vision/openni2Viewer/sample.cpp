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
#include "saiga/core/util/Thread/threadName.h"
#include "saiga/core/util/Thread/threadPool.h"
#include "saiga/core/util/tostring.h"
#include "saiga/extra/network/RGBDCameraNetwork.h"
#include "saiga/vision/camera/all.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), renderer(renderer)
{
    Saiga::createGlobalThreadPool(8);
    Saiga::setThreadName("main");
    std::cout << "init done" << std::endl;

    init(renderer.base());
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    textureDisplay.init(base, renderer.renderPass);
}



void VulkanExample::update(float dt)
{
    if (!rgbdcamera) return;

    Saiga::RGBDFrameData newFrameData;
    auto gotFrame = rgbdcamera->getImage(newFrameData);

    if (gotFrame)
    {
        frameData     = std::move(newFrameData);
        updateTexture = true;
        tg.addTime();

        if (capturing)
        {
            auto str = Saiga::leadingZeroString(frameId, 5);
            auto tmp = frameData;
            Saiga::globalThreadPool->enqueue([=]() {
                tmp.colorImg.save(std::string(dir) + str + ".png");
                tmp.depthImg.save(std::string(dir) + str + ".saigai");
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
            std::cout << "Waiting for camera..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        bool gotImage = rgbdcamera->getImageSync(frameData);

        std::cout << "create image texture: " << frameData.depthImg.height << "x" << frameData.depthImg.width << std::endl;

        rgbImage.create(frameData.colorImg.h, frameData.colorImg.w);
        //    Saiga::ImageTransformation::addAlphaChannel(frameData->colorImg.getImageView(),rgbImage.getImageView());

        texture = std::make_shared<Saiga::Vulkan::Texture2D>();
        texture->fromImage(renderer.base(), rgbImage);


        texture2 = std::make_shared<Saiga::Vulkan::Texture2D>();
        //    Saiga::TemplatedImage<ucvec4> depthmg(frameData->depthImg.height,frameData->depthImg.width);
        depthmg.create(frameData.depthImg.height, frameData.depthImg.width);
        std::cout << frameData.depthImg << std::endl;
        std::cout << depthmg << std::endl;
        Saiga::ImageTransformation::depthToRGBA(frameData.depthImg.getImageView(), depthmg.getImageView(), 0, 7000);
        texture2->fromImage(renderer.base(), depthmg);



        textureDes  = textureDisplay.createAndUpdateDescriptorSet(*texture);
        textureDes2 = textureDisplay.createAndUpdateDescriptorSet(*texture2);

        std::cout << "init done " << std::endl;
        initTexture = false;
    }

    if (updateTexture)
    {
        texture->uploadImage(frameData.colorImg, true);
        Saiga::ImageTransformation::depthToRGBA(frameData.depthImg, depthmg, 0, 8);
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
                                     vec2(frameData.colorImg.width, frameData.colorImg.height));
        textureDisplay.renderTexture(cmd, textureDes2, vec2(frameData.colorImg.width, 0),
                                     vec2(frameData.depthImg.width, frameData.depthImg.height));
    }
}

void VulkanExample::renderGUI()
{
    parentWindow.renderImGui();


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
    intr.deptho.w = depthWidth;
    intr.deptho.h = depthHeight;
    intr.fps      = fps;



    if (ImGui::Checkbox("Capture", &capturing))
    {
        if (capturing)
        {
            std::filesystem::remove_all(std::string(dir));
            std::filesystem::create_directory(std::string(dir));
            intr.fromConfigFile(std::string(dir) + "config.ini");
            frameId = 0;
        }
    }



    if (ImGui::Button("Load From File"))
    {
        intr.fromConfigFile(std::string(dir) + "config.ini");
        rgbdcamera  = std::make_unique<Saiga::FileRGBDCamera>(dir, intr);
        initTexture = true;
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

    Saiga::RGBDCameraOpenni* cam = dynamic_cast<Saiga::RGBDCameraOpenni*>(rgbdcamera.get());
    if (cam)
    {
        cam->imgui();
    }

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
