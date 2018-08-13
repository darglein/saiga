/*
* Vulkan Example - imGui (https://github.com/ocornut/imgui)
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "sample.h"
#include <saiga/imgui/imgui.h>
#include "saiga/util/color.h"
#include "saiga/image/imageTransformations.h"
#include "saiga/network/RGBDCameraNetwork.h"
#include "saiga/openni2/RGBDCameraInput.h"



#if defined(SAIGA_OPENGL_INCLUDED)
#error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow &window, Saiga::Vulkan::VulkanForwardRenderer &renderer)
    :  Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), renderer(renderer)
{
    //    rgbdcamera.open();


    std::string file = "server.ini";
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());
    auto ip         = ini.GetAddString ("server","ip","10.0.0.2");
    auto port        = ini.GetAddLong ("server","port",9000);
    if(ini.changed()) ini.SaveFile(file.c_str());




#if 0
    auto cam = std::make_shared<Saiga::RGBDCameraNetwork>();
    cam->connect(ip,port);
    rgbdcamera = cam;
#else
    auto cam = std::make_shared<Saiga::RGBDCameraInput>();
    Saiga::RGBDCameraInput::CameraOptions co1,co2;
    co2.h = 240;
    co2.w = 320;
    cam->open(co1,co2);
    rgbdcamera = cam;
#endif

    frameData = cam->makeFrameData();

    cout << "init done" << endl;
    //    it =
    //    it->makeReciever();
}

VulkanExample::~VulkanExample()
{

}

void VulkanExample::init(Saiga::Vulkan::VulkanBase &base)
{



    textureDisplay.init(base,renderer.renderPass);



    rgbdcamera->readFrame(*frameData);

    texture = std::make_shared<Saiga::Vulkan::Texture2D>();
    texture->fromImage(renderer.base,frameData->colorImg);


    texture2 = std::make_shared<Saiga::Vulkan::Texture2D>();
    Saiga::TemplatedImage<ucvec4> depthmg(frameData->depthImg.height,frameData->depthImg.width);
    Saiga::ImageTransformation::depthToRGBA(frameData->depthImg,depthmg,0,7000);
    texture2->fromImage(renderer.base,depthmg);



    textureDes = textureDisplay.createAndUpdateDescriptorSet(*texture);
    textureDes2 = textureDisplay.createAndUpdateDescriptorSet(*texture2);

    cout << "init done " << endl;
}



void VulkanExample::update(float dt)
{

}

void VulkanExample::transfer(vk::CommandBuffer cmd)
{

    rgbdcamera->readFrame(*frameData);


    texture->uploadImage(renderer.base,frameData->colorImg);

    Saiga::TemplatedImage<ucvec4> depthmg(frameData->depthImg.height,frameData->depthImg.width);
    Saiga::ImageTransformation::depthToRGBA(frameData->depthImg,depthmg,0,7000);
    texture2->uploadImage(renderer.base,depthmg);



}


void VulkanExample::render(vk::CommandBuffer cmd)
{

    textureDisplay.bind(cmd);

    {
        textureDisplay.bindDescriptorSets(cmd,textureDes);
        vk::Viewport vp(0,0,frameData->colorImg.width,frameData->colorImg.height);
        cmd.setViewport(0,vp);
        textureDisplay.blitMesh.render(cmd);
    }


    {
        textureDisplay.bindDescriptorSets(cmd,textureDes2);
        vk::Viewport vp(frameData->colorImg.width,0,frameData->depthImg.width,frameData->depthImg.height);
        cmd.setViewport(0,vp);
        textureDisplay.blitMesh.render(cmd);
    }
}

void VulkanExample::renderGUI()
{


    parentWindow.renderImGui();


}


void VulkanExample::keyPressed(SDL_Keysym key)
{
    switch(key.scancode){
    case SDL_SCANCODE_ESCAPE:
        parentWindow.close();
        break;
    default:
        break;
    }
}

void VulkanExample::keyReleased(SDL_Keysym key)
{
}


