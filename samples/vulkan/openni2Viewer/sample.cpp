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




    auto cam = std::make_shared<Saiga::RGBDCameraNetwork>();
    cam->connect(ip,port);
    rgbdcamera = cam;

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



    rgbdcamera->readFrame();

    texture = std::make_shared<Saiga::Vulkan::Texture2D>();
    texture->fromImage(renderer.base,rgbdcamera->colorImg);


    texture2 = std::make_shared<Saiga::Vulkan::Texture2D>();
    Saiga::TemplatedImage<ucvec4> depthmg(rgbdcamera->depthImg.height,rgbdcamera->depthImg.width);
    Saiga::ImageTransformation::depthToRGBA(rgbdcamera->depthImg,depthmg,0,7000);
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

    rgbdcamera->readFrame();


    texture->uploadImage(renderer.base,rgbdcamera->colorImg);

    Saiga::TemplatedImage<ucvec4> depthmg(rgbdcamera->depthImg.height,rgbdcamera->depthImg.width);
    Saiga::ImageTransformation::depthToRGBA(rgbdcamera->depthImg,depthmg,0,7000);
    texture2->uploadImage(renderer.base,depthmg);



}


void VulkanExample::render(vk::CommandBuffer cmd)
{

    textureDisplay.bind(cmd);

    {
        textureDisplay.bindDescriptorSets(cmd,textureDes);
        vk::Viewport vp(0,0,rgbdcamera->colorImg.width,rgbdcamera->colorImg.height);
        cmd.setViewport(0,vp);
        textureDisplay.blitMesh.render(cmd);
    }


    {
        textureDisplay.bindDescriptorSets(cmd,textureDes2);
        vk::Viewport vp(rgbdcamera->colorImg.width,0,rgbdcamera->depthImg.width,rgbdcamera->depthImg.height);
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


