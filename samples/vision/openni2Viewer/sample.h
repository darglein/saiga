/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/vision/camera/RGBDCamera.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/renderModules/LineAssetRenderer.h"
#include "saiga/vulkan/renderModules/PointCloudRenderer.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/renderModules/TexturedAssetRenderer.h"

class VulkanExample : public Saiga::Updating,
                      public Saiga::Vulkan::VulkanForwardRenderingInterface,
                      public Saiga::SDL_KeyListener
{
   public:
    VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer);
    ~VulkanExample();

    void init(Saiga::Vulkan::VulkanBase& base);


    virtual void update(float dt) override;
    virtual void transfer(vk::CommandBuffer cmd) override;
    virtual void render(vk::CommandBuffer cmd) override;
    virtual void renderGUI() override;

   private:
    Saiga::RGBDFrameData frameData;
    std::unique_ptr<Saiga::RGBDCamera> rgbdcamera;

    Saiga::TemplatedImage<ucvec4> rgbImage;
    Saiga::TemplatedImage<ucvec4> depthmg;


    std::shared_ptr<Saiga::Vulkan::Texture2D> texture;
    std::shared_ptr<Saiga::Vulkan::Texture2D> texture2;


    char dir[256]      = "recording/";
    bool capturing     = false;
    int frameId        = 0;
    bool updateTexture = false;
    bool initTexture   = false;

    Saiga::Vulkan::StaticDescriptorSet textureDes;
    Saiga::Vulkan::StaticDescriptorSet textureDes2;
    Saiga::Vulkan::TextureDisplay textureDisplay;

    Saiga::Vulkan::VulkanForwardRenderer& renderer;



    ImGui::HzTimeGraph tg;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};
