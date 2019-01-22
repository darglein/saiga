/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include "saiga/sdl/sdl_camera.h"
#include "saiga/sdl/sdl_eventhandler.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/memory/VulkanMemory.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/renderModules/LineAssetRenderer.h"
#include "saiga/vulkan/renderModules/PointCloudRenderer.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/renderModules/TexturedAssetRenderer.h"
#include "saiga/window/Interfaces.h"

#include <vector>
class VulkanExample : public Saiga::Updating,
                      public Saiga::Vulkan::VulkanForwardRenderingInterface,
                      public Saiga::SDL_KeyListener
{
   public:
    VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer);
    ~VulkanExample() override;

    void init(Saiga::Vulkan::VulkanBase& base) override;


    void update(float dt) override;
    void transfer(vk::CommandBuffer cmd) override;
    void render(vk::CommandBuffer cmd) override;
    void renderGUI() override;

   private:
    std::vector<vec3> boxOffsets;
    Saiga::SDLCamera<Saiga::PerspectiveCamera> camera;
    bool change        = false;
    bool uploadChanges = true;
    Saiga::Object3D teapotTrans;

    std::shared_ptr<Saiga::Vulkan::Texture2D> texture;

    Saiga::Vulkan::VulkanTexturedAsset box;
    Saiga::Vulkan::VulkanVertexColoredAsset teapot, plane;
    Saiga::Vulkan::VulkanLineVertexColoredAsset grid, frustum;
    Saiga::Vulkan::VulkanPointCloudAsset pointCloud;
    Saiga::Vulkan::AssetRenderer assetRenderer;
    Saiga::Vulkan::LineAssetRenderer lineAssetRenderer;
    Saiga::Vulkan::PointCloudRenderer pointCloudRenderer;
    Saiga::Vulkan::TexturedAssetRenderer texturedAssetRenderer;

    //
    vk::DescriptorSet textureDes;
    Saiga::Vulkan::TextureDisplay textureDisplay;

    Saiga::Vulkan::VulkanForwardRenderer& renderer;

    bool displayModels = true;


    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};
