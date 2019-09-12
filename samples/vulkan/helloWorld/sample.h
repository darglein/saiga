/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/vulkan/memory/VulkanMemory.h"
#include "saiga/vulkan/pipeline/DescriptorSet.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/renderModules/LineAssetRenderer.h"
#include "saiga/vulkan/renderModules/PointCloudRenderer.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/renderModules/TexturedAssetRenderer.h"
#include "saiga/vulkan/window/SDLSample.h"

#include <vector>
using namespace Saiga;
class VulkanExample : public VulkanSDLExampleBase
{
   public:
    VulkanExample();
    ~VulkanExample() override;

    void init(Saiga::Vulkan::VulkanBase& base);


    void update(float dt) override;
    void transfer(vk::CommandBuffer cmd) override;
    void render(vk::CommandBuffer cmd) override;
    void renderGUI() override;

   private:
    std::vector<vec3> boxOffsets;
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
    Saiga::Vulkan::StaticDescriptorSet textureDes;
    Saiga::Vulkan::TextureDisplay textureDisplay;


    bool displayModels = true;
};
