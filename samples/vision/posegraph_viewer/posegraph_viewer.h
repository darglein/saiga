/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once


#include "saiga/vision/pgo/PGORecursive.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/renderModules/LineAssetRenderer.h"
#include "saiga/vulkan/renderModules/PointCloudRenderer.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/renderModules/TexturedAssetRenderer.h"
#include "saiga/vulkan/window/SDLSample.h"


class VulkanExample : public Saiga::VulkanSDLExampleBase
{
   public:
    VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer);
    ~VulkanExample() override;

    void init(Saiga::Vulkan::VulkanBase& base);


    void update(float dt) override;
    void transfer(vk::CommandBuffer cmd) override;
    void render(vk::CommandBuffer cmd) override;
    void renderGUI() override;

   private:
    std::vector<vec3> boxOffsets;
    bool change = false;
    //    bool uploadChanges = true;
    float rms  = 0;
    float chi2 = 0;
    Saiga::Object3D teapotTrans;

    std::shared_ptr<Saiga::Vulkan::Texture2D> texture;

    Saiga::Vulkan::VulkanTexturedAsset box;
    Saiga::Vulkan::VulkanVertexColoredAsset teapot, plane;
    Saiga::Vulkan::VulkanLineVertexColoredAsset grid, frustum;
    Saiga::Vulkan::VulkanPointCloudAsset pointCloud;
    Saiga::Vulkan::AssetRenderer assetRenderer;
    Saiga::Vulkan::LineAssetRenderer lineAssetRenderer;
    Saiga::Vulkan::VulkanPointCloudAsset lineAsset;
    std::vector<Saiga::VertexNC> lines;
    //
    vk::DescriptorSet textureDes;
    Saiga::Vulkan::TextureDisplay textureDisplay;


    //    bool displayModels = true;

    Saiga::PoseGraph scene;
    Saiga::OptimizationOptions baoptions;

    std::vector<std::string> datasets, baldatasets;
};
