/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once


#include "saiga/vision/recursive/BARecursive.h"
#include "saiga/vision/scene/Scene.h"
#include "saiga/vision/scene/SynteticScene.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/renderModules/LineAssetRenderer.h"
#include "saiga/vulkan/renderModules/PointCloudRenderer.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/renderModules/TexturedAssetRenderer.h"
#include "saiga/vulkan/window/SDLSample.h"

using namespace Saiga;

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
    bool change           = false;
    bool uploadChanges    = true;
    float rms             = 0;
    int minMatchEdge      = 1000;
    float maxEdgeDistance = 1;
    Saiga::Object3D teapotTrans;

    Saiga::Scene scene;
    Saiga::SynteticScene sscene;
    std::shared_ptr<Saiga::Vulkan::Texture2D> texture;

    Saiga::Vulkan::VulkanPointCloudAsset graphLines;

    Saiga::Vulkan::VulkanTexturedAsset box;
    Saiga::Vulkan::VulkanVertexColoredAsset teapot, plane;
    Saiga::Vulkan::VulkanLineVertexColoredAsset grid, frustum;
    Saiga::Vulkan::VulkanPointCloudAsset pointCloud;
    Saiga::Vulkan::AssetRenderer assetRenderer;
    Saiga::Vulkan::LineAssetRenderer lineAssetRenderer;
    Saiga::Vulkan::PointCloudRenderer pointCloudRenderer;

    //
    vk::DescriptorSet textureDes;
    Saiga::Vulkan::TextureDisplay textureDisplay;


    bool displayModels = true;
    bool showImgui     = true;

    Saiga::BAOptions baoptions;
    Saiga::BARec barec;

    std::vector<std::string> datasets;
};
