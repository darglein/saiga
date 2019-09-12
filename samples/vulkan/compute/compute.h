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
#include "saiga/vulkan/CommandPool.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/pipeline/DescriptorSet.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/texture/Texture.h"
#include "saiga/vulkan/window/SDLSample.h"
using namespace Saiga;

class Compute : public VulkanSDLExampleBase
{
   public:
    Compute();
    ~Compute();

    void init(Saiga::Vulkan::VulkanBase& base);

    void update(float dt) override;

    void render(vk::CommandBuffer cmd) override;



   private:
    Saiga::SDLCamera<Saiga::PerspectiveCamera> camera;

    Saiga::Object3D teapotTrans;
    Saiga::Vulkan::VulkanVertexColoredAsset teapot, plane;
    Saiga::Vulkan::AssetRenderer assetRenderer;


    Saiga::Vulkan::VulkanBase* vulkanDevice;
    vk::Device device;

    struct
    {  // Uniform buffer object containing particle system parameters
        std::vector<int> data;
        Saiga::Vulkan::Buffer storageBuffer;
        Saiga::Vulkan::Texture2D storageTexture;
        vk::CommandBuffer commandBuffer;  // Command buffer storing the dispatch commands and barriers
    } compute;
    Saiga::Vulkan::ComputePipeline computePipeline;
    Saiga::Vulkan::StaticDescriptorSet descriptorSet;
};
