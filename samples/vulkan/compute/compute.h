/*
* Vulkan Example - imGui (https://github.com/ocornut/imgui)
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/sdl/sdl_camera.h"
#include "saiga/window/Interfaces.h"

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"

#include "saiga/vulkan/CommandPool.h"

class Compute :  public Saiga::Updating, public Saiga::Vulkan::VulkanForwardRenderingInterface, public Saiga::SDL_KeyListener
{
public:
    Compute(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer);
    ~Compute();

    void init(Saiga::Vulkan::VulkanBase& base);

    virtual void update(float dt) override;
    virtual void render(VkCommandBuffer cmd) override;
    virtual void renderGUI() override;
private:
    Saiga::SDLCamera<Saiga::PerspectiveCamera> camera;

    Saiga::Object3D teapotTrans;
    Saiga::Vulkan::VulkanVertexColoredAsset teapot,plane;
    Saiga::Vulkan::AssetRenderer assetRenderer;

    Saiga::Vulkan::VulkanForwardRenderer &renderer;

    bool displayModels = true;



    Saiga::Vulkan::VulkanBase* vulkanDevice;
    vk::Device device;

    struct {					// Uniform buffer object containing particle system parameters
        std::vector<int> data;
        Saiga::Vulkan::Buffer storageBuffer;

        vk::Queue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
        Saiga::Vulkan::CommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
        vk::CommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
        VkFence fence;								// Synchronization fence to avoid rewriting compute CB if still in use


    } compute;
    Saiga::Vulkan::ComputePipeline computePipeline;
    vk::DescriptorSet       descriptorSet;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};

