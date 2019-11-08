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
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/memory/BufferChunkAllocator.h"
#include "saiga/vulkan/memory/VulkanMemory.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/renderModules/LineAssetRenderer.h"
#include "saiga/vulkan/renderModules/PointCloudRenderer.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/renderModules/TexturedAssetRenderer.h"
#include "saiga/vulkan/window/SDLSample.h"

#include <random>
#include <utility>
#include <vector>
using namespace Saiga;
using Saiga::Vulkan::Memory::BufferMemoryLocation;
class VulkanExample : public VulkanSDLExampleBase, public Saiga::SDL_KeyListener
{
    std::array<std::string, 5> image_names{"cat.png", "red-panda.png", "dog.png", "pika.png", "ludi.png"};
    std::array<std::shared_ptr<Saiga::Image>, 5> images;
    std::vector<std::shared_ptr<Saiga::Vulkan::Buffer>> test_allocs;
    std::vector<std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t>> allocations;
    std::vector<std::tuple<std::shared_ptr<Saiga::Vulkan::Texture2D>, Saiga::Vulkan::DynamicDescriptorSet, int32_t>>
        tex_allocations;
    std::vector<std::tuple<std::shared_ptr<Saiga::Vulkan::Texture2D>, Saiga::Vulkan::DynamicDescriptorSet, int32_t>>
        to_delete_tex;
    std::vector<std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t>> num_allocations;

    std::mt19937 mersenne_twister, auto_mersenne, defrag_test_mersenne;


    //    std::array<vk::DeviceSize, 4> tex_sizes{256, 512, 1024, 2048};
    std::array<vk::DeviceSize, 4> sizes{256 * 256, 512 * 512, 1024 * 1024, 16 * 1024 * 1024};

    Saiga::Vulkan::Memory::BufferType buffer_type{vk::BufferUsageFlagBits::eTransferDst,
                                                  vk::MemoryPropertyFlagBits::eDeviceLocal};
    Saiga::Vulkan::Memory::ImageType image_type{vk::ImageUsageFlagBits::eSampled,
                                                vk::MemoryPropertyFlagBits::eDeviceLocal};

    Saiga::Vulkan::TextureDisplay textureDisplay;
    vk::DescriptorSet textureDes = nullptr;


    bool enable_auto_index = false;
    bool enable_defragger  = false;
    int auto_allocs        = 0;

   public:
    VulkanExample();
    ~VulkanExample() override;

    void init(Saiga::Vulkan::VulkanBase& base);


    void update(float dt) override;
    void transfer(vk::CommandBuffer cmd) override;
    void render(vk::CommandBuffer cmd) override;
    void renderGUI() override;

   private:
    int defrag_tests          = 1000;
    int defrag_rounds         = 50;
    int defrag_current_test   = -1;
    int defrag_current_round  = 0;
    float defrag_current_loc  = -1;
    float defrag_current_free = -1;
    std::vector<unsigned int> defrag_seeds;

    bool show_textures = false;
    Saiga::SDLCamera<Saiga::PerspectiveCamera> camera;


    //    bool displayModels = true;


    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;

    void alloc_index(int index);

    std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t> allocate(Saiga::Vulkan::Memory::BufferType type,
                                                                         unsigned long long int size);
    std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t> allocateEmpty(Saiga::Vulkan::Memory::BufferType type,
                                                                              unsigned long long int size);

    std::tuple<std::shared_ptr<Saiga::Vulkan::Texture2D>, Saiga::Vulkan::DynamicDescriptorSet, int32_t> allocate(
        Saiga::Vulkan::Memory::ImageType type, unsigned long long int size);
    void cleanup();
    void speedProfiling() const;
    void fragmentationProfiling();
    void defragProfiling();
};
