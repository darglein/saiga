/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "sample.h"

#include "saiga/core/image/imageTransformations.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/color.h"

#include <algorithm>
#include <saiga/core/imgui/imgui.h>

#include <glm/gtc/matrix_transform.hpp>
#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), mersenne_twister(), renderer(renderer)
{
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f, true);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = vec3(0);

    window.setCamera(&camera);

    auto_mersenne = std::mt19937();
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    for (int i = 0; i < image_names.size(); ++i)
    {
        auto image = std::make_shared<Saiga::Image>(image_names[i]);

        if (image->type == Saiga::UC3)
        {
            auto img2 = std::make_shared<Saiga::TemplatedImage<ucvec4>>(image->height, image->width);
            Saiga::ImageTransformation::addAlphaChannel(image->getImageView<ucvec3>(), img2->getImageView(), 255);
            image = img2;
        }

        images[i] = image;
    }
    num_allocations.resize(10, std::make_pair(nullptr, 0));

    textureDisplay.init(base, renderer.renderPass);
}



void VulkanExample::update(float dt)
{
    camera.update(dt);
    camera.interpolate(dt, 0);


    if (enable_auto_index)
    {
        static std::uniform_int_distribution<> alloc_dist(0, 9);

        // std::cout << alloc_dist(auto_mersenne) << std::endl;
        alloc_index(alloc_dist(auto_mersenne));
        auto_allocs++;
    }
    //    renderer.base.memory.vertexIndexAllocator.deallocate(m_location3);
    //    m_location3 = renderer.base.memory.vertexIndexAllocator.allocate(1025);
}

void VulkanExample::transfer(vk::CommandBuffer cmd) {}


void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (show_textures && !tex_allocations.empty() && textureDes)
    {
        if (textureDisplay.bind(cmd))
        {
            textureDisplay.renderTexture(cmd, textureDes, vec2(10, 10), vec2(256, 256));
        }
    }
}

void VulkanExample::renderGUI()
{
    static std::uniform_int_distribution<unsigned long> alloc_dist(1UL, 15UL), size_dist(0UL, 3UL), image_dist(0, 2);

    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Example settings");

    bool old_enable = enable_defragger;
    ImGui::Checkbox("Defragger enabled", &old_enable);
    if (old_enable != enable_defragger)
    {
        enable_defragger = old_enable;
        renderer.base.memory.enable_defragmentation(buffer_type, enable_defragger);
    }
    ImGui::Checkbox("Auto allocate indexed", &enable_auto_index);
    ImGui::Text("%d", auto_allocs);


    if (ImGui::Button("Allocate Image"))
    {
        renderer.base.memory.enable_defragmentation(image_type, false);
        renderer.base.memory.stop_defrag(image_type);
        auto num_allocs = alloc_dist(mersenne_twister);

        for (int i = 0; i < num_allocs; ++i)
        {
            auto size = image_dist(mersenne_twister);
            // allocations.push_back(renderer.base.memory.allocate(buffer_type, size));
            tex_allocations.push_back(allocate(image_type, size));
        }
        renderer.base.memory.enable_defragmentation(buffer_type, enable_defragger);
        renderer.base.memory.start_defrag(buffer_type);
    }

    if (ImGui::Button("Deallocate Image"))
    {
        renderer.base.memory.enable_defragmentation(image_type, false);
        renderer.base.memory.stop_defrag(image_type);

        auto num_allocs = std::min(alloc_dist(mersenne_twister), tex_allocations.size());


        for (int i = 0; i < num_allocs; ++i)
        {
            auto index = mersenne_twister() % tex_allocations.size();

            // renderer.base.memory.deallocateBuffer(buffer_type, allocations[index].first);
            tex_allocations.erase(tex_allocations.begin() + index);
        }
        renderer.base.memory.enable_defragmentation(image_type, enable_defragger);
        renderer.base.memory.start_defrag(image_type);
    }

    ImGui::Checkbox("Show textures", &show_textures);
    if (show_textures && !tex_allocations.empty())
    {
        int new_index = texture_index;
        ImGui::SliderInt("Texture Index", &new_index, 0, tex_allocations.size() - 1);
        if (new_index != texture_index)
        {
            texture_index = new_index;
            textureDes    = textureDisplay.createAndUpdateDescriptorSet(*tex_allocations[texture_index].first);
        }
    }
    else
    {
        textureDes = nullptr;
    }
    ImGui::End();

    parentWindow.renderImGui();
}


void VulkanExample::keyPressed(SDL_Keysym key)
{
    static std::uniform_int_distribution<unsigned long> alloc_dist(1UL, 15UL), size_dist(0UL, 3UL);



    int single_unassign = -1;
    unsigned long num_allocs;
    switch (key.scancode)
    {
        case SDL_SCANCODE_0:
            single_unassign = 10;
            break;
        case SDL_SCANCODE_9:
            single_unassign = 9;
            break;
        case SDL_SCANCODE_8:
            single_unassign = 8;
            break;
        case SDL_SCANCODE_7:
            single_unassign = 7;
            break;
        case SDL_SCANCODE_6:
            single_unassign = 6;
            break;
        case SDL_SCANCODE_5:
            single_unassign = 5;
            break;
        case SDL_SCANCODE_4:
            single_unassign = 4;
            break;
        case SDL_SCANCODE_3:
            single_unassign = 3;
            break;
        case SDL_SCANCODE_2:
            single_unassign = 2;
            break;
        case SDL_SCANCODE_1:
            single_unassign = 1;
            break;

        case SDL_SCANCODE_A:
            renderer.base.memory.enable_defragmentation(buffer_type, false);
            renderer.base.memory.stop_defrag(buffer_type);
            num_allocs = alloc_dist(mersenne_twister);

            for (int i = 0; i < num_allocs; ++i)
            {
                auto size = sizes[size_dist(mersenne_twister)];
                // allocations.push_back(renderer.base.memory.allocate(buffer_type, size));
                allocations.push_back(allocate(buffer_type, size));
            }
            renderer.base.memory.enable_defragmentation(buffer_type, enable_defragger);
            renderer.base.memory.start_defrag(buffer_type);
            break;
        case SDL_SCANCODE_D:

            renderer.base.memory.enable_defragmentation(buffer_type, false);
            renderer.base.memory.stop_defrag(buffer_type);

            num_allocs = std::min(alloc_dist(mersenne_twister), allocations.size());


            for (int i = 0; i < num_allocs; ++i)
            {
                auto index = mersenne_twister() % allocations.size();

                // renderer.base.memory.deallocateBuffer(buffer_type, allocations[index].first);
                allocations.erase(allocations.begin() + index);
            }
            renderer.base.memory.enable_defragmentation(buffer_type, enable_defragger);
            renderer.base.memory.start_defrag(buffer_type);

            break;
        case SDL_SCANCODE_F:
            enable_defragger = !enable_defragger;

            renderer.base.memory.enable_defragmentation(buffer_type, enable_defragger);
            break;
        case SDL_SCANCODE_ESCAPE:
            parentWindow.close();
            break;
        default:
            break;
    }

    if (single_unassign > 0)
    {
        auto index = single_unassign - 1;
        alloc_index(index);
    }
}

void VulkanExample::alloc_index(int index)
{
    if (num_allocations[index].first)
    {
        // num_allocations[index].first->destroy();
        // BufferMemoryLocation* loc = num_allocations[index].first;
        // allocations.erase(allocations.begin() + index);
        // renderer.base.memory.deallocateBuffer(buffer_type, loc);
        num_allocations[index] = std::make_pair(nullptr, 0);
    }
    else
    {
        num_allocations[index] = allocate(buffer_type, sizes[3]);
        // num_allocations[index] = renderer.base.memory.allocate(buffer_type, sizes[3]);
    }
}

void VulkanExample::keyReleased(SDL_Keysym key) {}

std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t> VulkanExample::allocate(
    Saiga::Vulkan::Memory::BufferType type, unsigned long long int size)
{
    static std::uniform_int_distribution<unsigned long> init_dist(0UL, 1024UL);

    auto start = init_dist(mersenne_twister);

    std::vector<uint32_t> mem;
    mem.resize(size / sizeof(uint32_t) + 1);
    std::iota(mem.begin(), mem.end(), start);
    std::shared_ptr<Saiga::Vulkan::Buffer> buffer = std::make_shared<Saiga::Vulkan::Buffer>();
    buffer->createBuffer(renderer.base, size, type.usageFlags, type.memoryFlags);

    buffer->stagedUpload(renderer.base, size, mem.data());

    buffer->mark_dynamic();

    return std::make_pair(buffer, start);
}

std::pair<std::shared_ptr<Saiga::Vulkan::Texture2D>, uint32_t> VulkanExample::allocate(
    Saiga::Vulkan::Memory::ImageType type, unsigned long long int index)
{
    std::shared_ptr<Saiga::Vulkan::Texture2D> texture = std::make_shared<Saiga::Vulkan::Texture2D>();

    // std::vector<uint32_t> mem;
    // mem.resize(size * size);
    //
    // std::iota(mem.begin(), mem.end(), 0);

    // Saiga::Vulkan::StagingBuffer staging;
    // staging.init(renderer.base, size * size, mem.data());

    texture->fromImage(renderer.base, *images[index]);
    // auto init_operation =
    //    texture->fromStagingBuffer(renderer.base, size, size, vk::Format::eR8G8B8A8Uint, staging,
    //                               *renderer.base.transferQueue, renderer.base.transferQueue->commandPool);
    //
    // renderer.base.device.waitForFences(init_operation.fence, true, 100000000000);
    // renderer.base.transferQueue->commandPool.freeCommandBuffer(init_operation.cmd);
    //
    // renderer.base.device.destroy(init_operation.fence);
    texture->mark_dynamic();

    return std::make_pair(texture, 0);
}