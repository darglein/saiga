/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "sample.h"

#include "saiga/image/imageTransformations.h"
#include "saiga/imgui/imgui.h"
#include "saiga/util/color.h"

#include <algorithm>
#include <saiga/imgui/imgui.h>

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
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    // m_location1 = base.memory.vertexIndexAllocator.allocate(1024);
    // m_location2 = base.memory.vertexIndexAllocator.allocate(1024);
    // m_location3 = base.memory.vertexIndexAllocator.allocate(1024);
    num_allocations.resize(10, nullptr);
}



void VulkanExample::update(float dt)
{
    camera.update(dt);
    camera.interpolate(dt, 0);

    //    renderer.base.memory.vertexIndexAllocator.deallocate(m_location3);
    //    m_location3 = renderer.base.memory.vertexIndexAllocator.allocate(1025);
}

void VulkanExample::transfer(vk::CommandBuffer cmd) {}


void VulkanExample::render(vk::CommandBuffer cmd) {}

void VulkanExample::renderGUI()
{
    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Example settings");
    ImGui::End();

    parentWindow.renderImGui();
}


void VulkanExample::keyPressed(SDL_Keysym key)
{
    static std::uniform_int_distribution<unsigned long> alloc_dist(1UL, 15UL), size_dist(0UL, 2UL);
    static std::array<vk::DeviceSize, 3> sizes{256 * 256, 512 * 512, 1024 * 1024};

    static const Saiga::Vulkan::Memory::BufferType buffer_type{vk::BufferUsageFlagBits::eVertexBuffer,
                                                               vk::MemoryPropertyFlagBits::eDeviceLocal};
    static bool enable_defragger = false;


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

        case SDL_SCANCODE_J:
            for (int i = 0; i < num_allocations.size(); ++i)
            {
                if (num_allocations[i]->memory)
                {
                    renderer.base.memory.deallocateBuffer(buffer_type, num_allocations[i]);
                }
            }
            num_allocations.clear();
            for (int i = 0; i < 10; ++i)
            {
                // auto size = sizes[size_dist(mersenne_twister)];
                num_allocations.push_back(renderer.base.memory.allocate(buffer_type, sizes[1]));
            }
            break;

        case SDL_SCANCODE_A:

            num_allocs = alloc_dist(mersenne_twister);

            for (int i = 0; i < num_allocs; ++i)
            {
                auto size = sizes[size_dist(mersenne_twister)];
                allocations.push_back(renderer.base.memory.allocate(buffer_type, size));
            }
            break;
        case SDL_SCANCODE_D:
            num_allocs = std::min(alloc_dist(mersenne_twister), allocations.size());

            for (int i = 0; i < num_allocs; ++i)
            {
                auto index = mersenne_twister() % allocations.size();

                MemoryLocation* loc = allocations[index];
                allocations.erase(allocations.begin() + index);
                renderer.base.memory.deallocateBuffer(buffer_type, loc);
            }
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
        if (num_allocations[index] && num_allocations[index]->memory)
        {
            MemoryLocation* loc = num_allocations[index];
            // allocations.erase(allocations.begin() + index);
            renderer.base.memory.deallocateBuffer(buffer_type, loc);
            loc->memory            = nullptr;
            num_allocations[index] = nullptr;
        }
        else
        {
            num_allocations[index] = renderer.base.memory.allocate(buffer_type, sizes[1]);
        }
    }
}

void VulkanExample::keyReleased(SDL_Keysym key) {}
