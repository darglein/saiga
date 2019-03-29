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
#include "saiga/core/util/imath.h"

#include <algorithm>
#include <iterator>
#include <saiga/core/imgui/imgui.h>

#include <glm/gtc/matrix_transform.hpp>
#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

VulkanExample::VulkanExample(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), mersenne_twister(), renderer(renderer)
{
    SAIGA_ASSERT(image_names.size() == images.size());
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f, true);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = vec3(0);

    window.setCamera(&camera);

    auto_mersenne = std::mt19937();

    init(renderer.base());
}

VulkanExample::~VulkanExample() {}

void VulkanExample::init(Saiga::Vulkan::VulkanBase& base)
{
    for (auto i = 0U; i < image_names.size(); ++i)
    {
        auto image = std::make_shared<Saiga::Image>(image_names[i]);

        if (image->type == Saiga::UC3)
        {
            auto img2 = std::make_shared<Saiga::TemplatedImage<ucvec4>>(image->height, image->width);
            Saiga::ImageTransformation::addAlphaChannel(image->getImageView<ucvec3>(), img2->getImageView(), 255);
            image = img2;
        }

        images[i] = image;

        LOG(INFO) << image_names[i] << " " << image.get()->size();
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

    std::for_each(to_delete_tex.begin(), to_delete_tex.end(), [](auto& entry) { std::get<2>(entry) -= 1; });

    auto end = std::remove_if(to_delete_tex.begin(), to_delete_tex.end(),
                              [](const auto& entry) { return std::get<2>(entry) < 0; });

    to_delete_tex.erase(end, to_delete_tex.end());
    //    renderer.base().memory.vertexIndexAllocator.deallocate(m_location3);
    //    m_location3 = renderer.base().memory.vertexIndexAllocator.allocate(1025);
}

void VulkanExample::transfer(vk::CommandBuffer cmd) {}


void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (show_textures && !tex_allocations.empty())
    {
        if (textureDisplay.bind(cmd))
        {
            const int width = 16;
            int index       = 0;


            std::stringstream ss;
            for (auto& texture : tex_allocations)
            {
                // auto set = textureDisplay.createAndUpdateDescriptorSet(*(texture.first));
                // VLOG(1) << "Displaying " << texture.first->memoryLocation->data.sampler;
                ss << (std::get<0>(texture))->memoryLocation->data.sampler << " ";
                vec2 position((index % width) * 64, (index / width) * 64);
                textureDisplay.renderTexture(cmd, std::get<1>(texture), position, vec2(64, 64));
                index++;
            }
            VLOG(1) << ss.str();
            // VLOG(1) << "===============";
        }
    }
}

void VulkanExample::renderGUI()
{
    static std::uniform_int_distribution<unsigned long> alloc_dist(1, 5), size_dist(0UL, 3UL), image_dist(0, 4);

    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Example settings");

    bool old_enable = enable_defragger;
    ImGui::Checkbox("Defragger enabled", &old_enable);
    if (old_enable != enable_defragger)
    {
        enable_defragger = old_enable;
        renderer.base().memory.enable_defragmentation(buffer_type, enable_defragger);
    }

    ImGui::Checkbox("Auto allocate indexed", &enable_auto_index);
    ImGui::Text("%d", auto_allocs);

    if (ImGui::Button("Allocate Image"))
    {
        renderer.base().memory.enable_defragmentation(image_type, false);
        renderer.base().memory.stop_defrag(image_type);
        auto num_allocs = alloc_dist(mersenne_twister);

        for (auto i = 0U; i < num_allocs; ++i)
        {
            auto index = image_dist(mersenne_twister);
            // allocations.push_back(renderer.base().memory.allocate(buffer_type, size));
            tex_allocations.push_back(allocate(image_type, index));
        }
        renderer.base().memory.enable_defragmentation(buffer_type, enable_defragger);
        renderer.base().memory.start_defrag(buffer_type);
    }

    if (ImGui::Button("Deallocate Image"))
    {
        renderer.base().memory.enable_defragmentation(image_type, false);
        renderer.base().memory.stop_defrag(image_type);

        auto num_allocs = std::min(alloc_dist(mersenne_twister), tex_allocations.size());


        for (auto i = 0U; i < num_allocs; ++i)
        {
            auto index = mersenne_twister() % tex_allocations.size();

            std::move(tex_allocations.begin() + index, tex_allocations.begin() + index + 1,
                      std::back_inserter(to_delete_tex));
            // auto& alloc = tex_allocations[index];
            // to_delete_tex.push_back(std::make_tuple(std::move(alloc.first), std::move(alloc.second), 12));
            tex_allocations.erase(tex_allocations.begin() + index);
        }
        renderer.base().memory.enable_defragmentation(image_type, enable_defragger);
        renderer.base().memory.start_defrag(image_type);
    }

    ImGui::Checkbox("Show textures", &show_textures);
    ImGui::End();

    parentWindow.renderImGui();
}


void VulkanExample::keyPressed(SDL_Keysym key)
{
    static std::uniform_int_distribution<unsigned long> alloc_dist(1UL, 8UL), size_dist(0UL, 3UL), image_dist(0, 4);



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
            renderer.base().memory.enable_defragmentation(buffer_type, false);
            renderer.base().memory.stop_defrag(buffer_type);
            num_allocs = alloc_dist(mersenne_twister);

            for (auto i = 0U; i < num_allocs; ++i)
            {
                auto size = sizes[size_dist(mersenne_twister)];
                // allocations.push_back(renderer.base().memory.allocate(buffer_type, size));
                allocations.push_back(allocate(buffer_type, size));
            }
            renderer.base().memory.enable_defragmentation(buffer_type, enable_defragger);
            renderer.base().memory.start_defrag(buffer_type);
            break;
        case SDL_SCANCODE_D:

            renderer.base().memory.enable_defragmentation(buffer_type, false);
            renderer.base().memory.stop_defrag(buffer_type);

            num_allocs = std::min(alloc_dist(mersenne_twister), allocations.size());

            for (auto i = 0U; i < num_allocs; ++i)
            {
                auto index = mersenne_twister() % allocations.size();

                // renderer.base().memory.deallocateBuffer(buffer_type, allocations[index].first);
                allocations.erase(allocations.begin() + index);
            }
            renderer.base().memory.enable_defragmentation(buffer_type, enable_defragger);
            renderer.base().memory.start_defrag(buffer_type);

            break;
        case SDL_SCANCODE_Z:
        {
            renderer.base().memory.enable_defragmentation(image_type, false);
            renderer.base().memory.stop_defrag(image_type);
            num_allocs = alloc_dist(mersenne_twister);

            for (auto i = 0U; i < num_allocs; ++i)
            {
                auto index = image_dist(mersenne_twister);
                // allocations.push_back(renderer.base().memory.allocate(buffer_type, size));
                tex_allocations.push_back(allocate(image_type, index));
            }
            renderer.base().memory.enable_defragmentation(buffer_type, enable_defragger);
            renderer.base().memory.start_defrag(buffer_type);
        }
        break;
        case SDL_SCANCODE_C:
        {
            renderer.base().memory.enable_defragmentation(image_type, false);
            renderer.base().memory.stop_defrag(image_type);

            num_allocs = std::min(alloc_dist(mersenne_twister), tex_allocations.size());


            for (auto i = 0U; i < num_allocs; ++i)
            {
                auto index = mersenne_twister() % tex_allocations.size();
                LOG(INFO) << "Dealloc image " << index;
                std::move(tex_allocations.begin() + index, tex_allocations.begin() + index + 1,
                          std::back_inserter(to_delete_tex));
                tex_allocations.erase(tex_allocations.begin() + index);
            }
            renderer.base().memory.enable_defragmentation(image_type, enable_defragger);
            renderer.base().memory.start_defrag(image_type);
        }
        break;
        case SDL_SCANCODE_R:
            renderer.base().memory.enable_defragmentation(buffer_type, false);
            renderer.base().memory.stop_defrag(buffer_type);

            for (int i = 0; i < 10; ++i)
            {
                alloc_index(i);
            }
            renderer.base().memory.enable_defragmentation(buffer_type, enable_defragger);
            renderer.base().memory.start_defrag(buffer_type);
            break;
        case SDL_SCANCODE_V:
        {
            for (auto& numAlloc : num_allocations)
            {
                if (numAlloc.first)
                {
                    auto& buffer      = *numAlloc.first.get();
                    auto num_elements = buffer.size() / sizeof(uint32_t);
                    std::vector<uint32_t> copy(num_elements);

                    buffer.stagedDownload(copy.data());


                    if (copy[0] != numAlloc.second)
                    {
                        LOG(ERROR) << "Wrong value " << copy[0] << " != " << numAlloc.second;
                    }
                    else
                    {
                        LOG(INFO) << "Verified " << copy[0];
                    }
                }
            }
        }
        break;
        case SDL_SCANCODE_ESCAPE:
            cleanup();
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
        // renderer.base().memory.deallocateBuffer(buffer_type, loc);
        num_allocations[index] = std::make_pair(nullptr, 0);
    }
    else
    {
        num_allocations[index] = allocate(buffer_type, sizes[3]);
        // num_allocations[index] = renderer.base().memory.allocate(buffer_type, sizes[3]);
    }
}

void VulkanExample::keyReleased(SDL_Keysym key) {}

std::pair<std::shared_ptr<Saiga::Vulkan::Buffer>, uint32_t> VulkanExample::allocate(
    Saiga::Vulkan::Memory::BufferType type, unsigned long long int size)
{
    static std::uniform_int_distribution<unsigned long> init_dist(0UL, 1024UL);

    auto start = init_dist(mersenne_twister);

    LOG(INFO) << "Creating buffer of size " << size << " beginning at " << start;
    std::vector<uint32_t> mem;
    mem.resize(size / sizeof(uint32_t));
    std::iota(mem.begin(), mem.end(), start);
    std::shared_ptr<Saiga::Vulkan::Buffer> buffer = std::make_shared<Saiga::Vulkan::Buffer>();
    buffer->createBuffer(renderer.base(), size, type.usageFlags, type.memoryFlags);

    buffer->stagedUpload(renderer.base(), size, mem.data());

    buffer->mark_dynamic();

    return std::make_pair(buffer, start);
}

std::tuple<std::shared_ptr<Saiga::Vulkan::Texture2D>, Saiga::Vulkan::DynamicDescriptorSet, int32_t>
VulkanExample::allocate(Saiga::Vulkan::Memory::ImageType type, unsigned long long int index)
{
    std::shared_ptr<Saiga::Vulkan::Texture2D> texture = std::make_shared<Saiga::Vulkan::Texture2D>();

    texture->fromImage(renderer.base(), *images[index]);
    texture->mark_dynamic();

    auto descriptor = textureDisplay.createDynamicDescriptorSet();
    descriptor.assign(0, texture.get());
    return std::make_tuple(texture, std::move(descriptor), 12);
}

void VulkanExample::cleanup()
{
    renderer.base().device.waitIdle();

    LOG(INFO) << allocations.size();
    // if (!allocations.empty())
    //{
    //    allocations.resize(0);
    //}
    // if (!num_allocations.empty())
    //{
    //    num_allocations.resize(0);
    //}
    // if (!tex_allocations.empty())
    //{
    //    tex_allocations.resize(0);
    //}
}
