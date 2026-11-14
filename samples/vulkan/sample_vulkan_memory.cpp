/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Allocates and frees dynamic (movable) buffers and textures to fragment the memory, and defragments it with VMA's
// defragmentation (see Vulkan::Memory::VulkanMemory).
//
// Keys: A/D allocate/free buffers, Z/C allocate/free textures, V verify the buffer contents, J full defragmentation.
// The incremental defragmentation (one pass per frame) is enabled in the GUI or with enableDefragmentation=true in
// config.ini.

#include "saiga/core/image/imageTransformations.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/memory/VulkanMemory.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/window/GLFWSample.h"

#include <array>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

using namespace Saiga;

class VulkanExample : public VulkanExampleBase
{
   public:
    VulkanExample();

    void render(vk::CommandBuffer cmd) override;
    void renderGUI() override;

   private:
    struct BufferAllocation
    {
        std::unique_ptr<Vulkan::Buffer> buffer;
        // The content is first, first + 1, ...
        uint32_t first;
    };
    struct TextureAllocation
    {
        std::unique_ptr<Vulkan::Texture2D> texture;
        // Follows the texture when it is moved.
        Vulkan::DynamicDescriptorSet descriptor;
    };

    std::array<std::string, 5> imageNames{"textures/box.png", "textures/redie.png", "textures/sample.png",
                                          "textures/landscape.jpg", "textures/bar.png"};
    std::vector<Image> images;

    std::vector<BufferAllocation> buffers;
    std::vector<TextureAllocation> textures;
    Vulkan::TextureDisplay textureDisplay;

    std::mt19937 random{3465};
    bool showTextures = true;

    Vulkan::Memory::VulkanMemory& memory() { return renderer->base().memory; }

    void allocateBuffers();
    void freeBuffers();
    void allocateTextures();
    void freeTextures();
    void verifyBuffers();

    void keyPressed(int key, int scancode, int mods) override;
};

VulkanExample::VulkanExample()
{
    auto& base = renderer->base();
    for (auto& name : imageNames)
    {
        images.emplace_back(name);
    }
    textureDisplay.init(base, renderer->renderingFormats());

    for (int i = 0; i < 4; ++i)
    {
        allocateBuffers();
        allocateTextures();
    }
}

void VulkanExample::allocateBuffers()
{
    std::uniform_int_distribution<int> countDist(1, 8), sizeDist(1, 64);
    auto count = countDist(random);
    for (int i = 0; i < count; ++i)
    {
        size_t size = sizeDist(random) * 64 * 1024;
        BufferAllocation alloc;
        alloc.first = random() % 100000;

        std::vector<uint32_t> data(size / sizeof(uint32_t));
        std::iota(data.begin(), data.end(), alloc.first);

        alloc.buffer = std::make_unique<Vulkan::Buffer>();
        alloc.buffer->createBuffer(renderer->base(), size, vk::BufferUsageFlagBits::eStorageBuffer,
                                   vk::MemoryPropertyFlagBits::eDeviceLocal);
        alloc.buffer->stagedUpload(renderer->base(), size, data.data());
        alloc.buffer->mark_dynamic();
        buffers.push_back(std::move(alloc));
    }
}

void VulkanExample::freeBuffers()
{
    std::uniform_int_distribution<int> countDist(1, 8);
    auto count = std::min<size_t>(countDist(random), buffers.size());
    for (size_t i = 0; i < count; ++i)
    {
        // The buffer is freed after the frames in flight.
        buffers.erase(buffers.begin() + random() % buffers.size());
    }
}

void VulkanExample::allocateTextures()
{
    std::uniform_int_distribution<int> countDist(1, 4);
    auto count = countDist(random);
    for (int i = 0; i < count; ++i)
    {
        TextureAllocation alloc;
        alloc.texture = std::make_unique<Vulkan::Texture2D>();
        alloc.texture->fromImage(renderer->base(), images[random() % images.size()]);
        alloc.texture->mark_dynamic();
        alloc.descriptor = textureDisplay.createDynamicDescriptorSet();
        alloc.descriptor.assign(0, alloc.texture.get());
        textures.push_back(std::move(alloc));
    }
}

void VulkanExample::freeTextures()
{
    std::uniform_int_distribution<int> countDist(1, 4);
    auto count = std::min<size_t>(countDist(random), textures.size());
    for (size_t i = 0; i < count; ++i)
    {
        // The texture and the descriptor set are freed after the frames in flight.
        textures.erase(textures.begin() + random() % textures.size());
    }
}

void VulkanExample::verifyBuffers()
{
    int errors = 0;
    for (auto& alloc : buffers)
    {
        std::vector<uint32_t> data(alloc.buffer->size() / sizeof(uint32_t));
        alloc.buffer->stagedDownload(data.data());
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (data[i] != alloc.first + i)
            {
                errors++;
                break;
            }
        }
    }
    if (errors == 0)
        std::cout << "Verified " << buffers.size() << " buffers" << std::endl;
    else
        std::cout << errors << " of " << buffers.size() << " buffers have a wrong content!" << std::endl;
}

void VulkanExample::render(vk::CommandBuffer cmd)
{
    if (!showTextures || textures.empty()) return;
    if (textureDisplay.bind(cmd))
    {
        const int columns = 16;
        int index         = 0;
        for (auto& alloc : textures)
        {
            vec2 position((index % columns) * 64, (index / columns) * 64);
            textureDisplay.renderTexture(cmd, alloc.descriptor, position, vec2(64, 64));
            index++;
        }
    }
}

void VulkanExample::renderGUI()
{
    ImGui::SetNextWindowSize(ImVec2(320, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Memory");

    ImGui::TextWrapped(
        "Keys: A/D allocate/free buffers, Z/C allocate/free textures, V verify the buffer contents, J full "
        "defragmentation.");
    ImGui::Text("%zu buffers, %zu textures", buffers.size(), textures.size());

    auto [blocks, used] = memory().dynamicMemoryUsage();
    ImGui::Text("Dynamic memory: %s used of %s", sizeToString(used).c_str(), sizeToString(blocks).c_str());
    ImGui::Checkbox("Incremental defragmentation", &memory().enableDefragmentation);
    auto stats = memory().getDefragmentationStats();
    ImGui::Text("Moved %u allocations (%s), freed %u blocks", stats.allocationsMoved,
                sizeToString(stats.bytesMoved).c_str(), stats.deviceMemoryBlocksFreed);
    ImGui::Checkbox("Show textures", &showTextures);
    ImGui::End();

    VulkanExampleBase::renderGUI();
}

void VulkanExample::keyPressed(int key, int scancode, int mods)
{
    // Don't trigger the hotkeys while typing into an ImGui field.
    if (ImGui::captureKeyboard()) return;

    // The key events are handled between the frames, so the full defragmentation is not in a frame recording.
    switch (key)
    {
        case GLFW_KEY_A:
            allocateBuffers();
            break;
        case GLFW_KEY_D:
            freeBuffers();
            break;
        case GLFW_KEY_Z:
            allocateTextures();
            break;
        case GLFW_KEY_C:
            freeTextures();
            break;
        case GLFW_KEY_V:
            verifyBuffers();
            break;
        case GLFW_KEY_J:
            memory().full_defrag();
            break;
        default:
            // e.g. G (toggle GUI)
            VulkanExampleBase::keyPressed(key, scancode, mods);
            break;
    }
}

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    VulkanExample example;
    example.run();
    return 0;
}
