/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * Compute shaders with the Vulkan module:
 *  - Once at startup: a compute shader writes into a storage buffer, which is read back and printed.
 *  - Every frame: a compute shader animates a pattern in a storage image, which is then displayed.
 */

#include "saiga/core/image/managedImage.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/pipeline/DescriptorSet.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/texture/Texture.h"
#include "saiga/vulkan/window/GLFWSample.h"

#include <cstring>
#include <saiga/core/imgui/imgui.h>

using namespace Saiga;

class Compute : public VulkanExampleBase
{
   public:
    Compute();
    ~Compute() override;

    void update(float dt) override;
    void transfer(vk::CommandBuffer cmd) override;
    void render(vk::CommandBuffer cmd) override;
    void renderGUI() override;

   private:
    void bufferTest(Vulkan::VulkanBase& base);

    static constexpr int imageSize = 512;
    float time                     = 0;
    bool animate                   = true;

    Vulkan::Texture2D storageImage;
    Vulkan::ComputePipeline patternPipeline;
    Vulkan::StaticDescriptorSet patternSet;

    Vulkan::TextureDisplay textureDisplay;
    Vulkan::StaticDescriptorSet displaySet;
};

Compute::Compute()
{
    auto& base = renderer->base();

    bufferTest(base);

    // The storage image: written by the compute shader, sampled by the texture display. It stays in the GENERAL
    // layout, which supports both.
    {
        TemplatedImage<ucvec4> img(imageSize, imageSize);
        img.getImageView().set(ucvec4(0, 0, 0, 255));
        storageImage.fromImage(base, img, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);

        vk::CommandBuffer cmd = base.mainQueue.commandPool.createAndBeginOneTimeBuffer();
        storageImage.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);
        cmd.end();
        base.mainQueue.submitAndWait(cmd);
        base.mainQueue.commandPool.freeCommandBuffer(cmd);
    }

    patternPipeline.init(base, 1);
    patternPipeline.addDescriptorSetLayout(
        {{0, {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}}});
    patternPipeline.addPushConstantRange({vk::ShaderStageFlagBits::eCompute, 0, sizeof(float)});
    patternPipeline.shaderPipeline.loadCompute(base.device, "vulkan/compute_pattern.comp");
    patternPipeline.create();

    patternSet     = patternPipeline.createDescriptorSet();
    auto imageInfo = storageImage.getDescriptorInfo();
    base.device.updateDescriptorSets(
        vk::WriteDescriptorSet(patternSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &imageInfo, nullptr, nullptr),
        nullptr);

    textureDisplay.init(base, renderer->renderingFormats());
    displaySet = textureDisplay.createAndUpdateDescriptorSet(storageImage);
}

Compute::~Compute() {}

void Compute::bufferTest(Vulkan::VulkanBase& base)
{
    // test.comp: numbers[i] = i, numbers[0] += 10 for each of the 10 invocations.
    std::vector<int> data(10, 1);
    Vulkan::Buffer storageBuffer;
    storageBuffer.createBuffer(base, sizeof(int) * data.size(), vk::BufferUsageFlagBits::eStorageBuffer,
                               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    storageBuffer.upload(data.data(), data.size() * sizeof(int));

    // test.comp also writes one pixel of a storage image.
    Vulkan::Texture2D texture;
    {
        TemplatedImage<ucvec4> img(16, 16);
        img.getImageView().set(ucvec4(0, 0, 255, 255));
        texture.fromImage(base, img, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);
    }

    Vulkan::ComputePipeline pipeline;
    pipeline.init(base, 1);
    pipeline.addDescriptorSetLayout(
        {{0, {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
         {1, {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}}});
    pipeline.shaderPipeline.loadCompute(base.device, "vulkan/test.comp");
    pipeline.create();

    vk::CommandBuffer cmd = base.computeQueue->commandPool.createAndBeginOneTimeBuffer();
    texture.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);

    auto set              = pipeline.createDescriptorSet();
    auto bufferInfo       = storageBuffer.createInfo();
    auto imageInfo        = texture.getDescriptorInfo();
    imageInfo.imageLayout = vk::ImageLayout::eGeneral;
    base.device.updateDescriptorSets(
        {
            vk::WriteDescriptorSet(set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &bufferInfo, nullptr),
            vk::WriteDescriptorSet(set, 1, 0, 1, vk::DescriptorType::eStorageImage, &imageInfo, nullptr, nullptr),
        },
        nullptr);

    if (pipeline.bind(cmd))
    {
        pipeline.bindDescriptorSet(cmd, set);
        cmd.dispatch(1, 1, 1);
    }
    // Make the shader writes visible to the host read below.
    vk::MemoryBarrier2 toHost(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
                              vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, toHost, nullptr, nullptr));
    cmd.end();
    base.computeQueue->submitAndWait(cmd);
    base.computeQueue->commandPool.freeCommandBuffer(cmd);

    storageBuffer.download(data.data(), data.size() * sizeof(int));
    std::cout << "Compute buffer test:";
    for (int i : data) std::cout << " " << i;
    std::cout << std::endl;
}

void Compute::update(float dt)
{
    if (animate) time += dt;
    VulkanExampleBase::update(dt);
}

void Compute::transfer(vk::CommandBuffer cmd)
{
    using Stage     = vk::PipelineStageFlagBits2;
    using Access    = vk::AccessFlagBits2;
    auto range      = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    vk::Image image = storageImage.memoryLocation->data.image;

    // The previous frame's texture display reads the image: write after that (write-after-read).
    vk::ImageMemoryBarrier2 beforeWrite(Stage::eFragmentShader, {}, Stage::eComputeShader, Access::eShaderStorageWrite,
                                        vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, VK_QUEUE_FAMILY_IGNORED,
                                        VK_QUEUE_FAMILY_IGNORED, image, range);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, beforeWrite));

    if (patternPipeline.bind(cmd))
    {
        patternPipeline.bindDescriptorSet(cmd, patternSet);
        patternPipeline.pushConstant(cmd, vk::ShaderStageFlagBits::eCompute, sizeof(float), &time);
        cmd.dispatch((imageSize + 15) / 16, (imageSize + 15) / 16, 1);
    }

    // The texture display samples the image in this frame.
    vk::ImageMemoryBarrier2 afterWrite(Stage::eComputeShader, Access::eShaderStorageWrite, Stage::eFragmentShader,
                                       Access::eShaderSampledRead, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                                       VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, afterWrite));
}

void Compute::render(vk::CommandBuffer cmd)
{
    if (textureDisplay.bind(cmd))
    {
        float s = 512;
        textureDisplay.renderTexture(cmd, displaySet, vec2(400, 100), vec2(s, s));
    }
}

void Compute::renderGUI()
{
    ImGui::SetNextWindowSize(ImVec2(250, 100), ImGuiCond_FirstUseEver);
    ImGui::Begin("Compute");
    ImGui::Checkbox("Animate", &animate);
    ImGui::Text("Time %.2f", time);
    ImGui::End();
    VulkanExampleBase::renderGUI();
}

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Compute example;
    example.run();
    return 0;
}
