/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/vulkan/FrameTimings.h"
#include "saiga/vulkan/VulkanAsset.h"
#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"

#include "vulkan_test_context.h"

using namespace Saiga;

// The forward renderer with a HeadlessWindow: it renders into offscreen images, which are read back.
class VulkanHeadlessRenderer : public VulkanTest
{
};

// Regression (CodeRabbit, S13): durations were computed from converted timestamps without the mask of the valid bits,
// so a section across a wrap of a 36-bit counter became huge (no device needed)
TEST(FrameTimings, TimestampDifferenceAcrossAWrap)
{
    const uint64_t mask = (uint64_t(1) << 36) - 1;
    EXPECT_DOUBLE_EQ(Vulkan::timestampDifferenceMs(mask - 9, 10, mask, 1.0), 20 * 1e-6);
    EXPECT_DOUBLE_EQ(Vulkan::timestampDifferenceMs(1000, 3000, mask, 2.0), 4000 * 1e-6);
}

TEST_F(VulkanHeadlessRenderer, ClearColor)
{
    auto img = RenderFrame();
    ASSERT_EQ(img.w, width);
    ASSERT_EQ(img.h, height);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) ASSERT_EQ(img(y, x), ClearColor()) << "at " << x << "," << y;
}

// More frames than frames in flight, so every offscreen image is reused.
TEST_F(VulkanHeadlessRenderer, ManyFrames)
{
    int rendered   = 0;
    app.onRender   = [&](vk::CommandBuffer) { rendered++; };
    int transfered = 0;
    app.onTransfer = [&](vk::CommandBuffer) { transfered++; };

    for (int i = 0; i < 5; ++i)
    {
        auto img = RenderFrame();
        ASSERT_EQ(img.w, width);
        EXPECT_EQ(img(0, 0), ClearColor());
    }
    window->renderFrames(4);

    EXPECT_EQ(rendered, 9);
    EXPECT_EQ(transfered, 9);
    EXPECT_EQ(app.updates, 9);
}

TEST_F(VulkanHeadlessRenderer, TextureDisplay)
{
    Vulkan::Texture2D texture;
    {
        TemplatedImage<ucvec4> img(4, 4);
        img.getImageView().set(ucvec4(255, 0, 0, 255));
        texture.fromImage(base(), img);
    }

    Vulkan::TextureDisplay display;
    display.init(base(), renderer->renderingFormats());
    auto set = display.createAndUpdateDescriptorSet(texture);

    // A red rectangle at [8,24) x [4,20)
    bool bound   = false;
    app.onRender = [&](vk::CommandBuffer cmd)
    {
        bound = display.bind(cmd);
        if (bound) display.renderTexture(cmd, set, vec2(8, 4), vec2(16, 16));
    };

    auto img = RenderFrame();
    ASSERT_TRUE(bound);
    ASSERT_EQ(img.w, width);

    ucvec4 red(255, 0, 0, 255);
    EXPECT_EQ(img(4, 8), red);
    EXPECT_EQ(img(12, 16), red);
    EXPECT_EQ(img(19, 23), red);

    EXPECT_EQ(img(3, 8), ClearColor());
    EXPECT_EQ(img(4, 7), ClearColor());
    EXPECT_EQ(img(20, 23), ClearColor());
    EXPECT_EQ(img(19, 24), ClearColor());
    EXPECT_EQ(img(40, 50), ClearColor());
}

TEST_F(VulkanHeadlessRenderer, ColoredAsset)
{
    // A triangle around the center in clip space (identity view and projection). Both windings, so it is visible
    // independent of the culling.
    Vulkan::VulkanVertexColoredAsset triangle;
    for (vec3 p : {vec3(-0.5f, -0.5f, 0.5f), vec3(0.5f, -0.5f, 0.5f), vec3(0, 0.5f, 0.5f)})
    {
        VertexNC v(p, vec3(0, 0, 1), vec3(0, 1, 0));
        triangle.addVertex(v);
    }
    triangle.addFace(0, 1, 2);
    triangle.addFace(0, 2, 1);
    triangle.init(base());

    Vulkan::AssetRenderer assetRenderer;
    assetRenderer.init(base(), renderer->renderingFormats());

    app.onTransfer = [&](vk::CommandBuffer cmd)
    { assetRenderer.updateUniformBuffers(cmd, mat4::Identity(), mat4::Identity()); };

    bool bound   = false;
    app.onRender = [&](vk::CommandBuffer cmd)
    {
        bound = assetRenderer.bind(cmd);
        if (bound)
        {
            assetRenderer.pushModel(cmd, mat4::Identity());
            triangle.render(cmd);
        }
    };

    auto img = RenderFrame();
    ASSERT_TRUE(bound);
    ASSERT_EQ(img.w, width);

    // The shaded color depends on the lighting, but green is the only channel of the vertex color.
    ucvec4 center = img(height / 2, width / 2);
    EXPECT_NE(center, ClearColor());
    EXPECT_GT(center(1), center(0));
    EXPECT_GT(center(1), center(2));

    EXPECT_EQ(img(0, 0), ClearColor());
    EXPECT_EQ(img(height - 1, 0), ClearColor());
    EXPECT_EQ(img(0, width - 1), ClearColor());
    EXPECT_EQ(img(height - 1, width - 1), ClearColor());
}

// S36 (A8): ImGui::Image of a texture added to the ImGui renderer appears in the frame (each draw command binds the
// descriptor set of its texture ID; the font keeps its own); a removed texture is released through deferDelete and the
// frames go on without it
namespace
{
struct ImageGuiApp : public Vulkan::VulkanForwardRenderingInterface, public Updating
{
    ImTextureID texture = ImTextureID_Invalid;
    void render(vk::CommandBuffer) override {}
    void update(float) override {}
    void renderGUI() override
    {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::Begin("image", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        ImGui::Text("text");
        if (texture != ImTextureID_Invalid)
        {
            ImGui::SetCursorPos(ImVec2(16, 16));
            ImGui::Image(texture, ImVec2(16, 16));
        }
        ImGui::End();
        ImGui::PopStyleVar(2);
    }
};
}  // namespace

TEST(VulkanImGui, ImageDraws)
{
    if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
    WindowParameters wp;
    wp.width  = 64;
    wp.height = 48;
    wp.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
    wp.imguiParameters.enable = true;
    Vulkan::VulkanParameters vp;
    vp.enableImgui = true;
    auto window    = std::make_unique<Vulkan::HeadlessWindow>(wp);
    auto renderer  = std::make_unique<Vulkan::VulkanForwardRenderer>(*window, vp);
    ImageGuiApp app;
    renderer->setRenderObject(app);
    window->setUpdateObject(app);
    renderer->should_render_imgui      = true;
    Vulkan::ImGuiVulkanRenderer* imgui = renderer->imguiRenderer();
    ASSERT_NE(imgui, nullptr);

    Vulkan::Texture2D texture;
    {
        TemplatedImage<ucvec4> img(4, 4);
        img.getImageView().set(ucvec4(255, 0, 0, 255));
        texture.fromImage(renderer->base(), img);
    }
    const vk::DescriptorImageInfo info = texture.getDescriptorInfo();
    app.texture                        = imgui->addTexture(info.imageView, info.sampler, info.imageLayout);
    EXPECT_NE(app.texture, Vulkan::ImGuiVulkanRenderer::kFontTexture);
    EXPECT_EQ(imgui->textureCount(), 1u);

    auto frame = [&]()
    {
        TemplatedImage<ucvec4> result;
        renderer->requestScreenshot([&](const TemplatedImage<ucvec4>& img) { result = img; });
        window->renderFrames(1);
        return result;
    };
    // ImGui shows a new window from its second frame on
    frame();
    auto img = frame();
    ASSERT_EQ(img.w, 64);
    const ucvec4 red(255, 0, 0, 255);
    EXPECT_EQ(img(24, 24), red);
    EXPECT_EQ(img(17, 17), red);
    EXPECT_NE(img(40, 40), red);
    EXPECT_NE(img(24, 40), red);

    // Removed: the command of the unknown ID is not drawn, the frames continue (the set is freed after its frames)
    imgui->removeTexture(app.texture);
    EXPECT_EQ(imgui->textureCount(), 0u);
    auto after = frame();
    ASSERT_EQ(after.w, 64);
    EXPECT_NE(after(24, 24), red);
    app.texture = ImTextureID_Invalid;
    frame();
    renderer->waitIdle();
    texture.destroy();
}
