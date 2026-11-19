/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Vulkan::StandaloneWindow with the headless backend: the main loop renders into offscreen images, ImGui included.
// Every test is skipped without a Vulkan 1.3 driver. On Mesa's software driver lavapipe, as CI does:
//   VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ctest -R test_vulkan_standalone_window

#include "saiga/core/imgui/imgui.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/window/HeadlessWindow.h"
#include "saiga/vulkan/window/StandaloneWindow.h"

#include "gtest/gtest.h"

using namespace Saiga;

namespace
{
const ucvec4 background(200, 30, 40, 255);

// As ints, so that a failure prints numbers instead of characters.
ivec4 Pixel(const TemplatedImage<ucvec4>& image, int row, int col)
{
    return image(row, col).cast<int>();
}

WindowParameters HeadlessParameters()
{
    WindowParameters params;
    params.backend = WindowParameters::Backend::Headless;
    params.width   = 320;
    params.height  = 240;
    // Needs SAIGA_WRITE_PROJECT_DIR_TO_HEADER, as the other tests that load files.
    params.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
    return params;
}

// Draws an ImGui window that fills the display with the background color and a button in its top left corner.
// run() closes the window after a few updates.
class HeadlessApp : public Vulkan::StandaloneWindow, public Vulkan::VulkanForwardRenderingInterface
{
   public:
    HeadlessApp() : StandaloneWindow(HeadlessParameters(), Vulkan::VulkanParameters())
    {
        renderer                      = CreateRenderer<Vulkan::VulkanForwardRenderer>(*this);
        renderer->should_render_imgui = true;
    }

    void update(float dt) override
    {
        if (++updates == 3) window->close();
    }

    void render(vk::CommandBuffer cmd) override { renders++; }

    void renderGUI() override
    {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(background(0) / 255.f, background(1) / 255.f,
                                                        background(2) / 255.f, background(3) / 255.f));
        ImGui::Begin("test", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        if (ImGui::Button("press", ImVec2(100, 50))) presses++;
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
    }

    // Renders one frame and returns it, row 0 at the top.
    TemplatedImage<ucvec4> Frame()
    {
        TemplatedImage<ucvec4> result;
        renderer->requestScreenshot([&](const TemplatedImage<ucvec4>& img) { result = img; });
        Window().renderFrames(1);
        return result;
    }

    Vulkan::HeadlessWindow& Window() { return dynamic_cast<Vulkan::HeadlessWindow&>(*window); }

    Vulkan::VulkanForwardRenderer* renderer = nullptr;
    int updates                             = 0;
    int renders                             = 0;
    int presses                             = 0;
};

class VulkanStandaloneWindowTest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        if (!Vulkan::Instance::suitableDeviceAvailable()) GTEST_SKIP() << "No Vulkan 1.3 device available";
        app = std::make_unique<HeadlessApp>();
    }

    void TearDown() override { app.reset(); }

    std::unique_ptr<HeadlessApp> app;
};

}  // namespace

TEST(VulkanWindowFactory, CreatesTheWindowOfTheBackend)
{
    auto window = Vulkan::CreateVulkanWindow(HeadlessParameters());
    ASSERT_TRUE(window);
    EXPECT_TRUE(dynamic_cast<Vulkan::HeadlessWindow*>(window.get()));
    EXPECT_EQ(window->getWidth(), 320);
    EXPECT_EQ(window->getHeight(), 240);
}

TEST_F(VulkanStandaloneWindowTest, RunsTheMainLoopHeadless)
{
    EXPECT_TRUE(app->renderer->isHeadless());

    TemplatedImage<ucvec4> frame;
    app->renderer->requestScreenshot([&](const TemplatedImage<ucvec4>& img) { frame = img; });

    app->run();

    EXPECT_EQ(app->updates, 3);
    EXPECT_GT(app->renders, 0);
    EXPECT_EQ(frame.width, 320);
    EXPECT_EQ(frame.height, 240);
}

TEST_F(VulkanStandaloneWindowTest, DrawsImGuiIntoTheFrame)
{
    ASSERT_TRUE(ImGui::GetCurrentContext());
    EXPECT_STREQ(ImGui::GetIO().BackendPlatformName, "saiga_headless");
    EXPECT_FALSE(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable);

    // ImGui shows a new window from its second frame on.
    app->Frame();
    auto frame = app->Frame();
    ASSERT_EQ(frame.width, 320);
    ASSERT_EQ(frame.height, 240);
    EXPECT_EQ(Pixel(frame, 200, 300), background.cast<int>());
    EXPECT_EQ(Pixel(frame, 10, 200), background.cast<int>());
    EXPECT_NE(Pixel(frame, 25, 50), background.cast<int>());
}

TEST_F(VulkanStandaloneWindowTest, TakesInputFromImGuisQueue)
{
    app->Frame();
    // One event per frame: ImGui spreads a move and a button change of the same frame over two frames.
    auto click = [this](float x, float y)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.AddMousePosEvent(x, y);
        app->Frame();
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
        app->Frame();
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
        app->Frame();
    };

    click(50, 25);
    EXPECT_EQ(app->presses, 1);

    // Away from the button.
    click(250, 200);
    EXPECT_EQ(app->presses, 1);
}

TEST_F(VulkanStandaloneWindowTest, Resizes)
{
    app->Frame();
    app->Window().setSize(400, 100);
    EXPECT_EQ(app->Window().getWidth(), 400);
    EXPECT_EQ(app->Window().getHeight(), 100);

    auto frame = app->Frame();
    ASSERT_EQ(frame.width, 400);
    ASSERT_EQ(frame.height, 100);
    EXPECT_EQ(ImGui::GetIO().DisplaySize.x, 400);
    EXPECT_EQ(ImGui::GetIO().DisplaySize.y, 100);
    EXPECT_EQ(Pixel(frame, 90, 390), background.cast<int>());
    EXPECT_NE(Pixel(frame, 25, 50), background.cast<int>());
}

TEST_F(VulkanStandaloneWindowTest, KeepsTheClipboardInTheProcess)
{
    app->Frame();
    ImGui::SetClipboardText("copied");
    EXPECT_STREQ(ImGui::GetClipboardText(), "copied");
}
