/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// StandaloneWindow with the headless backend: an OffscreenWindow with ImGui and no window system.
// Every test is skipped without an EGL display. On Mesa's software renderer, as CI does:
//   __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json ctest -R test_opengl_window

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_main_menu.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/rendering/empty_renderer.h"
#include "saiga/opengl/window/StandaloneWindow.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>

using namespace Saiga;

namespace
{
const ucvec4 background(200, 30, 40, 255);

// As ints, so that a failure prints numbers instead of characters.
ivec4 Pixel(const TemplatedImage<ucvec4>& image, int row, int col)
{
    return image(row, col).cast<int>();
}

// Draws a window that fills the display with the background color and a button in its top left corner.
class HeadlessApp : public StandaloneWindow
{
   public:
    explicit HeadlessApp(const WindowParameters& windowParameters)
        : StandaloneWindow(windowParameters, OpenGLParameters())
    {
        renderer = CreateRenderer<EmptyRenderer>();
        // The editor layout and the log window would cover the test's window.
        editor_gui.enabled    = false;
        console.should_render = false;
    }

    void render(RenderInfo render_info) override
    {
        if (render_info.render_pass != RenderPass::GUI) return;

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

    // Renders one frame into the default framebuffer and returns it, row 0 at the top.
    TemplatedImage<ucvec4> Frame()
    {
        window->render();
        window->swap();
        glViewport(0, 0, window->getWidth(), window->getHeight());
        return window->ScreenshotDefaultFramebuffer();
    }

    OpenGLWindow& Window() { return *window; }

    EmptyRenderer* renderer = nullptr;
    int presses             = 0;
};

class OpenGLWindowTest : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        WindowParameters params;
        params.backend = WindowParameters::Backend::Headless;
        params.width   = 320;
        params.height  = 240;
        // Needs SAIGA_WRITE_PROJECT_DIR_TO_HEADER, as the other tests that load files.
        params.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
        try
        {
            app = std::make_unique<HeadlessApp>(params);
        }
        catch (const std::runtime_error& e)
        {
            GTEST_SKIP() << e.what();
        }
        // ImGui shows a new window from its second frame on.
        app->Frame();
    }

    void TearDown() override
    {
        // As the OpenGLTest fixture: a test must not leave a GL error behind.
        if (app) EXPECT_EQ(glGetError(), GL_NO_ERROR);
        app.reset();
    }

    std::unique_ptr<HeadlessApp> app;
};

}  // namespace

TEST_F(OpenGLWindowTest, CreatesAHeadlessWindow)
{
    OpenGLWindow& window = app->Window();
    EXPECT_TRUE(window.isCreated());
    EXPECT_TRUE(window.isShown());
    EXPECT_EQ(window.getWidth(), 320);
    EXPECT_EQ(window.getHeight(), 240);
    EXPECT_EQ(window.getContentScale(), 1.0f);
    ASSERT_TRUE(app->renderer->imgui);
    EXPECT_STREQ(ImGui::GetIO().BackendPlatformName, "saiga_headless");
    // Viewports would need a platform backend that opens OS windows.
    EXPECT_FALSE(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable);

    window.setWindowTitle("title");
    EXPECT_EQ(window.getWindowTitle(), "title");
}

TEST_F(OpenGLWindowTest, RendersImGuiIntoTheDefaultFramebuffer)
{
    auto frame = app->Frame();
    ASSERT_EQ(frame.width, 320);
    ASSERT_EQ(frame.height, 240);
    // Away from the button.
    EXPECT_EQ(Pixel(frame, 200, 300), background.cast<int>());
    EXPECT_EQ(Pixel(frame, 10, 200), background.cast<int>());
    // The button has its own color.
    EXPECT_NE(Pixel(frame, 25, 50), background.cast<int>());
}

TEST_F(OpenGLWindowTest, TakesInputFromImGuisQueue)
{
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

TEST_F(OpenGLWindowTest, Resizes)
{
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

TEST_F(OpenGLWindowTest, KeepsTheClipboardInTheProcess)
{
    ImGui::SetClipboardText("copied");
    EXPECT_STREQ(ImGui::GetClipboardText(), "copied");
}

#if defined(__linux__)
// Regression: a GLFW window that failed to initialize asserted in its destructor (terminateOpenGL without a context),
// so StandaloneWindow could not report the failure.
TEST(OpenGLWindowFactory, ReturnsNullWithoutADisplay)
{
    std::string display         = std::getenv("DISPLAY") ? std::getenv("DISPLAY") : "";
    std::string wayland_display = std::getenv("WAYLAND_DISPLAY") ? std::getenv("WAYLAND_DISPLAY") : "";
    unsetenv("DISPLAY");
    unsetenv("WAYLAND_DISPLAY");

    WindowParameters params;
    params.backend = WindowParameters::Backend::GLFW;
    params.saigaParameters.shaderDirectory.push_back(SAIGA_PROJECT_SOURCE_DIR "/shader");
    EXPECT_FALSE(CreateOpenGLWindow(params, OpenGLParameters()));

    if (!display.empty()) setenv("DISPLAY", display.c_str(), 1);
    if (!wayland_display.empty()) setenv("WAYLAND_DISPLAY", wayland_display.c_str(), 1);
}
#endif

TEST(WindowParameters, ReadsTheBackendFromTheConfigFile)
{
    const char* file = "test_opengl_window_config.ini";
    auto read        = [file](const std::string& value, WindowParameters::Backend before)
    {
        {
            std::ofstream out(file);
            out << "[window]\nwindow_backend = " << value << "\n";
        }
        WindowParameters params;
        params.backend = before;
        params.fromConfigFile(file);
        return params.backend;
    };

    EXPECT_EQ(read("headless", WindowParameters::Backend::GLFW), WindowParameters::Backend::Headless);
    EXPECT_EQ(read("Headless", WindowParameters::Backend::GLFW), WindowParameters::Backend::Headless);
    EXPECT_EQ(read("glfw", WindowParameters::Backend::Headless), WindowParameters::Backend::GLFW);
    // An unknown name keeps the backend.
    EXPECT_EQ(read("egl", WindowParameters::Backend::Headless), WindowParameters::Backend::Headless);
    std::remove(file);
}
