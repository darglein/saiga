/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/renderer.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui_main_menu.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/window/OpenGLWindow.h"

namespace Saiga
{
OpenGLRenderer::OpenGLRenderer(OpenGLWindow& window) : window(&window)
{
    cameraBuffer.createGLBuffer(nullptr, sizeof(CameraDataGLSL), GL_DYNAMIC_DRAW);


    ResizeTarget(window.getWidth(), window.getHeight());

    window.setRenderer(this);
    default_framebuffer.MakeDefaultFramebuffer();
    // ImGUI
    imgui = window.createImGui();
}

OpenGLRenderer::~OpenGLRenderer() {}

void OpenGLRenderer::render(const RenderInfo& renderInfo)
{
    SAIGA_ASSERT(renderInfo.cameras.size() == 1);
    SAIGA_ASSERT(target_framebuffer);


    int target_w = window->getWidth();
    int target_h = window->getHeight();


    // 1. Render the imgui
    //      - If we are in editor mode this will also tell us the 3DView window-size
    if (imgui)
    {
        SAIGA_ASSERT(ImGui::GetCurrentContext());


        imgui->beginFrame();
        window->renderImGui();
        dynamic_cast<RenderingInterface*>(rendering)->render(nullptr, RenderPass::GUI);
        renderImgui();

        if (editor_gui.enabled)
        {
            ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
            ImGui::Begin("3DView", nullptr, flags);

            use_mouse_input_in_3dview    = ImGui::IsWindowHovered(ImGuiHoveredFlags_RootWindow);
            use_keyboard_input_in_3dview = use_mouse_input_in_3dview;

            auto w_size = ImGui::GetWindowContentRegionMax();
            target_w    = w_size.x;
            target_h    = w_size.y;
            ImGui::End();
        }
    }

    if (!editor_gui.enabled)
    {
        // In fullscreen mode we check, if a gui element is used
        use_mouse_input_in_3dview    = !ImGui::captureMouse();
        use_keyboard_input_in_3dview = !ImGui::captureKeyboard();
    }

    ResizeTarget(target_w, target_h);

    SAIGA_ASSERT(target_w == outputWidth && target_h == outputHeight);

    // 2. Render 3DView to framebuffer
    Camera* camera    = renderInfo.cameras.front().first;
    ViewPort viewport = ViewPort({0, 0}, {outputWidth, outputHeight});
    auto target_fb    = editor_gui.enabled ? target_framebuffer.get() : &default_framebuffer;
    target_fb->bind();
    glClear(GL_COLOR_BUFFER_BIT);
    renderGL(target_fb, viewport, camera);


    // 3. Add rendered 3DView to imgui (in editor mode only)
    if (imgui)
    {
        if (editor_gui.enabled)
        {
            ImGui::Begin("3DView");
            ImGui::Texture(target_framebuffer->getTextureColor(0).get(), ImGui::GetWindowSize(), true);
            ImGui::End();
        }
        // The imgui frame is now done
        // -> Render it to the screen (default FB)
        imgui->endFrame();
        default_framebuffer.bind();
        imgui->render();
    }
}

void OpenGLRenderer::ResizeTarget(int windowWidth, int windowHeight)
{
    if (windowWidth == outputWidth && windowHeight == outputHeight)
    {
        // Already at correct size
        // -> Skip resize
        return;
    }

    if (windowWidth <= 0 || windowHeight <= 0)
    {
        std::cerr << "Warning: The window size must be greater than zero." << std::endl;
        windowWidth  = std::max(windowWidth, 1);
        windowHeight = std::max(windowHeight, 1);
    }

    outputWidth  = windowWidth;
    outputHeight = windowHeight;

    if (!target_framebuffer)
    {
        target_framebuffer = std::make_unique<Framebuffer>();
        target_framebuffer->create();

        std::shared_ptr<Texture> color = std::make_shared<Texture>();
        color->create(outputWidth, outputHeight, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);

        std::shared_ptr<Texture> depth_stencil = std::make_shared<Texture>();
        depth_stencil->create(outputWidth, outputHeight, GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, GL_UNSIGNED_INT_24_8);

        target_framebuffer->attachTexture(color);
        target_framebuffer->attachTextureDepthStencil(depth_stencil);
        target_framebuffer->check();
    }
    else
    {
        target_framebuffer->resize(outputWidth, outputHeight);
    }

    std::cout << "OpenGLRenderer::resize -> " << windowWidth << "x" << windowHeight << std::endl;
}


void OpenGLRenderer::bindCamera(Camera* cam)
{
    CameraDataGLSL cd(cam);
    cameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
    cameraBuffer.bind(CAMERA_DATA_BINDING_POINT);
}

void RenderingParameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    //    auto section = "Rendering";

    if (ini.changed()) ini.SaveFile(file.c_str());
}


}  // namespace Saiga
