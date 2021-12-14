/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/opengl/rendering/renderer.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui_main_menu.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/opengl/glImageFormat.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/window/OpenGLWindow.h"

namespace Saiga
{
OpenGLRenderer::OpenGLRenderer(OpenGLWindow& window) : window(&window)
{
    editor_gui.enabled = true;
    main_menu.AddItem(
        "Saiga", "Log", []() { console.should_render = !console.should_render; }, 299, "F10");

    cameraBuffer.createGLBuffer(nullptr, sizeof(CameraDataGLSL), GL_DYNAMIC_DRAW);


    ResizeTarget(window.getWidth(), window.getHeight());

    window.setRenderer(this);
    default_framebuffer.MakeDefaultFramebuffer();
    // ImGUI
    imgui = window.createImGui();
    timer = std::make_shared<GLTimerSystem>();
}

OpenGLRenderer::~OpenGLRenderer() {}

void OpenGLRenderer::render(const RenderInfo& renderInfo)
{
    SAIGA_ASSERT(target_framebuffer);


    // Size and position of the 3D viewport
    // In editor mode this will be set by the imgui-window
    viewport_size        = ivec2(window->getWidth(), window->getHeight());
    viewport_offset      = ivec2(0, 0);
    bool render_viewport = true;

    PrepareImgui();

    if (render_viewport)
    {
        // 2. Render 3DView to framebuffer
        ResizeTarget(viewport_size.x(), viewport_size.y());
        Camera* camera = renderInfo.camera;
        camera->recomputeProj(outputWidth, outputHeight);
        ViewPort viewport = ViewPort({0, 0}, {outputWidth, outputHeight});
        auto target_fb    = editor_gui.enabled ? target_framebuffer.get() : &default_framebuffer;
        target_fb->bind();
        glClear(GL_COLOR_BUFFER_BIT);

        SAIGA_ASSERT(timer);
        timer->BeginFrame();
        renderGL(target_fb, viewport, camera);
        timer->EndFrame();
    }

    // 3. Add rendered 3DView to imgui (in editor mode only)
    FinalizeImgui();

#ifdef SAIGA_USE_FFMPEG
    if (encoder && encoder->isEncoding())
    {
        // feed encoder with 3d viewport
        auto encoder_size = encoder->Size();

        TemplatedImage<ucvec4> result(encoder_size.y(), encoder_size.x());

        if (editor_gui.enabled && record_view_port_only)
        {
            TemplatedImage<ucvec4> tmp(outputHeight, outputWidth);
            auto texture = target_framebuffer->getTextureColor(0);
            texture->download(tmp.data());
            tmp.getImageView().subImageView(0, 0, encoder_size.y(), encoder_size.x()).copyTo(result.getImageView());
        }
        else
        {
            // read data from default framebuffer and restore currently bound fb.
            GLint fb;
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fb);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glReadPixels(0, 0, result.width, result.height, getGlFormat(result.type), getGlType(result.type),
                         result.data());
            glBindFramebuffer(GL_FRAMEBUFFER, fb);
        }


        encoder->frame(result.getImageView().yFlippedImageView());
    }
#endif
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
        console << "Warning: The window size must be greater than zero." << std::endl;
        windowWidth  = std::max(windowWidth, 1);
        windowHeight = std::max(windowHeight, 1);
    }

    outputWidth  = windowWidth;
    outputHeight = windowHeight;

#ifdef SAIGA_USE_FFMPEG
    // The encoder size must be divisible by 2 for most of the codecs to work
    int encoder_w = iAlignDown(outputWidth, 2);
    int encoder_h = iAlignDown(outputHeight, 2);

    if (!encoder)
    {
        encoder = std::make_shared<VideoEncoder>(encoder_w, encoder_h);
    }
    else
    {
        encoder->resize(encoder_w, encoder_h);
    }
#endif

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

    console << "[Renderer] Target resized to " << windowWidth << "x" << windowHeight << std::endl;
}


void OpenGLRenderer::bindCamera(Camera* cam)
{
    CameraDataGLSL cd(cam);
    cameraBuffer.updateBuffer(&cd, sizeof(CameraDataGLSL), 0);
    cameraBuffer.bind(CAMERA_DATA_BINDING_POINT);
}

void OpenGLRenderer::PrepareImgui(bool compute_viewport_size)
{
    // 1. Render the imgui
    //      - If we are in editor mode this will also tell us the 3DView window-size
    if (imgui)
    {
        SAIGA_ASSERT(ImGui::GetCurrentContext());


        imgui->beginFrame();
        window->renderImGui();

        RenderInfo ri;
        ri.camera      = nullptr;
        ri.render_pass = RenderPass::GUI;
        dynamic_cast<RenderingInterface*>(rendering)->render(ri);
        renderImgui();
        console.render();
        timer->Imgui();
#ifdef SAIGA_USE_FFMPEG
        encoder->renderGUI();
#endif

        if (editor_gui.enabled)
        {
            ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

            use_mouse_input_in_3dview    = false;
            use_keyboard_input_in_3dview = false;

            if (ImGui::Begin("3DView", nullptr, flags))
            {
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 1));
                ImGui::BeginChild("viewer_child", ImVec2(0, 0), false,
                                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove);
                ImGui::PopStyleColor();

                use_mouse_input_in_3dview = ImGui::IsWindowHovered();
                use_keyboard_input_in_3dview =
                    use_mouse_input_in_3dview || (ImGui::IsWindowFocused() && !ImGui::captureKeyboard());

                if (compute_viewport_size)
                {
                    viewport_offset.x() = ImGui::GetCursorPosX() + ImGui::GetWindowPos().x;
                    viewport_offset.y() = ImGui::GetCursorPosY() + ImGui::GetWindowPos().y;

                    auto w_size   = ImGui::GetWindowContentRegionMax();
                    viewport_size = ivec2(w_size.x, w_size.y);
                }
                ImGui::EndChild();
            }
            else
            {
                render_viewport = false;
            }
            ImGui::End();
        }
    }

    if (!editor_gui.enabled)
    {
        // In Fullscreen mode we check, if a gui element is used
        use_mouse_input_in_3dview    = !ImGui::captureMouse();
        use_keyboard_input_in_3dview = !ImGui::captureKeyboard();
    }
}

void OpenGLRenderer::FinalizeImgui()
{
    if (imgui)
    {
        if (editor_gui.enabled && render_viewport)
        {
            ImGui::Begin("3DView");
            ImGui::BeginChild("viewer_child");
            ImGui::Texture(target_framebuffer->getTextureColor(0).get(), ImVec2(viewport_size.x(), viewport_size.y()),
                           true);
            ImGui::EndChild();
            ImGui::End();
        }
        // The imgui frame is now done
        // -> Render it to the screen (default FB)
        imgui->endFrame();
        default_framebuffer.bind();
        imgui->render();
    }
}
void OpenGLRenderer::renderImgui()
{
    RendererBase::renderImgui();

    if (ImGui::Button("screenshot viewport"))
    {
        TemplatedImage<ucvec4> img(outputHeight, outputWidth);
        target_framebuffer->getTextureColor(0)->download(img.data8());
        img.getImageView().flipY();
        img.save("screenshot_3dview.png");
    }
}
void RenderingParameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    //    auto section = "Rendering";

    if (ini.changed()) ini.SaveFile(file.c_str());
}


}  // namespace Saiga
