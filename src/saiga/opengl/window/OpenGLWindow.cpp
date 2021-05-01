/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "OpenGLWindow.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/framework/framework.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/String.h"
#include "saiga/core/util/tostring.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/glImageFormat.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/TextureLoader.h"

#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

namespace Saiga
{
OpenGLWindow::OpenGLWindow(WindowParameters _windowParameters, OpenGLParameters openglParameters)
    : WindowBase(_windowParameters), openglParameters(openglParameters)
{
    initSaigaGL(openglParameters);
}

OpenGLWindow::~OpenGLWindow()
{
    cleanupSaigaGL();
    //    delete renderer;
}


void OpenGLWindow::renderImGui(bool* p_open)
{
    if (!showImgui) return;

    p_open = &showImgui;

    int w = 340;
    int h = 240;
    ImGui::SetNextWindowPos(ImVec2(0, getHeight() - h), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver);
    ImGui::Begin("OpenGLWindow", &showImgui);

    mainLoop.renderImGuiInline();


    ImGui::Text("Camera Position: %s", to_string(getCamera()->getPosition().transpose()).c_str());
    ImGui::Text("Camera Direction: %s", to_string(-make_vec3(getCamera()->getDirection()).transpose()).c_str());
    if (ImGui::Button("Printf camera"))
    {
        std::cout << "camera.position = vec4(" << getCamera()->position(0) << ", " << getCamera()->position(1) << ", "
                  << getCamera()->position(2) << ", " << getCamera()->position(3) << ");" << std::endl;
        std::cout << "camera.rot = " << getCamera()->rot << ";" << std::endl;
        //        createTRSmatrix()
    }

    if (ImGui::Button("Reload Shaders"))
    {
        shaderLoader.reload();
    }

    ImGui::SameLine();

    ImGui::Checkbox("auto reload", &auto_reload_shaders);

    if (ImGui::Button("Screenshot"))
    {
        ScreenshotDefaultFramebuffer().save("screenshot.png");
    }


    ImGui::End();
}

bool OpenGLWindow::create()
{
    initSaiga(windowParameters.saigaParameters);

    // init window and opengl context
    if (!initWindow())
    {
        std::cerr << "Failed to initialize Window!" << std::endl;
        return false;
    }


    loadGLFunctions();
    assert_no_glerror();


    if (!initInput())
    {
        std::cerr << "Failed to initialize Input!" << std::endl;
        return false;
    }


    glDebugMessageCallback(Error::DebugLogConst, NULL);

    assert_no_glerror();

    return true;
}

void OpenGLWindow::destroy()
{
    terminateOpenGL();
    cleanupSaiga();
    freeContext();
}



TemplatedImage<ucvec4> OpenGLWindow::ScreenshotDefaultFramebuffer()
{
    GLint dims[4] = {0};
    glGetIntegerv(GL_VIEWPORT, dims);
    int w = dims[2];
    int h = dims[3];

    TemplatedImage<ucvec4> out;
    out.create(h, w);

    // read data from default framebuffer and restore currently bound fb.
    GLint fb;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glReadPixels(0, 0, out.width, out.height, getGlFormat(out.type), getGlType(out.type), out.data());
    glBindFramebuffer(GL_FRAMEBUFFER, fb);

    out.getImageView().flipY();
    return out;
}



void OpenGLWindow::update(float dt)
{
    checkEvents();
    if (updating) updating->update(dt);

    if (auto_reload_shaders)
    {
        shaderLoader.reload();
    }
}



void OpenGLWindow::swap()
{
    if (windowParameters.finishBeforeSwap) glFinish();
    swapBuffers();
}



}  // namespace Saiga
