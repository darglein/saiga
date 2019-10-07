/**
 * Copyright (c) 2017 Darius Rückert
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


    ImGui::Text("Camera Position: %s", to_string(getCamera()->getPosition()).c_str());
    ImGui::Text("Camera Direction: %s", to_string(-make_vec3(getCamera()->getDirection())).c_str());
    if (ImGui::Button("Printf camera"))
    {
        std::cout << "camera.position = vec4" << getCamera()->position << ";" << std::endl;
        std::cout << "camera.rot = quat" << getCamera()->rot << ";" << std::endl;
        //        createTRSmatrix()
    }

    if (ImGui::Button("Reload Shaders"))
    {
        shaderLoader.reload();
    }

    if (ImGui::Button("Screenshot"))
    {
        screenshot("screenshot.png");
    }

    ImGui::Checkbox("showRendererImgui", &showRendererImgui);

    ImGui::End();

    if (showRendererImgui && renderer)
    {
        renderer->renderImGui();
    }
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

    std::cout << ">> Window inputs initialized!" << std::endl;
    assert_no_glerror();

    return true;
}

void OpenGLWindow::destroy()
{
    terminateOpenGL();
    cleanupSaiga();
    freeContext();
}



void OpenGLWindow::readToExistingImage(Image& out)
{
    // read data from default framebuffer and restore currently bound fb.
    GLint fb;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    //    glReadPixels(0,0,out.width,out.height,GL_RGB,GL_UNSIGNED_BYTE,out.getRawData());

    //    SAIGA_ASSERT(0);
    //    glReadPixels(0,0,out.width,out.height,out.Format().getGlFormat(),out.Format().getGlType(),out.getRawData());
    glReadPixels(0, 0, out.width, out.height, getGlFormat(out.type), getGlType(out.type), out.data());


    glBindFramebuffer(GL_FRAMEBUFFER, fb);
}


void OpenGLWindow::readToImage(Image& out)
{
    //    int w = renderer->outputWidth;
    //    int h = renderer->outputHeight;

    //    out.width = w;
    //    out.height = h;
    //    out.create(h,w,UC3);
    //    out.Format() = ImageFormat(3,8,ImageElementFormat::UnsignedNormalized);
    SAIGA_ASSERT(0);
    //    out.create();

    //    readToExistingImage(out);
}


void OpenGLWindow::screenshot(const std::string& file)
{
    Image img;
    readToImage(img);
    img.save(file);
    //    TextureLoader::instance()->saveImage(file,img);
}

void OpenGLWindow::screenshotRender(const std::string& file)
{
    SAIGA_ASSERT(0);
#if 0
    //    std::cout<<"Window::screenshotRender "<<file<<endl;
    int w = renderer->width;
    int h = renderer->height;

    Image img;
    img.width = w;
    img.height = h;
    //    img.Format() = ImageFormat(3,8,ImageElementFormat::UnsignedNormalized);
    SAIGA_ASSERT(0);
    img.create();

    auto tex = getRenderer()->postProcessor.getCurrentTexture();
    tex->bind();
    glGetTexImage(tex->getTarget(),0,GL_RGB,GL_UNSIGNED_BYTE,img.data());
    tex->unbind();

    //    TextureLoader::instance()->saveImage(file,img);
    img.save(file);
#endif
}

void OpenGLWindow::getDepthFloat(Image& out)
{
    SAIGA_ASSERT(0);
#if 0
    int w = renderer->outputWidth;
    int h = renderer->outputHeight;

    out.width = w;
    out.height = h;
    //    out.Format() = ImageFormat(1,32,ImageElementFormat::FloatingPoint);
    SAIGA_ASSERT(0);
    out.create();


    Image img;
    img.width = w;
    img.height = h;
    //    img.Format() = ImageFormat(4,8,ImageElementFormat::UnsignedNormalized);
    SAIGA_ASSERT(0);
    img.create();


    auto tex = getRenderer()->gbuffer.getTextureDepth();
    tex->bind();
    glGetTexImage(tex->getTarget(),0,GL_DEPTH_STENCIL,GL_UNSIGNED_INT_24_8,img.data());
    tex->unbind();

    for(int i = 0; i < h; ++i)
    {
        for(int j = 0; j < w; ++j)
        {
#    if 0
            unsigned int v = img.getPixel<unsigned int>(j,i);
            //override stencil bits with 0
            v = v & 0xFFFFFF00;
            float d = uintToNFloat(v);
            out.getPixel<float>(j,i) = d;
#    endif
        }
    }
#endif
}

void OpenGLWindow::screenshotRenderDepth(const std::string& file)
{
    SAIGA_ASSERT(0);
#if 0
    //    std::cout<<"Window::screenshotRender "<<file<<endl;
    int w = renderer->width;
    int h = renderer->height;

    Image img;
    getDepthFloat(img);

    Image img2;
    img2.width = w;
    img2.height = h;
    //    img2.Format() = ImageFormat(1,16,ImageElementFormat::UnsignedNormalized);
    SAIGA_ASSERT(0);
    img2.create();

    for(int i = 0; i < h; ++i)
    {
        for(int j = 0; j < w; ++j)
        {
#    if 0
            float d = img.getPixel<float>(j,i);
            d = currentCamera->linearDepth(d);
            img2.getPixel<unsigned short>(j,i) = d * 0xFFFF;
#    endif
        }
    }




    //    TextureLoader::instance()->saveImage(file,img2);
    img2.save(file);
#endif
}


void OpenGLWindow::update(float dt)
{
    checkEvents();
    if (updating) updating->update(dt);
}



void OpenGLWindow::swap()
{
    if (windowParameters.finishBeforeSwap) glFinish();
    swapBuffers();
}



}  // namespace Saiga
