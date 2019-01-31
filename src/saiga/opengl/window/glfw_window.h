/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/config.h"
#include "saiga/core/glfw/glfw_eventhandler.h"
#include "saiga/core/glfw/glfw_joystick.h"
#include "saiga/opengl/window/OpenGLWindow.h"

#include <array>
#include <list>
#include <mutex>
#include <thread>

struct GLFWwindow;
struct GLFWcursor;

namespace Saiga
{
class Image;

class SAIGA_GLOBAL glfw_Window : public OpenGLWindow, public glfw_ResizeListener
{
   public:
    GLFWwindow* window = nullptr;

    bool recordingVideo = false;

    glfw_Window(WindowParameters windowParameters, OpenGLParameters openglParameters);
    virtual ~glfw_Window();


    void setCursorPosition(int x, int y);
    void showMouseCursor();
    void hideMouseCursor();
    void disableMouseCursor();
    void setGLFWcursor(GLFWcursor* cursor);
    GLFWcursor* createGLFWcursor(Image* image, int midX, int midY);
    void setWindowIcon(Image* image);

    virtual bool window_size_callback(GLFWwindow* window, int width, int height) override;

   protected:
    virtual bool initWindow() override;
    virtual bool initInput() override;
    virtual bool shouldClose() override;
    virtual void checkEvents() override;
    virtual void swapBuffers() override;
    virtual void freeContext() override;
    virtual void loadGLFunctions() override;

   public:
    void destroy();
    // static glfw stuff
    static void error_callback(int error, const char* description);
    static bool initGlfw();
    static void getCurrentPrimaryMonitorResolution(int* width, int* height);
    static void getMaxResolution(int* width, int* height);

    virtual std::shared_ptr<ImGui_GL_Renderer> createImGui() override;
};

}  // namespace Saiga
