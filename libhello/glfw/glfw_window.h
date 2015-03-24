#pragma once


#include "libhello/window/window.h"

#include <GLFW/glfw3.h>

#include "libhello/glfw/glfw_eventhandler.h"


class glfw_Window : public Window{
protected:
    GLFWwindow* window;

    bool initWindow();
    bool initInput();
public:

    glfw_Window(const std::string &name,int window_width,int window_height);
    virtual ~glfw_Window(){}

    void close();
    void startMainLoop();
    void startMainLoopConstantUpdateRenderInterpolation(int ticksPerSecond);


    static void error_callback(int error, const char* description);


    void setGLFWcursor(GLFWcursor* cursor);
    GLFWcursor* createGLFWcursor(Image* image, int midX, int midY);
};
