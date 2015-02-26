#pragma once


#include "libhello/window/window.h"

#include <GLFW/glfw3.h>

#include "libhello/sdl/sdl_eventhandler.h"
#include "libhello/glfw/glfw_eventhandler.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>


class glfw_Window : public Window{
protected:
    GLFWwindow* window;

    bool initWindow();
    bool initInput();
public:
    SDL_EventHandler eventHandler;

    glfw_Window(const std::string &name,int window_width,int window_height);


    void close();
    void startMainLoop();


    static void error_callback(int error, const char* description);


};
