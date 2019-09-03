/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/util/crash.h"

#include "simpleWindow.h"


int main(int argc, char* args[])
{
    // Add a signal handler for SIGSEGV and print the stack trace when a SIGSEGV is caught
    catchSegFaults();

    WindowParameters windowParameters;
    initSample(windowParameters.saigaParameters);
    windowParameters.fromConfigFile("config.ini");

    OpenGLParameters openglParameters;
    openglParameters.fromConfigFile("config.ini");

    // 1. Create an SDL window.
    // This also creates the required OpenGL context.
    glfw_Window window(windowParameters, openglParameters);

    // 2. Create the OpenGL renderer
    DeferredRenderer renderer(window);

    // 3. Create an object of our class, which is both renderable and updateable
    Sample simpleWindow(window, renderer);

    // Everyhing is initilalized, we can run the main loop now!
    MainLoopParameters mainLoopParameters;
    mainLoopParameters.fromConfigFile("config.ini");
    window.startMainLoop(mainLoopParameters);
    return 0;
}
