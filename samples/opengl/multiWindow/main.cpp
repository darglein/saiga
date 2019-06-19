/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/sdl/sdl.h"
#include "saiga/core/util/crash.h"
#include "saiga/core/util/tostring.h"

#include "simpleWindow.h"


int main(int argc, char* args[])
{
    // Add a signal handler for SIGSEGV and print the stack trace when a SIGSEGV is caught
    catchSegFaults();

    WindowParameters windowParameters;
    initSample(windowParameters.saigaParameters);
    windowParameters.fromConfigFile("config.ini");



    for (int i = 1; i <= 3; ++i)
    {
        {
            windowParameters.name = "Window " + to_string(i);
            OpenGLParameters openglParameters;
            openglParameters.fromConfigFile("config.ini");

            // 1. Create an SDL window.
            // This also creates the required OpenGL context.
            SDLWindow window(windowParameters, openglParameters);

            // 2. Create the OpenGL renderer
            Deferred_Renderer renderer(window);

            // 3. Create an object of our class, which is both renderable and updateable
            Sample simpleWindow(window, renderer);

            // Everyhing is initilalized, we can run the main loop now!
            window.startMainLoop();
        }
        std::cout << "window closed. Opening next window in..." << std::endl;

        for (int j = 0; i < 3 && j < 3; ++j)
        {
            std::cout << (3 - j) << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }


    return 0;
}
