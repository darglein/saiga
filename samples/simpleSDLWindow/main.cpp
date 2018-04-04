/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/framework.h"
#include "saiga/sdl/sdl_window.h"
#include "saiga/rendering/deferred_renderer.h"
#include "saiga/util/crash.h"

#include "simpleWindow.h"

#undef main

int main( int argc, char* args[] )
{
    cout << "asdf" << endl;
    Triangle t(vec3(0),vec3(2,0,0),vec3(2,1,0));
    cout << t << endl;
    cout << glm::degrees(glm::atan(0.5)) << " " << glm::degrees(t.minimalAngle()) << endl;
    return 0;
    //Add a signal handler for SIGSEGV and print the stack trace when a SIGSEGV is caught
    catchSegFaults();

    //Specify some window parameters and create a SDL window
    WindowParameters windowParameters;
    windowParameters.name = "Simple SDL Window";
    windowParameters.vsync = false;
    windowParameters.resizeAble = true;
    windowParameters.mode = WindowParameters::Mode::windowed;
    windowParameters.width = 1280;
    windowParameters.height = 720;

    auto window = new SDLWindow(windowParameters);

    //Create the deferred rendering engine
    RenderingParameters rp; //Use default rendering settings
    window->init(rp);

    //Everything from saiga is now setup, so we can start our own program now
    SimpleWindow* simpleWindow = new SimpleWindow(window);

    int updatesPerSecond = 60;
    int framesPerSecond = 0; //no limit
    window->startMainLoop(updatesPerSecond,framesPerSecond);
	 
    delete simpleWindow;
    delete window;

    return 0;
}
