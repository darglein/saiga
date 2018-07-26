/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/gphoto2/gphoto.h"

#include "saiga/sdl/sdl.h"
#include "saiga/util/crash.h"

#include "cameraCalibration.h"



int main( int argc, char* args[] )
{
    //Add a signal handler for SIGSEGV and print the stack trace when a SIGSEGV is caught
    catchSegFaults();


    GPhoto gp;

    Image img;
    while(true)
    {
        if(gp.hasNewImage(img))
            cout << "got image " << img << endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }


    return 0;

    WindowParameters windowParameters;
    windowParameters.fromConfigFile("config.ini");
    windowParameters.name = "Forward Rendering";

    // 1. Create an SDL window.
    // This also creates the required OpenGL context.
    SDLWindow window(windowParameters);

    // 2. Create the OpenGL renderer
    Forward_Renderer renderer(window);


    // 3. Create an object of our class, which is both renderable and updateable
    Sample simpleWindow(window,renderer);

    // Everyhing is initilalized, we can run the main loop now!
    MainLoopParameters mainLoopParameters;
    mainLoopParameters.fromConfigFile("config.ini");
    mainLoopParameters.framesPerSecond = 0;
    window.startMainLoop(mainLoopParameters);
    return 0;
}
