#include "compute.h"
#include "saiga/framework.h"
#include "saiga/vulkan/window/SDLWindow.h"



int main(const int argc, const char *argv[])
{
    Saiga::WindowParameters windowParameters;
    windowParameters.fromConfigFile("config.ini");
    windowParameters.name = "Forward Rendering";


    Saiga::Vulkan::SDLWindow window(windowParameters);


    Saiga::Vulkan::VulkanForwardRenderer renderer(window,true);


    Compute example(window,renderer);
    example.init();

    Saiga::MainLoopParameters params;
    params.framesPerSecond = 60;
    window.startMainLoop(params);

    return 0;
}
