
#if 1
#include "compute.h"
#include "saiga/framework.h"
#include "saiga/vulkan/window/SDLWindow.h"

#undef main

extern int maingsdgdfg();

int main(const int argc, const char *argv[])
{

    {
        Saiga::WindowParameters windowParameters;
        windowParameters.fromConfigFile("config.ini");
        windowParameters.name = "Forward Rendering";


            Saiga::Vulkan::SDLWindow window(windowParameters);
//        Saiga::Vulkan::GLFWWindow window(windowParameters);

        Saiga::Vulkan::VulkanParameters vulkanParams;

        Saiga::Vulkan::VulkanForwardRenderer renderer(window,vulkanParams);


        Compute example(window,renderer);
        renderer.initChildren();

        Saiga::MainLoopParameters params;
        params.framesPerSecond = 60;
        window.startMainLoop(params);
    }

//    maingsdgdfg();
    //        return 0;
    return 0;
}
#endif
