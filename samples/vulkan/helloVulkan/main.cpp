#include "sample.h"
#include "saiga/framework/framework.h"
#include "saiga/vulkan/window/SDLWindow.h"

#undef main

extern int maingsdgdfg();
int main(const int argc, const char *argv[])
{
    using namespace  Saiga;


    {
        Saiga::WindowParameters windowParameters;
        Saiga::initSample(windowParameters.saigaParameters);
        windowParameters.fromConfigFile("config.ini");


        Saiga::Vulkan::SDLWindow window(windowParameters);
        //        Saiga::Vulkan::GLFWWindow window(windowParameters);

        Saiga::Vulkan::VulkanParameters vulkanParams;
        vulkanParams.enableValidationLayer = true;
        Saiga::Vulkan::VulkanForwardRenderer renderer(window,vulkanParams);

        

        VulkanExample example(window,renderer);
        renderer.initChildren();

        MainLoopParameters params;
        params.mainLoopInfoTime = 1;
        params.framesPerSecond = 0;
        window.startMainLoop(params);

        renderer.waitIdle();
    }

    return 0;
}
