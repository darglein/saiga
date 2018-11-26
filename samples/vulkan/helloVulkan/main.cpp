#include "saiga/framework/framework.h"
#include "saiga/util/easylogging++.h"
#include "saiga/vulkan/window/SDLWindow.h"
#include "sample.h"

#undef main

extern int maingsdgdfg();
int main(const int argc, const char* argv[])
{
    using namespace Saiga;

    LOG(INFO) << "My first info log using default logger";

    {
        Saiga::WindowParameters windowParameters;
        Saiga::initSample(windowParameters.saigaParameters);
        windowParameters.fromConfigFile("config.ini");


        Saiga::Vulkan::SDLWindow window(windowParameters);
        //        Saiga::Vulkan::GLFWWindow window(windowParameters);

        Saiga::Vulkan::VulkanParameters vulkanParams;
        vulkanParams.enableValidationLayer = true;
        Saiga::Vulkan::VulkanForwardRenderer renderer(window, vulkanParams);


        VulkanExample example(window, renderer);
        renderer.initChildren();
        //        example.init();

        MainLoopParameters params;
        params.mainLoopInfoTime = 1;
        params.framesPerSecond  = 60;
        window.startMainLoop(params);

        renderer.waitIdle();
    }

    //    maingsdgdfg();
    return 0;
}
