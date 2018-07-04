#include "sample.h"
#include "saiga/framework.h"
#include "saiga/vulkan/window/SDLWindow.h"



int main(const int argc, const char *argv[])
{
    using namespace  Saiga;


    WindowParameters windowParameters;
    windowParameters.fromConfigFile("config.ini");
    windowParameters.name = "Forward Rendering";


    Saiga::Vulkan::SDLWindow window(windowParameters);


    Saiga::Vulkan::VulkanForwardRenderer renderer(window,true);


    VulkanExample example(window,renderer);
    example.init();

    MainLoopParameters params;
    params.framesPerSecond = 60;
    window.startMainLoop(params);

    return 0;
}
