#include "sample.h"
#include "saiga/framework.h"
#include "saiga/vulkan/base/vulkanexamplebase.h"


int main(const int argc, const char *argv[])
{
    using namespace  Saiga;


    WindowParameters windowParameters;
    windowParameters.fromConfigFile("config.ini");
    windowParameters.name = "Forward Rendering";


    Vulkan::SDLWindow window(windowParameters);
    window.setupWindow();

    VulkanForwardRenderer renderer(window,true);

    VulkanExample example(window,renderer);
    example.init();

    renderer.thing = &example;

    renderer.renderLoop();


    return 0;
}
