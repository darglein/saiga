#include "sample.h"
#include "saiga/framework.h"
//#include "saiga/vulkan/base/vulkanexamplebase.h"
#include "saiga/vulkan/window/SDLWindow.h"

int main(const int argc, const char *argv[])
{
    using namespace  Saiga;


    WindowParameters windowParameters;
    windowParameters.fromConfigFile("config.ini");
    windowParameters.name = "Forward Rendering";


    Saiga::Vulkan::SDLWindow window(windowParameters);


    Saiga::Vulkan::VulkanForwardRenderer renderer(window,true);

//    std::this_thread::sleep_for(std::chrono::seconds(3));
//    return 0;

    VulkanExample example(window,renderer);
    example.init();


    window.startMainLoop();

    return 0;
}
