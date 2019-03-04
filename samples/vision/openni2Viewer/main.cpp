#include "saiga/core/framework/framework.h"
#include "saiga/vulkan/window/SDLWindow.h"

#include "sample.h"

int main(int, const char**)
{
    using namespace Saiga;


    {
        Saiga::WindowParameters windowParameters;
        Saiga::initSample(windowParameters.saigaParameters);
        windowParameters.fromConfigFile("config.ini");

        windowParameters.width  = 640 * 2;
        windowParameters.height = 800;

        Saiga::Vulkan::SDLWindow window(windowParameters);

        Saiga::Vulkan::VulkanParameters vulkanParams;
        vulkanParams.enableValidationLayer = true;
        Saiga::Vulkan::VulkanForwardRenderer renderer(window, vulkanParams);


        VulkanExample example(window, renderer);


        window.startMainLoop();

        renderer.waitIdle();
    }

    return 0;
}
