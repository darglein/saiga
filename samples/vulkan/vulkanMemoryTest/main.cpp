#include "sample.h"
#include "saiga/framework/framework.h"
#include "saiga/vulkan/window/SDLWindow.h"
#include "saiga/util/easylogging++.h"

#undef main



#include <memory>
#include <vector>
#include "saiga/vulkan/buffer/BufferedAllocator.h"

int main(const int argc, const char *argv[])
{
    using namespace Saiga;

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


        BufferedAllocator<int> alloc(renderer.base.device,renderer.base.physicalDevice,vk::BufferUsageFlagBits::eUniformBuffer);


        std::vector<int,BufferedAllocator<int>> test(alloc);

        for(int i =0; i<10000;++i) {
            test.push_back(i);
        }

        for(int i =0; i<10000; ++i) {
            SAIGA_ASSERT(i == test[i]);
        }


        renderer.initChildren();

        MainLoopParameters params;
        params.mainLoopInfoTime = 1;
        params.framesPerSecond = 0;
        window.startMainLoop(params);

        renderer.waitIdle();
    }

    return 0;
}
