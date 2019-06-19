#include "saiga/core/framework/framework.h"
#include "saiga/core/sdl/sdl.h"
#include "saiga/core/util/Thread/threadPool.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/vulkan/memory/VulkanStlAllocator.h"
#include "saiga/vulkan/window/SDLWindow.h"

#include "sample.h"

#include <memory>
#include <vector>
int main(const int argc, const char* argv[])
{
    using namespace Saiga;
    //    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level [%fbase]: %msg");
    {
        Saiga::createGlobalThreadPool(-1);
        Saiga::WindowParameters windowParameters;
        Saiga::initSample(windowParameters.saigaParameters);
        windowParameters.fromConfigFile("config.ini");

        Saiga::Vulkan::SDLWindow window(windowParameters);
        //        Saiga::Vulkan::GLFWWindow window(windowParameters);

        Saiga::Vulkan::VulkanParameters vulkanParams;
        vulkanParams.fromConfigFile("config.ini");
        Saiga::Vulkan::VulkanForwardRenderer renderer(window, vulkanParams);



        VulkanExample example(window, renderer);

        Saiga::Vulkan::Memory::VulkanStlAllocator<int> alloc(renderer.base(), vk::BufferUsageFlagBits::eUniformBuffer);


        // std::vector<int, VulkanStlAllocator<int>> test(alloc);
        // test.reserve(1000);
        // for (int i = 0; i < 10000; ++i)
        //{
        //    test.push_back(i);
        //}
        //
        // for (int i = 0; i < 10000; ++i)
        //{
        //    SAIGA_ASSERT(i == test[i]);
        //}


        //        renderer.initChildren();

        MainLoopParameters params;
        params.mainLoopInfoTime = 1;
        params.framesPerSecond  = 0;
        window.startMainLoop(params);

        renderer.waitIdle();
    }

    return 0;
}
