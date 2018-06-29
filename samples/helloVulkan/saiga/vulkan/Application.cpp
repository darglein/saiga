/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Application.h"
#include <chrono>
#include <thread>

namespace Saiga {
namespace Vulkan {

Application::Application(int width, int height)
    : forwardRenderer(*this,swapChain)
{
    window.createWindow(width,height);

    createInstance(true);

    init_physical_device();

    createDevice();


    //    init_global_layer_properties();
    //    init_instance();


    auto surface = window.createSurfaceKHR(inst);


//    swapChain = new Vulkan::SwapChain(inst, physicalDevice, device);
    swapChain.create(inst, physicalDevice, device);
    swapChain.setSurface(surface);
    swapChain.create(&window.width, &window.height);



    mainCommandPool.create(*this,swapChain.queueNodeIndex,vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    cmd = mainCommandPool.createCommandBuffer(*this);

    forwardRenderer.create(width,height);
}

Application::~Application()
{

}

void Application::run()
{
    // draw

    int count = 100;
    for(int i = 0; i < count; ++i)
    {
        cout << "render " << i << endl;
        update();


        vk::CommandBufferResetFlags resetFlags;
        cmd.reset(resetFlags);
        cmd.begin( vk::CommandBufferBeginInfo() );


        vk::Viewport viewport;
        vk::Rect2D scissor;
        viewport.height = (float)window.height;
        viewport.width = (float)window.width;
        viewport.minDepth = (float)0.0f;
        viewport.maxDepth = (float)1.0f;
        viewport.x = 0;
        viewport.y = 0;
        cmd.setViewport(0, 1, &viewport);

        scissor.extent.width = window.width;
        scissor.extent.height = window.height;
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        cmd.setScissor(0, 1, &scissor);



        forwardRenderer.begin(cmd);

        render(cmd);

        cmd.endRenderPass();
        //        res = vkEndCommandBuffer(info.cmd);
        cmd.end();


       forwardRenderer.end(cmd);


        std::this_thread::sleep_for(std::chrono::milliseconds(16));



    }

}

}
}
