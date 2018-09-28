#include "vulkan/vulkan.hpp"
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <thread>
#include <chrono>
#include <iostream>


static SDL_Window* window;

uint32_t w = 1280;
uint32_t h = 720;


void swapTestVulkan()
{
    SDL_Init( SDL_INIT_VIDEO );
    window = SDL_CreateWindow("Vulkan", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, w,h, SDL_WINDOW_VULKAN);

    vk::Instance instance;
    {
        //create instance
        unsigned int count = 0;
        SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr);
        std::vector<const char*> windowExtensions(count);
        SDL_Vulkan_GetInstanceExtensions(window, &count, windowExtensions.data());

        vk::ApplicationInfo appInfo;
        appInfo.apiVersion = VK_API_VERSION_1_0;

        vk::InstanceCreateInfo instanceCreateInfo;
        instanceCreateInfo.pApplicationInfo = &appInfo;
        instanceCreateInfo.enabledExtensionCount = windowExtensions.size();
        instanceCreateInfo.ppEnabledExtensionNames = windowExtensions.data();

        instance = vk::createInstance(instanceCreateInfo);
    }

    vk::PhysicalDevice physicalDevice = instance.enumeratePhysicalDevices()[0];
    vk::Device device;
    {
        //create device

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        vk::DeviceQueueCreateInfo info;
        info.queueFamilyIndex = 0;
        info.queueCount = 1;
        queueCreateInfos.push_back(info);

        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = queueCreateInfos.size();
        deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();


        std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        deviceCreateInfo.enabledExtensionCount = deviceExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

        device = physicalDevice.createDevice(deviceCreateInfo);
    }

    vk::SurfaceKHR surface;

    vk::Format colorFormat;
    vk::ColorSpaceKHR colorSpace;
    vk::SwapchainKHR swapChain;
    {
        //create surface
        VkSurfaceKHR s;
        SDL_Vulkan_CreateSurface(window,instance, &s);
        surface = s;

        std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevice.getSurfaceFormatsKHR(surface);
        for (auto&& surfaceFormat : surfaceFormats)
        {
            if (surfaceFormat.format == vk::Format::eB8G8R8A8Unorm)
            {
                colorFormat = surfaceFormat.format;
                colorSpace = surfaceFormat.colorSpace;
                break;
            }
        }
    }
    {
        vk::SwapchainCreateInfoKHR createInfo;
        createInfo.surface = surface;
        createInfo.minImageCount = 3;
        createInfo.imageFormat = colorFormat;
        createInfo.imageColorSpace = colorSpace;
        createInfo.imageExtent = vk::Extent2D{ w,h };
        createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
        createInfo.preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
        createInfo.imageArrayLayers = 1;
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
        createInfo.presentMode = vk::PresentModeKHR::eImmediate;
                createInfo.presentMode = vk::PresentModeKHR::eMailbox;
        //createInfo.presentMode = vk::PresentModeKHR::eFifo;
        createInfo.clipped = true;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

        // Enable transfer source on swap chain images if supported
        //        if (surfCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        //            createInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        //        }

        // Enable transfer destination on swap chain images if supported
        //        if (surfCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
        //            createInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        //        }
        createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
        swapChain = device.createSwapchainKHR(createInfo);
        assert(swapChain);
    }

    std::vector<vk::Image> images = device.getSwapchainImagesKHR(swapChain);;
    std::cout << "got " << images.size() << " images." << std::endl;
    std::vector<vk::ImageView> imageViews;//(images.size());
    for(auto i : images)
    {
        vk::ImageViewCreateInfo colorAttachmentView;
        colorAttachmentView.format = colorFormat;
        colorAttachmentView.components = {
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity
        };
        colorAttachmentView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        colorAttachmentView.subresourceRange.baseMipLevel = 0;
        colorAttachmentView.subresourceRange.levelCount = 1;
        colorAttachmentView.subresourceRange.baseArrayLayer = 0;
        colorAttachmentView.subresourceRange.layerCount = 1;
        colorAttachmentView.viewType = vk::ImageViewType::e2D;
        colorAttachmentView.image = i;
        imageViews.push_back(device.createImageView(colorAttachmentView));
    }


    vk::RenderPass renderPass;
    {
        std::array<vk::AttachmentDescription, 1> attachments = {};
        // Color attachment
        attachments[0].format = colorFormat;
        attachments[0].samples = vk::SampleCountFlagBits::e1;
        attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
        attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
        attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachments[0].initialLayout = vk::ImageLayout::eUndefined;
        attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;


        vk::AttachmentReference colorReference = {};
        colorReference.attachment = 0;
        colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

        //render pass
        vk::SubpassDescription subpassDescription = {};
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;

        // Subpass dependencies for layout transitions
        std::array<vk::SubpassDependency, 1> dependencies;

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
        dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
        dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;


        vk::RenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.attachmentCount = 1;//static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDescription;
        renderPassInfo.dependencyCount = 1;//static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();
        renderPass = device.createRenderPass(renderPassInfo);
        assert(renderPass);
    }


    std::vector<vk::Framebuffer>frameBuffers;
    {
        frameBuffers.resize(images.size());
        for (uint32_t i = 0; i < frameBuffers.size(); i++)
        {
            vk::ImageView attachments[1];
            attachments[0] = imageViews[i];


            vk::FramebufferCreateInfo frameBufferCreateInfo = {};
            frameBufferCreateInfo.renderPass = renderPass;
            frameBufferCreateInfo.attachmentCount = 1;
            frameBufferCreateInfo.pAttachments = attachments;
            frameBufferCreateInfo.width = w;
            frameBufferCreateInfo.height = h;
            frameBufferCreateInfo.layers = 1;

            frameBuffers[i] = device.createFramebuffer(frameBufferCreateInfo);
            assert(frameBuffers[i]);
            //               VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &framebuffer));
        }

    }


    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;
    {
        //command pool
        vk::CommandPoolCreateInfo info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,0);
        commandPool = device.createCommandPool(info);
        assert(commandPool);
        vk::CommandBufferAllocateInfo cmdBufAllocateInfo(
                    commandPool,
                    vk::CommandBufferLevel::ePrimary,
                    images.size()
                    );

        commandBuffers = device.allocateCommandBuffers(cmdBufAllocateInfo);
    }
    vk::Queue queue = device.getQueue(0,0);
    assert(queue);



    std::vector<vk::Semaphore> imageVailable(images.size());
    std::vector<vk::Semaphore> renderComplete(images.size());
    std::vector<vk::Fence> frameFence(images.size());
    {
        for(unsigned int i = 0; i < images.size(); ++i)
        {
            vk::FenceCreateInfo fenceCreateInfo{
                vk::FenceCreateFlagBits::eSignaled
            };
            frameFence[i] = device.createFence(fenceCreateInfo);

            vk::SemaphoreCreateInfo semaphoreCreateInfo {
                vk::SemaphoreCreateFlags()
            };
            imageVailable[i] = device.createSemaphore(semaphoreCreateInfo);
            renderComplete[i] = device.createSemaphore(semaphoreCreateInfo);
        }

    }



    auto clear =  [&](uint32_t currentBuffer, int semID)
    {
        vk::CommandBuffer& cmd = commandBuffers[currentBuffer];

		cmd.reset({});

        vk::CommandBufferBeginInfo cmdBeginInfo;
        //cmdBeginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        cmd.begin(cmdBeginInfo);
        {
            vk::ClearValue clearValue;

//            static int i = 0;
			float f = 1;// (float)std::abs(sin(i++*0.01));
            clearValue.color = {  std::array<float,4>{f,0.f,0.f,1.f } };
            vk::RenderPassBeginInfo renderPassBeginInfo;
            renderPassBeginInfo.renderPass = renderPass;
            renderPassBeginInfo.renderArea.extent.width = w;
            renderPassBeginInfo.renderArea.extent.height = h;
            renderPassBeginInfo.clearValueCount = 1;
            renderPassBeginInfo.pClearValues = &clearValue;
            renderPassBeginInfo.framebuffer = frameBuffers[currentBuffer];
            cmd.beginRenderPass(renderPassBeginInfo,vk::SubpassContents::eInline);

            vk::Viewport viewport(0,0,w,h,0,1);
            cmd.setViewport(0,viewport);
            vk::Rect2D scissor({0,0}, {w,h} );
            cmd.setScissor(0,scissor);

            cmd.endRenderPass();
        }

        cmd.end();

        vk::PipelineStageFlags submitPipelineStages =  vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo submitInfo;
        submitInfo.pWaitDstStageMask = &submitPipelineStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageVailable[semID];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderComplete[semID];

        queue.submit(submitInfo,frameFence[semID]);
    };

    auto render = [&]()
    {
        static unsigned int semID = 0;

                device.waitForFences(frameFence[semID], VK_TRUE, std::numeric_limits<uint64_t>::max());


        device.resetFences(frameFence[semID]);
        uint32_t currentBuffer = device.acquireNextImageKHR(swapChain,UINT64_MAX,imageVailable[semID],nullptr).value;
        clear(currentBuffer,semID);


        vk::PresentInfoKHR presentInfo;
		
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChain;
        presentInfo.pImageIndices = &currentBuffer;
        presentInfo.pWaitSemaphores = &renderComplete[semID];
        presentInfo.waitSemaphoreCount = 1;

        queue.presentKHR(presentInfo);
        semID = (semID+1) % images.size();
    };

    for(int i = 0; i < 100; ++i)
        render();


    auto start = std::chrono::high_resolution_clock::now();
    int count = 0;
    int time = 3;

    while(true)
    {
		SDL_Event e;
		while(SDL_PollEvent(&e))
		{
		}
        render();

        count ++;
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now - start;
        if(duration > std::chrono::seconds(time))
            break;
    }
    std::cout << "Rendered " << count << " frames in " << time << " seconds. -> " << count/double(time) << " fps." << std::endl;

    SDL_DestroyWindow( window );
    SDL_Quit();
}
