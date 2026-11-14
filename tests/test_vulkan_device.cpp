/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "vulkan_test_context.h"

using namespace Saiga;

class VulkanDevice : public VulkanTest
{
};

TEST_F(VulkanDevice, HeadlessDevice)
{
    EXPECT_TRUE(renderer->isHeadless());
    EXPECT_TRUE(base().device);
    EXPECT_EQ(base().apiVersion, Vulkan::SAIGA_VULKAN_API_VERSION);
    EXPECT_GE(base().physicalDevice.getProperties().apiVersion, Vulkan::SAIGA_VULKAN_API_VERSION);

    EXPECT_TRUE(base().mainQueue.is_valid());
    ASSERT_NE(base().computeQueue, nullptr);
    ASSERT_NE(base().transferQueue, nullptr);
    EXPECT_TRUE(base().computeQueue->is_valid());
    EXPECT_TRUE(base().transferQueue->is_valid());

    // One offscreen image per frame in flight instead of the swap chain images (created with the first frame).
    window->renderFrames(1);
    EXPECT_EQ(renderer->swapChainSize(), (int)Vulkan::VulkanRenderer::framesInFlight);
    EXPECT_EQ(renderer->getColorFormat(), vk::Format::eR8G8B8A8Unorm);
}

TEST_F(VulkanDevice, FrameTimeline)
{
    uint32_t first_frame = base().current_frame;
    window->renderFrames(3);
    EXPECT_EQ(base().current_frame, first_frame + 3);
    EXPECT_EQ(app.updates, 3);

    // Frame f signals the timeline with f + 1.
    base().waitFrameValue(first_frame + 3);
    EXPECT_EQ(base().completedFrameValue(), uint64_t(first_frame + 3));
}

// A deferred deletion must not run before the frame that was recorded when it was requested has completed.
TEST_F(VulkanDevice, DeferredDelete)
{
    bool deleted = false;
    base().deferDelete([&]() { deleted = true; });

    base().processDeferredDeletes();
    EXPECT_FALSE(deleted);

    window->renderFrames(1);
    renderer->waitIdle();
    base().processDeferredDeletes();
    EXPECT_TRUE(deleted);
}
