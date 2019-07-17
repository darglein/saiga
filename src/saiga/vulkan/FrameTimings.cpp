//
// Created by Peter Eichinger on 2019-03-08.
//

#include "FrameTimings.h"

#include "saiga/core/util/easylogging++.h"

#include <algorithm>
#include <exception>


namespace Saiga::Vulkan
{
template <typename Finder>
void FrameTimings<Finder>::beginFrame(const FrameSync& sync)
{
    auto& timing = timings[next];
    timing.fence = sync.frameFence;
    current      = next;
    next         = (next + 1) % numberOfFrames;
}

template <typename Finder>
void FrameTimings<Finder>::update()
{
    // auto iter = running;

    while (running != next)
    {
        auto& timing = timings[running];

        auto finished = device.getFenceStatus(timing.fence) == vk::Result::eSuccess;

        if (finished)
        {
            device.getQueryPoolResults(queryPool, getFirst(running), getCount(), getCount() * 8, timing.sections.data(),
                                       8, vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

            if (lastFrameSections.has_value())
            {
                auto& lastValue = lastFrameSections.value();
                SectionTimesIter curr, next;
                decltype(insertionPoint->pauses.begin()) pauseIter;
                for (curr = lastValue.begin(), next = lastValue.begin() + 1, pauseIter = insertionPoint->pauses.begin();
                     next != lastValue.end(); curr++, next++, pauseIter++)
                {
                    const auto pause = next->first - curr->second;
                    *pauseIter       = pause;
                }
                const auto lastPause = timing.sections.begin()->first - curr->second;
                *pauseIter           = lastPause;
            }

            bestSection = finder.findSuitablePause(recentFramePauses, insertionPoint);

            lastFrameSections = timing.sections;
            insertionPoint++;
            if (insertionPoint == recentFramePauses.end())
            {
                insertionPoint = recentFramePauses.begin();
            }

            running = (running + 1) % numberOfFrames;
        }
        else
        {
            break;
        }
    }
}

template <typename Finder>
void FrameTimings<Finder>::reset()
{
    running = 0;
    next    = 0;
    destroyPool();
}

template <typename Finder>
void FrameTimings<Finder>::registerFrameSection(const std::string& name, uint32_t index)
{
    if (queryPool)
    {
        throw std::logic_error("Query pool must not be created. Call reset().");
    }
    auto inserted = frameSections.insert(std::make_pair(index, name));

    if (!inserted.second)
    {
        throw std::invalid_argument("Index already in use");
    }

    nameToSectionMap.insert(std::make_pair(name, index));
}

template <typename Finder>
void FrameTimings<Finder>::unregisterFrameSection(uint32_t index)
{
    if (queryPool)
    {
        throw std::logic_error("Query pool must not be created. Call reset().");
    }

    auto found = std::find_if(frameSections.begin(), frameSections.end(),
                              [=](const Entry& entry) { return entry.first == index; });

    if (found == frameSections.end())
    {
        throw std::invalid_argument("Index not in use");
    }

    nameToSectionMap.erase(found->second);
    frameSections.erase(found);
}

template <typename Finder>
void FrameTimings<Finder>::create(uint32_t _numberOfFrames, uint32_t _frameWindow)
{
    destroyPool();

    numberOfFrames = _numberOfFrames;
    current        = 0;
    next           = 0;
    running        = 0;
    frameWindow    = _frameWindow;
    auto queryPoolCreateInfo =
        vk::QueryPoolCreateInfo{vk::QueryPoolCreateFlags(), vk::QueryType ::eTimestamp,
                                static_cast<uint32_t>(numberOfFrames * frameSections.size() * 2)};
    queryPool = device.createQueryPool(queryPoolCreateInfo);

    timings.resize(numberOfFrames);
    std::fill(timings.begin(), timings.end(), Timing(frameSections.size()));

    recentFramePauses.resize(frameWindow);

    std::fill(recentFramePauses.begin(), recentFramePauses.end(), FramePauses(frameSections.size()));

    insertionPoint = recentFramePauses.begin();
}

template <typename Finder>
void FrameTimings<Finder>::destroyPool()
{
    if (device && queryPool)
    {
        device.destroyQueryPool(queryPool);

        queryPool = nullptr;
    }
}

template <typename Finder>
void FrameTimings<Finder>::enterSection(const std::string& name, vk::CommandBuffer cmd)
{
    auto index = nameToSectionMap[name];
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, queryPool, getBegin(index));
}

template <typename Finder>
void FrameTimings<Finder>::leaveSection(const std::string& name, vk::CommandBuffer cmd)
{
    auto index = nameToSectionMap[name];
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, getEnd(index));
}

template <typename Finder>
void FrameTimings<Finder>::finishFrame(vk::Semaphore semaphore)
{
    bool performed = false;

    if (bestSection.has_value())
    {
        auto best = bestSection.value();
        auto time = static_cast<int64_t>(0.8 * best.length);
        performed = memory->performTimedDefrag(time, semaphore);
    }

    if (!performed)
    {
        // No defrag took place. have to submit dummy command buffer to wait for semaphore
        vk::PipelineStageFlags wait{vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eAllGraphics};
        vk::SubmitInfo si;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &dummy;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores    = &semaphore;
        si.pWaitDstStageMask  = &wait;
        queue->submit(si, nullptr);
    }
}

template <typename Finder>
void FrameTimings<Finder>::resetFrame(vk::CommandBuffer cmd)
{
    cmd.resetQueryPool(queryPool, getFirst(current), getCount());
}


template class FrameTimings<FindMinPause>;

}  // namespace Saiga::Vulkan
