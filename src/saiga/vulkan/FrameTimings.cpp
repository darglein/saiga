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
void FrameTimings<Finder>::init(vk::Device _device, vk::PhysicalDevice physicalDevice, Saiga::Vulkan::Queue* _queue,
                                Memory::VulkanMemory* _memory)
{
    device = _device;
    queue  = _queue;
    memory = _memory;

    timestampPeriod = physicalDevice.getProperties().limits.timestampPeriod;
    auto validBits  = physicalDevice.getQueueFamilyProperties()[queue->getQueueFamilyIndex()].timestampValidBits;
    SAIGA_ASSERT(validBits > 0, "The main queue does not support timestamps");
    timestampMask = validBits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << validBits) - 1);
}

template <typename Finder>
void FrameTimings<Finder>::beginFrame(uint64_t timelineValue)
{
    auto& timing         = timings[next];
    timing.timelineValue = timelineValue;
    current              = next;
    next                 = (next + 1) % numberOfFrames;
}

template <typename Finder>
void FrameTimings<Finder>::update(vk::Semaphore _frameTimeline)
{
    frameTimeline = _frameTimeline;
    if (!frameTimeline) return;
    auto completed = device.getSemaphoreCounterValue(frameTimeline);
    // auto iter = running;

    while (running != next)
    {
        auto& timing = timings[running];

        auto finished = timing.timelineValue <= completed;

        if (finished)
        {
            auto result = device.getQueryPoolResults(queryPool, getFirst(running), getCount(), getCount() * 8,
                                                     timing.sections.data(), 8,
                                                     vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
            SAIGA_ASSERT(result == vk::Result::eSuccess);

            // Ticks -> nanoseconds
            for (auto& s : timing.sections)
            {
                s.first  = static_cast<uint64_t>((s.first & timestampMask) * timestampPeriod);
                s.second = static_cast<uint64_t>((s.second & timestampMask) * timestampPeriod);
            }

            if (lastFrameSections.has_value())
            {
                auto& lastValue = lastFrameSections.value();
                SectionTimesIter curr, next;
                decltype(insertionPoint->pauses.begin()) pauseIter;
                for (curr = lastValue.begin(), next = lastValue.begin() + 1, pauseIter = insertionPoint->pauses.begin();
                     next != lastValue.end(); curr++, next++, pauseIter++)
                {
                    // Sections begin at TOP_OF_PIPE and end at BOTTOM_OF_PIPE, so they can overlap.
                    *pauseIter = next->first > curr->second ? next->first - curr->second : 0;
                }
                auto first_begin = timing.sections.begin()->first;
                *pauseIter       = first_begin > curr->second ? first_begin - curr->second : 0;
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
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, queryPool, getBegin(index));
}

template <typename Finder>
void FrameTimings<Finder>::leaveSection(const std::string& name, vk::CommandBuffer cmd)
{
    auto index = nameToSectionMap[name];
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, queryPool, getEnd(index));
}

template <typename Finder>
void FrameTimings<Finder>::finishFrame(vk::Semaphore _frameTimeline, uint64_t timelineValue)
{
    if (bestSection.has_value() && memory)
    {
        // Pauses are in ns. Clamp, so that a huge (bogus) pause does not overflow.
        auto best = std::min<uint64_t>(bestSection.value().length, uint64_t(1e9));
        auto time = static_cast<int64_t>(0.8 * best);
        memory->performTimedDefrag(time, _frameTimeline, timelineValue);
    }
}

template <typename Finder>
void FrameTimings<Finder>::resetFrame(vk::CommandBuffer cmd)
{
    cmd.resetQueryPool(queryPool, getFirst(current), getCount());
}


template class FrameTimings<FindMinPause>;

}  // namespace Saiga::Vulkan
