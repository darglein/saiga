//
// Created by Peter Eichinger on 2019-03-08.
//

#include "FrameTimings.h"

#include "saiga/core/util/easylogging++.h"

#include <algorithm>
#include <exception>
#include <glm/glm.hpp>
namespace Saiga::Vulkan
{
void FrameTimings::beginFrame(const FrameSync& sync)
{
    auto& timing = timings[next];
    timing.fence = sync.frameFence;
    current      = next;
    next         = (next + 1) % numberOfFrames;
}

void FrameTimings::updateMeanStdDev(MovingMean& moving, uint64_t sample)
{
    auto newMean = alpha * sample  + (1-alpha) * moving.mean;

    auto delta_i = sample - moving.ema;
    auto new_ema = moving.ema + alpha * delta_i;
    auto new_emvar = (1-alpha) * (moving.emvar + alpha *delta_i * delta_i);

    moving.mean = newMean;
    moving.ema = new_ema;
    moving.emvar = new_emvar;
}
void FrameTimings::update()
{
    // auto iter = running;

    while (running != next)
    {
        auto& timing = timings[running];

        auto finished = device.getFenceStatus(timing.fence) == vk::Result::eSuccess;

        if (finished)
        {
            LOG(INFO) << "Frame " << running;
            device.getQueryPoolResults(queryPool, getFirst(running), getCount(), getCount() * 8, timing.sections.data(),
                                       8, vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
            for (auto& pair : timing.sections)
            {
                LOG(INFO) << "  " << pair.first << " " << pair.second;
            }

            if (lastFrameSections.has_value()) {
                auto& lastValue = lastFrameSections.value();
                decltype (lastValue.begin()) curr;
                decltype (lastValue.begin()) next;
                decltype (meanStdDev.begin()) meanIter;
                for(curr = lastValue.begin(), next = lastValue.begin()+1,meanIter =meanStdDev.begin() ; next != lastValue.end(); curr++, next++, meanIter++) {
                    // https://dsp.stackexchange.com/questions/811/determining-the-mean-and-standard-deviation-in-real-time
                    const auto pause = next->first - curr->second;
                    updateMeanStdDev(*meanIter, pause);
                }
                auto lastPause = timing.sections.begin()->first - curr->second;
                updateMeanStdDev(*meanIter, lastPause);
            }

            lastFrameSections = timing.sections;
            for (auto & mean: meanStdDev) {
                LOG(INFO) << "Mean: " << mean.mean << ",  STDEV: " << glm::sqrt(mean.emvar);
            }
            running = (running + 1) % numberOfFrames;
        }
        else
        {
            break;
        }
    }
}

void FrameTimings::reset()
{
    running = 0;
    next    = 0;
    destroyPool();
}

void FrameTimings::registerFrameSection(const std::string& name, uint32_t index)
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

void FrameTimings::unregisterFrameSection(uint32_t index)
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

void FrameTimings::create(uint32_t _numberOfFrames, uint32_t frameWindow)
{
    destroyPool();

    numberOfFrames = _numberOfFrames;
    current        = 0;
    next           = 0;
    running        = 0;
    auto queryPoolCreateInfo =
        vk::QueryPoolCreateInfo{vk::QueryPoolCreateFlags(), vk::QueryType ::eTimestamp,
                                static_cast<uint32_t>(numberOfFrames * frameSections.size() * 2)};
    queryPool = device.createQueryPool(queryPoolCreateInfo);

    timings.resize(numberOfFrames);
    std::fill(timings.begin(), timings.end(), Timing(frameSections.size()));

    meanStdDev.resize(frameSections.size());

    std::fill(meanStdDev.begin(), meanStdDev.end(), MovingMean());
}

void FrameTimings::destroyPool()
{
    if (device && queryPool)
    {
        device.destroyQueryPool(queryPool);

        queryPool = nullptr;
    }
}
void FrameTimings::enterSection(const std::string& name, vk::CommandBuffer cmd)
{
    auto index = nameToSectionMap[name];
    VLOG(1) << current << " " << current * frameSections.size() * 2 + index * 2;
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPool, getBegin(index));
}
void FrameTimings::leaveSection(const std::string& name, vk::CommandBuffer cmd)
{
    auto index = nameToSectionMap[name];
    VLOG(1) << current << " " << current * frameSections.size() * 2 + index * 2 + 1;
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, queryPool, getEnd(index));
}
void FrameTimings::resetFrame(vk::CommandBuffer cmd)
{
    cmd.resetQueryPool(queryPool, getFirst(current), getCount());
}



}  // namespace Saiga::Vulkan
