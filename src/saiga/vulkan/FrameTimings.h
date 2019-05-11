//
// Created by Peter Eichinger on 2019-03-08.
//

#pragma once

#include "saiga/vulkan/svulkan.h"

#include "FrameSync.h"
#include "memory/VulkanMemory.h"

#include <chrono>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace Saiga::Vulkan
{
struct FramePauses
{
    std::vector<uint64_t> pauses;

    FramePauses() = default;
    FramePauses(size_t sections) : pauses(sections) {}
};

struct SuitablePause
{
    uint64_t pauseIndex;
    uint64_t length;

    SuitablePause(uint64_t index, uint64_t len) : pauseIndex(index), length(len) {}
};

struct FindMinPause
{
    SuitablePause findSuitablePause(const std::vector<FramePauses>& recentPauses,
                                    std::vector<FramePauses>::const_iterator insertionPoint)
    {
        std::vector<uint64_t> minPerPause(recentPauses[0].pauses.size(), std::numeric_limits<uint64_t>::max());

        for (auto& frame : recentPauses)
        {
            for (auto i = 0U; i < minPerPause.size(); ++i)
            {
                minPerPause[i] = std::min(minPerPause[i], frame.pauses[i]);
            }
        }

        auto longestPause = std::max_element(minPerPause.begin(), minPerPause.end());
        return SuitablePause(std::distance(minPerPause.begin(), longestPause), *longestPause);
    }
};

template <typename Finder = FindMinPause>
class SAIGA_VULKAN_API FrameTimings
{
   private:
    using SectionTimes     = std::vector<std::pair<uint64_t, uint64_t>>;
    using SectionTimesIter = SectionTimes::iterator;
    using Entry            = std::pair<uint32_t, std::string>;
    struct KeyComparator
    {
        bool operator()(const Entry& entry, const Entry& other) const { return entry.first < other.first; }
    };

    struct SAIGA_VULKAN_API Timing
    {
        vk::Fence fence;

        SectionTimes sections;

        Timing() = default;
        explicit Timing(size_t numSections) : fence(nullptr), sections(numSections) {}
    };


    vk::CommandBuffer dummy = nullptr;

    std::optional<SuitablePause> bestSection;
    std::optional<SectionTimes> lastFrameSections;
    std::vector<FramePauses> recentFramePauses;
    std::vector<FramePauses>::iterator insertionPoint;


    vk::Device device;
    Saiga::Vulkan::Queue* queue;
    std::vector<Timing> timings;
    uint32_t numberOfFrames, next, current, running, frameWindow;

    vk::QueryPool queryPool;
    std::set<Entry, KeyComparator> frameSections;
    std::map<std::string, uint32_t> nameToSectionMap;
    Finder finder;
    Memory::VulkanMemory* memory;
    void destroyPool();

    inline uint32_t getCount() const { return static_cast<uint32_t>(frameSections.size() * 2); }
    inline uint32_t getFirst(uint32_t frame) const { return static_cast<uint32_t>(frame * getCount()); }
    inline uint32_t getBegin(uint32_t index) const { return getFirst(current) + index * 2; }
    inline uint32_t getEnd(uint32_t index) const { return getFirst(current) + index * 2 + 1; }

   public:
    FrameTimings() = default;

    ~FrameTimings()
    {
        if (device && dummy)
        {
            queue->commandPool.freeCommandBuffer(dummy);
        }
        destroyPool();
    }

    FrameTimings(vk::Device _device, Saiga::Vulkan::Queue* _queue, Memory::VulkanMemory* _memory)
        : device(_device),
          queue(_queue),
          timings(0),
          numberOfFrames(0),
          next(0),
          current(0),
          running(0),
          frameWindow(0),
          queryPool(nullptr),
          frameSections(),
          finder(),
          memory(_memory)
    {
        dummy = queue->commandPool.allocateCommandBuffer();

        vk::CommandBufferBeginInfo cbbi;
        cbbi.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        dummy.begin(cbbi);
        dummy.end();
    }

    FrameTimings& operator=(FrameTimings&& other) noexcept
    {
        if (this != &other)
        {
            device          = other.device;
            timings         = std::move(other.timings);
            numberOfFrames  = other.numberOfFrames;
            next            = other.next;
            current         = other.current;
            running         = other.running;
            frameWindow     = other.frameWindow;
            queryPool       = other.queryPool;
            frameSections   = std::move(other.frameSections);
            memory          = other.memory;
            dummy           = other.dummy;
            queue           = other.queue;
            other.device    = nullptr;
            other.queryPool = nullptr;
            other.memory    = nullptr;
            other.dummy     = nullptr;
            other.queue     = nullptr;
        }
        return *this;
    }

    FrameTimings(FrameTimings&&)      = delete;
    FrameTimings(const FrameTimings&) = delete;
    FrameTimings& operator=(const FrameTimings&) = delete;

    void beginFrame(const FrameSync& sync);
    void update();

    void registerFrameSection(const std::string& name, uint32_t index);
    void unregisterFrameSection(uint32_t index);
    void create(uint32_t numberOfFrames, uint32_t _frameWindow = 20);
    void reset();

    void enterSection(const std::string& name, vk::CommandBuffer cmd);
    void leaveSection(const std::string& name, vk::CommandBuffer cmd);

    void resetFrame(vk::CommandBuffer cmd);

    void finishFrame(vk::Semaphore semaphore);
};
}  // namespace Saiga::Vulkan
