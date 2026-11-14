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
        // The frame is complete when the frame timeline reaches this value.
        uint64_t timelineValue = 0;

        SectionTimes sections;

        Timing() = default;
        explicit Timing(size_t numSections) : sections(numSections) {}
    };

    // Nanoseconds per timestamp tick
    double timestampPeriod = 1;
    // Mask of the valid timestamp bits
    uint64_t timestampMask = ~uint64_t(0);

    std::optional<SuitablePause> bestSection;
    std::optional<SectionTimes> lastFrameSections;
    std::vector<FramePauses> recentFramePauses;
    std::vector<FramePauses>::iterator insertionPoint;


    vk::Device device;
    vk::Semaphore frameTimeline;
    Saiga::Vulkan::Queue* queue = nullptr;
    std::vector<Timing> timings;
    uint32_t numberOfFrames = 0, next = 0, current = 0, running = 0, frameWindow = 0;

    vk::QueryPool queryPool;
    std::set<Entry, KeyComparator> frameSections;
    std::map<std::string, uint32_t> nameToSectionMap;
    Finder finder;
    Memory::VulkanMemory* memory = nullptr;
    void destroyPool();

    inline uint32_t getCount() const { return static_cast<uint32_t>(frameSections.size() * 2); }
    inline uint32_t getFirst(uint32_t frame) const { return static_cast<uint32_t>(frame * getCount()); }
    inline uint32_t getBegin(uint32_t index) const { return getFirst(current) + index * 2; }
    inline uint32_t getEnd(uint32_t index) const { return getFirst(current) + index * 2 + 1; }

   public:
    FrameTimings() = default;
    ~FrameTimings() { destroyPool(); }

    FrameTimings(const FrameTimings&)            = delete;
    FrameTimings& operator=(const FrameTimings&) = delete;

    void init(vk::Device device, vk::PhysicalDevice physicalDevice, Saiga::Vulkan::Queue* queue,
              Memory::VulkanMemory* memory);

    // timelineValue: the frame timeline value the frame signals when it is complete.
    void beginFrame(uint64_t timelineValue);
    // Reads the timestamps of the completed frames.
    void update(vk::Semaphore frameTimeline);

    void registerFrameSection(const std::string& name, uint32_t index);
    void unregisterFrameSection(uint32_t index);
    void create(uint32_t numberOfFrames, uint32_t _frameWindow = 20);
    void reset();

    void enterSection(const std::string& name, vk::CommandBuffer cmd);
    void leaveSection(const std::string& name, vk::CommandBuffer cmd);

    void resetFrame(vk::CommandBuffer cmd);

    // Starts defragmentation copies (after the frame with the given timeline value) if a suitable pause was found.
    void finishFrame(vk::Semaphore frameTimeline, uint64_t timelineValue);
};
}  // namespace Saiga::Vulkan
