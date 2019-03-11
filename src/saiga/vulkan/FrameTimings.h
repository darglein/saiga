//
// Created by Peter Eichinger on 2019-03-08.
//

#pragma once

#include "saiga/vulkan/svulkan.h"

#include "FrameSync.h"

#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace Saiga::Vulkan
{
class SAIGA_VULKAN_API FrameTimings final
{
   private:
    using Entry = std::pair<uint32_t, std::string>;
    struct KeyComparator
    {
        bool operator()(const Entry& entry, const Entry& other) const { return entry.first < other.first; }
    };
    using Clock     = std::chrono::system_clock;
    using TimePoint = Clock::time_point;

    struct SAIGA_VULKAN_API Timing
    {
        vk::Fence fence;
        TimePoint begin, end;

        std::vector<std::pair<uint64_t, uint64_t>> sections;

        Timing() = default;
        explicit Timing(size_t numSections) : fence(nullptr), begin(), sections(numSections) {}
    };



    vk::Device device;
    std::vector<Timing> timings;
    uint32_t numberOfFrames;
    uint32_t next, current;
    uint32_t running;

    vk::QueryPool queryPool;
    std::set<Entry, KeyComparator> frameSections;
    std::map<std::string, uint32_t> nameToSectionMap;
    void destroyPool();

    inline uint32_t getBegin(uint32_t index) const { return current * frameSections.size() * 2 + index * 2; }
    inline uint32_t getEnd(uint32_t index) const { return current * frameSections.size() * 2 + index * 2 + 1; }
    inline uint32_t getFirst(uint32_t frame) const { return static_cast<uint32_t>(frame * frameSections.size() * 2); }
    inline uint32_t getCount() const { return static_cast<uint32_t>(frameSections.size() * 2); }

   public:
    FrameTimings() = default;

    ~FrameTimings() { destroyPool(); }

    FrameTimings(vk::Device _device)
        : device(_device),
          timings(0),
          numberOfFrames(0),
          next(0),
          current(0),
          running(0),
          queryPool(nullptr),
          frameSections()
    {
    }

    FrameTimings& operator=(FrameTimings&& other) noexcept
    {
        if (this != &other)
        {
            device         = other.device;
            timings        = std::move(other.timings);
            numberOfFrames = other.numberOfFrames;
            next           = other.next;
            current        = other.current;
            running        = other.running;
            queryPool      = other.queryPool;
            frameSections  = std::move(other.frameSections);

            other.device    = nullptr;
            other.queryPool = nullptr;
        }
        return *this;
    }


    void beginFrame(const FrameSync& sync);
    void update();

    void registerFrameSection(const std::string& name, uint32_t index);
    void unregisterFrameSection(uint32_t index);
    void create(uint32_t numberOfFrames);
    void reset();

    void enterSection(const std::string& name, vk::CommandBuffer cmd);
    void leaveSection(const std::string& name, vk::CommandBuffer cmd);

    void resetFrame(vk::CommandBuffer cmd);
};
}  // namespace Saiga::Vulkan