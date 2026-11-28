/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The World's streaming members (docs/engine_design.md §12.3; S24): the settings, chunkReady, completeResidency,
// refreshSimulated and the step start of a streaming world (G5).

#include "saiga/engine/core/Log.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/streaming/SimulationSet.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Engine
{
void World::setStreamingSettings(const StreamingSettings& settings)
{
    SAIGA_ASSERT(!inStep(), "World::setStreamingSettings inside a step");
    SAIGA_ASSERT(settings.maxInFlight >= 1, "StreamingSettings::maxInFlight must be at least 1");
    streamSettings = settings;
}

void World::setStreamingSynchronous(bool on)
{
    StreamingSettings s = streamSettings;
    s.synchronous       = on;
    setStreamingSettings(s);
}

ChunkId World::chunkAt(SpaceId space, double x, double z) const
{
    SAIGA_ASSERT(space < spaceCount(), "World::chunkAt: unknown space");
    const int64_t n = spaceTemplate(space).sizeChunks;
    auto index      = [n](double v)
    {
        if (std::isnan(v)) return int64_t(0);
        return int64_t(std::min(std::max(std::floor(v / 256), 0.0), double(n - 1)));
    };
    ChunkId c;
    c.space = space;
    c.x     = int32_t(index(x));
    c.z     = int32_t(index(z));
    return c;
}

bool World::chunkReady(SpaceId space, double x, double z) const
{
    if (!config.streaming) return true;
    if (space >= spaceCount()) return false;
    return residencyPtr->ready(chunkAt(space, x, z));
}

bool World::completeResidency(double timeoutSeconds)
{
    SAIGA_ASSERT(!inStep(), "World::completeResidency inside a step");
    if (!config.streaming) return true;
    const bool ok = residencyPtr->complete(timeoutSeconds);
    // The flags follow the ready set at once (a complete point between steps, a load)
    simulation->stepStart();
    return ok;
}

void World::refreshSimulated(Entity e)
{
    if (!config.streaming) return;
    simulation->refresh(e);
}

void World::streamingStepStart()
{
    Profiler::Timer timer(prof, stepStartScope);
    // The refresh (request part and promotion, the island count) when the centres changed since the last update: a
    // change between steps is seen by phase 1 exactly as after a load
    if (residencyPtr->centresChanged()) residencyPtr->refresh();
    // S25: the step-start hooks (the reloads of chunks made ready by the refresh, completeResidency or a load)
    // By id over the hooks registered at the start: a hook removed by an earlier one does not run, and the running
    // callable is a copy (a hook may remove itself)
    if (!stepStartHooks.empty())
    {
        std::vector<int> ids;
        ids.reserve(stepStartHooks.size());
        for (const auto& h : stepStartHooks) ids.push_back(h.first);
        for (int id : ids)
        {
            auto it = std::find_if(stepStartHooks.begin(), stepStartHooks.end(),
                                   [id](const std::pair<int, std::function<void()>>& h) { return h.first == id; });
            if (it == stepStartHooks.end()) continue;
            const std::function<void()> fn = it->second;
            fn();
        }
    }
    simulation->stepStart();
}

}  // namespace Engine
}  // namespace Saiga
