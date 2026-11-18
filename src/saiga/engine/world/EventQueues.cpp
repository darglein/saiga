/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// EventQueues: registration checks, the step begin and the hash of the pending events.

#include "saiga/engine/world/EventQueues.h"

#include "saiga/engine/ecs/Registry.h"

namespace Saiga
{
namespace Engine
{
void EventQueues::validateRegistration() const
{
    // Engine IDs 0..63 and game IDs 64.. must each be a prefix (every enum value up to the last registered one)
    auto checkRange = [&](size_t first, size_t last, const char* range)
    {
        bool gap  = false;
        int gapId = -1;
        (void)gapId;  // read only in the assert message
        for (size_t id = first; id <= last && id < queues.size(); ++id)
        {
            if (!queues[id])
            {
                if (!gap) gapId = int(id);
                gap = true;
            }
            else
            {
                SAIGA_ASSERT(!gap, std::string("validateRegistration: ") + range + " event ID " +
                                       std::to_string(gapId) + " is not registered, but " + queues[id]->name + " (ID " +
                                       std::to_string(id) + ") is");
            }
        }
    };
    checkRange(0, 63, "engine");
    checkRange(64, queues.size() == 0 ? 0 : queues.size() - 1, "game");
}

void EventQueues::beginStep()
{
    for (auto& q : queues)
        if (q) q->beginStep();
}

void EventQueues::hashPending(Hasher& h) const
{
    // In ID order, each queue in emit order
    for (auto& q : queues)
    {
        if (!q) continue;
        h.addU32(q->id).addU64(uint64_t(q->pendingCount()));
        q->hashPending(h);
    }
}

size_t EventQueues::registeredCount() const
{
    size_t n = 0;
    for (auto& q : queues) n += q ? 1 : 0;
    return n;
}

void EventQueues::checkNotParallel() const
{
    SAIGA_ASSERT(!registry.inParallelEach(), "events.emit inside parallelEach (R-PAR-1)");
}

}  // namespace Engine
}  // namespace Saiga
