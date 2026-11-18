/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SystemList: the systems per phase; Phase names.

#include "saiga/engine/world/SystemList.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace Engine
{
const char* phaseName(Phase p)
{
    switch (p)
    {
        case Phase::None:
            return "none";
        case Phase::Input:
            return "input";
        case Phase::Events:
            return "events";
        case Phase::Behaviour:
            return "behaviour";
        case Phase::Gameplay:
            return "gameplay";
        case Phase::Physics:
            return "physics";
        case Phase::Transform:
            return "transform";
        case Phase::Streaming:
            return "streaming";
        case Phase::Extract:
            return "extract";
    }
    return "?";
}

void SystemList::add(Phase phase, const char* name, SystemFn fn, RunOn runOn)
{
    SAIGA_ASSERT(int(phase) >= kFirstPhase && int(phase) <= kLastPhase, "SystemList::add: invalid phase");
    SAIGA_ASSERT(fn, std::string("SystemList::add: system ") + name + " has no function");
    for (auto& list : perPhase)
        for (auto& s : list)
            SAIGA_ASSERT(s.name != name, std::string("SystemList::add: the name ") + name + " is already used");
    perPhase[size_t(phase)].push_back(System{phase, name, fn, runOn});
}

size_t SystemList::size() const
{
    size_t n = 0;
    for (auto& list : perPhase) n += list.size();
    return n;
}

}  // namespace Engine
}  // namespace Saiga
