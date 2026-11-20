/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Entity.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// The outcome of World::tryLoad (S11)
struct LoadResult
{
    // false: the save was rejected and the world is unchanged; error says why
    bool committed = false;
    std::string error;
    // Committed, but these game objects threw in configure or onLoad: their entities (with descendants) were removed
    struct HookError
    {
        Entity entity;
        std::string className, message;
    };
    std::vector<HookError> hookErrors;
    // Descendants of the failed objects, removed with them (without hooks)
    std::vector<Entity> removed;
    // Load listeners that threw (the others still ran)
    std::vector<std::string> listenerErrors;
    bool ok() const { return committed && hookErrors.empty(); }
};

// Passed to load listeners after a committed load
struct LoadInfo
{
    uint64_t loadCount = 0;
    // The structural log records of the load: [logBegin, logEnd) in StructuralLog::appendedTotal() numbering (removes
    // of the old entities, creates of the loaded ones, removals after hook errors)
    uint64_t logBegin = 0, logEnd = 0;
    uint64_t fromStep = 0, toStep = 0;  // the step index before and after the load
};

}  // namespace Engine
}  // namespace Saiga
