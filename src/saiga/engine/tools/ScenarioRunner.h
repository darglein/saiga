/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioWorld.h"
#include "saiga/export.h"

#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
// The variants of a scenario (docs/engine_design.md §17.4)
enum class ScenarioVariant
{
    Plain,     // once
    Repeat,    // twice: equal per-step hashes (nondeterminism)
    SaveLoad,  // saved and loaded into a fresh World halfway (state outside components, R-STATE-1)
};
SAIGA_ENGINE_API const char* variantName(ScenarioVariant v);

struct ScenarioResult
{
    std::string scenario;
    ScenarioVariant variant = ScenarioVariant::Plain;
    bool passed             = false;
    std::vector<std::string> failures;  // failed checks, exceptions, variant mismatches
    std::string report;                 // failures, log lines, dumps, traces, diffs
    uint64_t steps = 0;
    double seconds = 0;
};

// Runs one scenario in one variant (headless, with the setup's seed and worker threads)
SAIGA_ENGINE_API ScenarioResult runScenario(const WorldSetup& setup, const ScenarioList::Entry& entry,
                                            ScenarioVariant variant, const ScenarioOptions& options = {});
// Every scenario whose name contains filter, in all three variants
SAIGA_ENGINE_API std::vector<ScenarioResult> runScenarios(const WorldSetup& setup, const ScenarioList& list,
                                                          const std::string& filter,
                                                          const ScenarioOptions& options = {});

}  // namespace Engine
}  // namespace Saiga
