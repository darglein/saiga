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
    // Host + clients over loopback without loss (§17.3, S16; ScenarioOptions::networkVariants, not local-only)
    HostClient,       // delay 0
    HostClientDelay,  // delay 2
    HostClientAsync,  // asynchronous client stepping (seeded 0, 1 or 2 steps per host step, 3 steps late)
};
// The network of a host + client variant (disabled for the single-world variants)
SAIGA_ENGINE_API ScenarioNetwork networkOf(ScenarioVariant v);
// The variants the runner runs for an entry: Plain, Repeat, SaveLoad, and the host + client ones when
// options.networkVariants is set and the entry is not local-only
SAIGA_ENGINE_API std::vector<ScenarioVariant> variantsFor(const ScenarioList::Entry& entry,
                                                          const ScenarioOptions& options);
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
    // Plain: the final state (run_scenario's snapshot), its step index and hash; empty / 0 for the other variants and
    // for runs that ended with a ScenarioAbort (maxSteps, a deadline) or an exception; kept for failed checks
    std::vector<uint8_t> finalSave;
    uint64_t finalStep = 0, finalHash = 0;
};

// Runs one scenario in one variant (headless, with the setup's seed and worker threads)
SAIGA_ENGINE_API ScenarioResult runScenario(const WorldSetup& setup, const ScenarioList::Entry& entry,
                                            ScenarioVariant variant, const ScenarioOptions& options = {});
// Every scenario whose name contains filter, in all its variants (variantsFor)
SAIGA_ENGINE_API std::vector<ScenarioResult> runScenarios(const WorldSetup& setup, const ScenarioList& list,
                                                          const std::string& filter,
                                                          const ScenarioOptions& options = {});

}  // namespace Engine
}  // namespace Saiga
