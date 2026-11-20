/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The scenario runner (docs/engine_design.md §17.2, §17.4; S12): plain, repeat and save/load variants.

#include "saiga/engine/tools/ScenarioRunner.h"

#include "saiga/engine/save/SaveDiff.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace Saiga
{
namespace Engine
{
namespace
{
const char* const kStoppedByRunner = "stopped by the runner";

struct Run
{
    std::vector<std::string> failures;
    std::string report;
    uint64_t steps     = 0;
    uint64_t finalHash = 0;
    std::vector<uint64_t> hashes;
    std::vector<uint8_t> finalSave;
};

Run runOnce(const WorldSetup& setup, const ScenarioList::Entry& entry, const ScenarioOptions& options,
            const std::function<void(ScenarioWorld&, Run&)>& configure,
            const std::function<void(ScenarioWorld&, Run&)>& finish = {})
{
    Run r;
    // An exception of a WorldSetup callback (components, systems, classes, data) fails this variant, not the whole run
    std::unique_ptr<ScenarioWorld> owned;
    try
    {
        owned = std::make_unique<ScenarioWorld>(setup, options);
    }
    catch (const std::exception& e)
    {
        r.failures.push_back(std::string("the world could not be created: ") + e.what());
        return r;
    }
    catch (...)
    {
        r.failures.push_back("the world could not be created (an exception that is not a std::exception)");
        return r;
    }
    ScenarioWorld& w = *owned;
    if (configure) configure(w, r);
    try
    {
        entry.run(w);
        w.checkAbort();
        if (finish) finish(w, r);
    }
    catch (const ScenarioAbort& a)
    {
        if (a.reason.compare(0, std::strlen(kStoppedByRunner), kStoppedByRunner) != 0)
            r.failures.push_back("aborted: " + a.reason);
    }
    catch (const std::exception& e)
    {
        r.failures.push_back(std::string("exception: ") + e.what());
    }
    catch (...)
    {
        r.failures.push_back("an exception that is not a std::exception");
    }
    for (const std::string& f : w.failures()) r.failures.push_back(f);
    r.steps = w.steps();
    try
    {
        if (!w.world().inStep())
        {
            r.finalHash = w.world().hash();
            r.finalSave = w.world().save();
        }
        r.report = w.report();
    }
    catch (const std::exception& e)
    {
        r.failures.push_back(std::string("the final state could not be saved: ") + e.what());
    }
    return r;
}

// The first differences of two saves, one line each
std::string describeDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
{
    DiffFilter f;
    f.limit          = 20;
    const SaveDiff d = diffSaves(a, b, f);
    std::string out;
    if (!d.error.empty()) return "\n  (no diff: " + d.error + ")";
    for (const SaveDifference& x : d.differences) out += "\n  " + x.describe();
    if (d.total > d.differences.size())
        out += "\n  ... and " + std::to_string(d.total - d.differences.size()) + " more";
    return out;
}

std::string hex(uint64_t h)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return buf;
}
}  // namespace

const char* variantName(ScenarioVariant v)
{
    switch (v)
    {
        case ScenarioVariant::Plain:
            return "Plain";
        case ScenarioVariant::Repeat:
            return "Repeat";
        case ScenarioVariant::SaveLoad:
            return "SaveLoad";
    }
    return "?";
}

ScenarioResult runScenario(const WorldSetup& setup, const ScenarioList::Entry& entry, ScenarioVariant variant,
                           const ScenarioOptions& options)
{
    const auto start = std::chrono::steady_clock::now();
    ScenarioResult result;
    result.scenario = entry.name;
    result.variant  = variant;
    std::string details;

    if (variant == ScenarioVariant::Plain)
    {
        Run r           = runOnce(setup, entry, options, {});
        result.failures = r.failures;
        result.steps    = r.steps;
        details         = r.report;
    }
    else if (variant == ScenarioVariant::Repeat)
    {
        auto record   = [](ScenarioWorld& w, Run& r) { w.control.hashes = &r.hashes; };
        Run a         = runOnce(setup, entry, options, record);
        WorldSetup s2 = setup;
        if (options.varyThreads) s2.workerThreads = setup.workerThreads > 1 ? 1 : 3;
        Run b           = runOnce(s2, entry, options, record);
        result.failures = a.failures;
        for (const std::string& f : b.failures)
            if (std::find(result.failures.begin(), result.failures.end(), f) == result.failures.end())
                result.failures.push_back("second run: " + f);
        result.steps = a.steps;
        details      = a.report;
        if (!b.failures.empty()) details += "  second run:\n" + b.report;
        // The first differing step (or the end of the shorter run)
        size_t k = 0;
        while (k < a.hashes.size() && k < b.hashes.size() && a.hashes[k] == b.hashes[k]) k++;
        const bool hashesDiffer = k < a.hashes.size() || k < b.hashes.size();
        if (hashesDiffer || a.finalHash != b.finalHash)
        {
            std::string message;
            if (hashesDiffer && k < a.hashes.size() && k < b.hashes.size())
            {
                // Both again up to that step, then the saves are compared
                std::vector<uint8_t> saveA, saveB;
                auto stopAt = [k](std::vector<uint8_t>* out)
                {
                    return [k, out](ScenarioWorld& w, Run&)
                    {
                        w.control.stopAt   = uint64_t(k) + 1;
                        w.control.stopSave = out;
                    };
                };
                const Run ra = runOnce(setup, entry, options, stopAt(&saveA));
                const Run rb = runOnce(s2, entry, options, stopAt(&saveB));
                message = "repeat: the runs differ after step " + std::to_string(k + 1) + " (hash " + hex(a.hashes[k]) +
                          " != " + hex(b.hashes[k]) + ")";
                // A rerun that ended before the step has no save to compare (its failures say why)
                auto missed = [&](const char* which, const Run& rerun)
                {
                    message += std::string("\n  (the ") + which + " rerun ended before step " + std::to_string(k + 1);
                    for (const std::string& f : rerun.failures) message += "; " + f;
                    message += ")";
                };
                if (saveA.empty()) missed("first", ra);
                if (saveB.empty()) missed("second", rb);
                if (!saveA.empty() && !saveB.empty())
                {
                    if (saveA == saveB)
                        message += "\n  (the reruns did not differ at that step: the difference is not reproducible)";
                    else
                        message += describeDiff(saveA, saveB);
                }
            }
            else if (hashesDiffer)
                message = "repeat: the runs took " + std::to_string(a.hashes.size()) + " and " +
                          std::to_string(b.hashes.size()) + " steps";
            else
                message = "repeat: the final states differ (hash " + hex(a.finalHash) + " != " + hex(b.finalHash) +
                          "): a change after the last step" + describeDiff(a.finalSave, b.finalSave);
            result.failures.push_back(message);
        }
    }
    else
    {
        // The plain run counts the steps; the round trip is at the first step-helper boundary at or after half of
        // them (at the end for scenarios with fewer than 2 steps)
        Run plain          = runOnce(setup, entry, options, {});
        const uint64_t mid = plain.steps / 2;
        if (!plain.failures.empty())
        {
            // A failing baseline fails the variant (its step count and final state are no reference)
            for (const std::string& f : plain.failures) result.failures.push_back("baseline run: " + f);
            result.steps = plain.steps;
            details      = plain.report;
        }
        else
        {
            uint64_t swappedAt = 0;
            Run r              = runOnce(
                setup, entry, options,
                [&](ScenarioWorld& w, Run&)
                {
                    if (plain.steps >= 2) w.control.saveLoadAt = mid;
                },
                [&](ScenarioWorld& w, Run&)
                {
                    if (!w.control.saveLoadDone) w.saveLoad();
                    swappedAt = w.control.swappedAtStep;
                });
            result.failures = r.failures;
            result.steps    = r.steps;
            details         = r.report;
            // No step ran on the loaded world (steps after the middle only through world() or api(), or the swap after
            // a call() that ran all remaining steps): the variant would claim coverage it did not have
            if (plain.steps >= 2 && r.failures.empty() && r.steps <= swappedAt)
                result.failures.push_back(
                    "save/load: the world was replaced at step " + std::to_string(swappedAt) + " of " +
                    std::to_string(r.steps) +
                    ", so nothing ran on the loaded world (a step helper or call() must start after step " +
                    std::to_string(mid) + ": step through the ScenarioWorld helpers or call() in smaller parts)");
            if (options.strictContinuation && result.failures.empty() && r.finalHash != plain.finalHash)
                result.failures.push_back("save/load: the final state differs from the plain run (hash " +
                                          hex(r.finalHash) + " != " + hex(plain.finalHash) +
                                          "): state outside the saved components changed the behavior (R-STATE-1)" +
                                          describeDiff(plain.finalSave, r.finalSave));
        }
    }

    result.passed  = result.failures.empty();
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.report  = entry.name + " (" + variantName(variant) + "): " + (result.passed ? "passed" : "FAILED") + "\n";
    if (!result.passed)
    {
        for (const std::string& f : result.failures) result.report += "  " + f + "\n";
        if (!details.empty()) result.report += "  context:\n" + details;
    }
    return result;
}

std::vector<ScenarioResult> runScenarios(const WorldSetup& setup, const ScenarioList& list, const std::string& filter,
                                         const ScenarioOptions& options)
{
    std::vector<ScenarioResult> results;
    for (const ScenarioList::Entry* e : list.match(filter))
        for (ScenarioVariant v : {ScenarioVariant::Plain, ScenarioVariant::Repeat, ScenarioVariant::SaveLoad})
            results.push_back(runScenario(setup, *e, v, options));
    return results;
}

}  // namespace Engine
}  // namespace Saiga
