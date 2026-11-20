/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

#include <functional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class ScenarioWorld;

using ScenarioFn = std::function<void(ScenarioWorld&)>;

/**
 * A list of scenarios (docs/engine_design.md §17.2, R-REG-1; S12): the engine's in tests/test_engine_scenarios.cpp, a
 * game's in EngineAppSetup::scenarios (S13). Names are unique and snake_case.
 */
class SAIGA_ENGINE_API ScenarioList
{
   public:
    struct Entry
    {
        std::string name;
        ScenarioFn run;
        bool localOnly = false;  // no host + client variant (S16)
    };

    void add(const char* name, ScenarioFn run, bool localOnly = false);
    const std::vector<Entry>& entries() const { return list; }
    // The entries whose name contains filter (all for an empty filter), in registration order
    std::vector<const Entry*> match(const std::string& filter) const;

   private:
    std::vector<Entry> list;
};

}  // namespace Engine
}  // namespace Saiga
