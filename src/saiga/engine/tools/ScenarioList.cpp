/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ScenarioList: registered scenarios by name.

#include "saiga/engine/tools/ScenarioList.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace Engine
{
void ScenarioList::add(const char* name, ScenarioFn run, bool localOnly)
{
    // Refused in every build (R-REG-1, like the other registration lists)
    if (!name || !*name || !run) SAIGA_EXIT_ERROR("ScenarioList::add: a name and a function");
    for (const char* c = name; *c; ++c)
        if (!((*c >= 'a' && *c <= 'z') || (*c >= '0' && *c <= '9') || *c == '_'))
            SAIGA_EXIT_ERROR(std::string("ScenarioList::add: the name ") + name + " is not snake_case");
    for (const Entry& e : list)
        if (e.name == name) SAIGA_EXIT_ERROR(std::string("ScenarioList::add: the name ") + name + " is already used");
    list.push_back(Entry{name, std::move(run), localOnly});
}

std::vector<const ScenarioList::Entry*> ScenarioList::match(const std::string& filter) const
{
    std::vector<const Entry*> out;
    for (const Entry& e : list)
        if (filter.empty() || e.name.find(filter) != std::string::npos) out.push_back(&e);
    return out;
}

const ScenarioList::Entry* ScenarioList::find(const std::string& name) const
{
    for (const Entry& e : list)
        if (e.name == name) return &e;
    return nullptr;
}

}  // namespace Engine
}  // namespace Saiga
