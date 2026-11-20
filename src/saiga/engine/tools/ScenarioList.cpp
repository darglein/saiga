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
    SAIGA_ASSERT(name && *name && run, "ScenarioList::add: a name and a function");
    for (const char* c = name; *c; ++c)
        SAIGA_ASSERT((*c >= 'a' && *c <= 'z') || (*c >= '0' && *c <= '9') || *c == '_',
                     std::string("ScenarioList::add: the name ") + name + " is not snake_case");
    for (const Entry& e : list)
        SAIGA_ASSERT(e.name != name, std::string("ScenarioList::add: the name ") + name + " is already used");
    list.push_back(Entry{name, std::move(run), localOnly});
}

std::vector<const ScenarioList::Entry*> ScenarioList::match(const std::string& filter) const
{
    std::vector<const Entry*> out;
    for (const Entry& e : list)
        if (filter.empty() || e.name.find(filter) != std::string::npos) out.push_back(&e);
    return out;
}

}  // namespace Engine
}  // namespace Saiga
