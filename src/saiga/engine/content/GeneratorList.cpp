/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/GeneratorList.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/content/gen/FlatGenerator.h"
#include "saiga/engine/world/SpaceTemplate.h"

#include <algorithm>
#include <set>

namespace Saiga
{
namespace Engine
{
void GeneratorList::add(GeneratorId id, std::shared_ptr<const Generator> generator)
{
    if (!generator) SAIGA_EXIT_ERROR("WorldSetup::generators: generator " + std::to_string(id) + " is null");
    const std::string name = generator->name() ? generator->name() : "";
    if (name.empty()) SAIGA_EXIT_ERROR("WorldSetup::generators: generator " + std::to_string(id) + " has no name");
    if (byId.count(id))
        SAIGA_EXIT_ERROR("WorldSetup::generators: the generator ID " + std::to_string(id) + " of '" + name +
                         "' is used by '" + byId.at(id).generator->name() + "'" +
                         (id == 0 ? " (0 is the engine's FlatGenerator)" : ""));
    if (findByName(name))
        SAIGA_EXIT_ERROR("WorldSetup::generators: the generator name '" + name + "' is registered twice");
    Entry e;
    e.id        = id;
    e.slots     = generator->slots();
    e.generator = std::move(generator);
    byId[id]    = std::move(e);
}

const GeneratorList::Entry* GeneratorList::find(GeneratorId id) const
{
    auto it = byId.find(id);
    return it == byId.end() ? nullptr : &it->second;
}

const GeneratorList::Entry* GeneratorList::findByName(const std::string& name) const
{
    for (const auto& [id, e] : byId)
        if (name == e.generator->name()) return &e;
    return nullptr;
}

std::vector<const GeneratorList::Entry*> GeneratorList::byName() const
{
    std::vector<const Entry*> out;
    for (const auto& [id, e] : byId) out.push_back(&e);
    std::sort(out.begin(), out.end(),
              [](const Entry* a, const Entry* b) { return std::string(a->generator->name()) < b->generator->name(); });
    return out;
}

void GeneratorList::validate(const std::vector<SpaceTemplate>& templates) const
{
    for (const auto& [id, e] : byId)
    {
        const Generator& g     = *e.generator;
        const std::string what = "generator '" + std::string(g.name()) + "': ";
        if (e.slots.size() > 200)
            SAIGA_EXIT_ERROR(what + std::to_string(e.slots.size()) +
                             " slots (at most 200; slot values 200-255 are reserved for engine keys)");
        if (g.legacyTerrain() && !e.slots.empty())
            SAIGA_EXIT_ERROR(what + "a legacyTerrain generator has no slots (it writes no generation state)");
        std::set<std::string> names;
        for (uint32_t s = 0; s < uint32_t(e.slots.size()); ++s)
        {
            const GeneratorSlot& slot = e.slots[s];
            const std::string sw      = what + "slot '" + slot.name + "': ";
            if (slot.name.empty()) SAIGA_EXIT_ERROR(what + "slot " + std::to_string(s) + " has no name");
            if (!names.insert(slot.name).second) SAIGA_EXIT_ERROR(sw + "the slot name is used twice");
            const uint32_t edit = slot.flags & (ReSnapOnEdit | TombstoneOnEdit);
            if (edit != ReSnapOnEdit && edit != TombstoneOnEdit)
                SAIGA_EXIT_ERROR(sw + "exactly one of ReSnapOnEdit and TombstoneOnEdit");
            const uint32_t n = g.candidatesPerChunk(s);
            if (n > (uint32_t(1) << 24))
                SAIGA_EXIT_ERROR(sw + "candidatesPerChunk " + std::to_string(n) + " is above 2^24");
            if (g.keptCount(s, 0) != n)
                SAIGA_EXIT_ERROR(sw + "keptCount(slot, 0) = " + std::to_string(g.keptCount(s, 0)) +
                                 " differs from candidatesPerChunk = " + std::to_string(n));
            for (int level = 1; level < 16; ++level)
                if (g.keptCount(s, level) > g.keptCount(s, level - 1))
                    SAIGA_EXIT_ERROR(sw + "keptCount increases at level " + std::to_string(level));
        }
    }
    const std::vector<SpaceTemplate> defaults(1);
    const std::vector<SpaceTemplate>& list = templates.empty() ? defaults : templates;
    for (const SpaceTemplate& t : list)
        if (!find(t.generator))
            SAIGA_EXIT_ERROR("space template '" + t.name + "': the generator " + std::to_string(t.generator) +
                             " is not registered (WorldSetup::generators)");
}

GeneratorList makeGeneratorList(void (*games)(GeneratorList&), const std::vector<SpaceTemplate>& templates)
{
    GeneratorList list;
    list.add(0, std::make_shared<FlatGenerator>());
    if (games) games(list);
    list.validate(templates);
    return list;
}

}  // namespace Engine
}  // namespace Saiga
