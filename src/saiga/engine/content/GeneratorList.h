/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// GeneratorList (docs/engine_design.md §7.9, §12.1, R-REG-1; S23a): the content generators of a world by GeneratorId.
// The engine registers FlatGenerator as 0, then WorldSetup::generators adds the game's. makeGeneratorList runs the
// registration checks (SAIGA_EXIT_ERROR in every build).

#pragma once

#include "saiga/engine/content/Generator.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/export.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
struct SpaceTemplate;

class SAIGA_ENGINE_API GeneratorList
{
   public:
    // Held by shared_ptr so streaming-pool jobs can keep a generator alive (D8: jobs never capture the World)
    void add(GeneratorId id, std::shared_ptr<const Generator> generator);

    struct Entry
    {
        GeneratorId id = 0;
        std::shared_ptr<const Generator> generator;
        // The slot table at registration (slots() is called once)
        std::vector<GeneratorSlot> slots;
    };
    // Null if not registered
    const Entry* find(GeneratorId id) const;
    const Entry* findByName(const std::string& name) const;
    // By ID
    const std::map<GeneratorId, Entry>& entries() const { return byId; }
    // The registered generators in name order (saves, messages)
    std::vector<const Entry*> byName() const;

    // The registration checks (every build, SAIGA_EXIT_ERROR): at most 200 slots, candidatesPerChunk <= 2^24, unique
    // slot names, exactly one of ReSnapOnEdit / TombstoneOnEdit per slot, keptCount(slot, 0) == candidatesPerChunk and
    // non-increasing in the level, unique names and IDs (add), a legacyTerrain generator without slots, every
    // template's generator registered
    void validate(const std::vector<SpaceTemplate>& templates) const;

   private:
    std::map<GeneratorId, Entry> byId;
};

// FlatGenerator as 0, then the game's generators (null: none); validated against the templates
SAIGA_ENGINE_API GeneratorList makeGeneratorList(void (*games)(GeneratorList&),
                                                 const std::vector<SpaceTemplate>& templates);

}  // namespace Engine
}  // namespace Saiga
