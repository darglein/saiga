/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// PrefabStore (docs/engine_design.md §14, §15.3; S23a): the prefabs read so far by (data directory, name, version),
// each read and checked once and never evicted (held by the ContentCache, shared by the worlds of a setup). Thread
// safe.

#pragma once

#include "saiga/engine/content/Prefab.h"
#include "saiga/export.h"

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API PrefabStore
{
   public:
    // The prefab, read on the first request; null with an error naming the file and the key (errors are not cached)
    std::shared_ptr<const Prefab> get(const std::string& dataDirectory, const std::string& name, uint16_t version,
                                      std::string& error);
    size_t size() const;

   private:
    mutable std::mutex mutex;
    std::map<std::tuple<std::string, std::string, uint16_t>, std::shared_ptr<const Prefab>> prefabs;
};

}  // namespace Engine
}  // namespace Saiga
