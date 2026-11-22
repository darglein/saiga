/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/content/PrefabStore.h"

namespace Saiga
{
namespace Engine
{
std::shared_ptr<const Prefab> PrefabStore::get(const std::string& dataDirectory, const std::string& name,
                                               uint16_t version, std::string& error)
{
    const auto key = std::make_tuple(dataDirectory, name, version);
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = prefabs.find(key);
        if (it != prefabs.end()) return it->second;
    }
    // Read outside the lock; the first insert wins
    auto prefab = std::make_shared<Prefab>();
    if (!readPrefab(dataDirectory, name, version, *prefab, error)) return nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    return prefabs.emplace(key, std::move(prefab)).first->second;
}

size_t PrefabStore::size() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return prefabs.size();
}

}  // namespace Engine
}  // namespace Saiga
