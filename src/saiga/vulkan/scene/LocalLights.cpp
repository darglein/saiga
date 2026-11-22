/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/LocalLights.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace Vulkan
{
LightHandle LocalLights::add(const ChunkId& chunk, const LightData& light)
{
    const LightHandle h{nextHandle++};
    indexOf[h.id] = table.size();
    table.push_back(Light{h, chunk, light, 0});
    changes++;
    return h;
}

void LocalLights::set(LightHandle h, const ChunkId& chunk, const LightData& light)
{
    auto it = indexOf.find(h.id);
    SAIGA_ASSERT(it != indexOf.end(), "RenderScene::setLight: unknown light handle");
    Light& l = table[it->second];
    SAIGA_ASSERT(l.list == 0, "RenderScene::setLight: the light belongs to a list (addLights)");
    l.chunk = chunk;
    l.data  = light;
    changes++;
}

void LocalLights::erase(LightHandle h)
{
    auto it = indexOf.find(h.id);
    SAIGA_ASSERT(it != indexOf.end(), "RenderScene::removeLight: unknown light handle");
    const size_t i = it->second;
    indexOf.erase(it);
    if (i + 1 != table.size())
    {
        table[i]                    = table.back();
        indexOf[table[i].handle.id] = i;
    }
    table.pop_back();
    changes++;
}

void LocalLights::remove(LightHandle h)
{
    auto it = indexOf.find(h.id);
    SAIGA_ASSERT(it != indexOf.end() && table[it->second].list == 0,
                 "RenderScene::removeLight: unknown light handle (or one of a list)");
    erase(h);
}

LightListHandle LocalLights::addList(const ChunkId& chunk, const std::vector<LightData>& lights)
{
    const LightListHandle list{nextList++};
    std::vector<LightHandle>& handles = lists[list.id];
    for (const LightData& d : lights)
    {
        const LightHandle h = add(chunk, d);
        table.back().list   = list.id;
        handles.push_back(h);
    }
    return list;
}

void LocalLights::removeList(LightListHandle list)
{
    auto it = lists.find(list.id);
    SAIGA_ASSERT(it != lists.end(), "RenderScene::removeLights: unknown light list");
    for (LightHandle h : it->second) erase(h);
    lists.erase(it);
}

const LocalLights::Light* LocalLights::find(LightHandle h) const
{
    auto it = indexOf.find(h.id);
    return it == indexOf.end() ? nullptr : &table[it->second];
}

}  // namespace Vulkan
}  // namespace Saiga
