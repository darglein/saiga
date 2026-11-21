/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// CosmeticDynamics: render-only dynamic instances of entities (S19b).

#include "games/common/CosmeticDynamics.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace Game
{
using namespace Engine;

void CosmeticDynamics::begin()
{
    for (Entry& e : entries) e.seen = false;
}

void CosmeticDynamics::set(Entity e, const std::vector<MeshId>& meshes, const std::vector<WorldPose>& poses)
{
    SAIGA_ASSERT(renderSink && meshes.size() == poses.size());
    Entry* entry = nullptr;
    for (Entry& x : entries)
        if (x.entity == e) entry = &x;
    if (!entry)
    {
        entries.push_back(Entry());
        entry         = &entries.back();
        entry->entity = e;
    }
    if (entry->meshes != meshes)
    {
        free(*entry);
        entry->meshes = meshes;
        for (MeshId m : meshes) entry->handles.push_back(renderSink->addDynamic(m));
    }
    entry->seen = true;
    for (size_t i = 0; i < poses.size(); ++i)
    {
        // The frame's pose: no interpolation on the GPU
        const ChunkPose c = toChunkPose(poses[i]);
        renderSink->setDynamic(entry->handles[i], c.chunk, c.pose, c.pose);
    }
}

void CosmeticDynamics::end()
{
    for (size_t i = 0; i < entries.size();)
    {
        if (entries[i].seen)
        {
            ++i;
            continue;
        }
        free(entries[i]);
        entries.erase(entries.begin() + std::ptrdiff_t(i));
    }
}

void CosmeticDynamics::clear()
{
    for (Entry& e : entries) free(e);
    entries.clear();
}

void CosmeticDynamics::free(Entry& e)
{
    for (InstanceHandle h : e.handles) renderSink->removeDynamic(h);
    e.handles.clear();
    e.meshes.clear();
}

}  // namespace Game
}  // namespace Saiga
