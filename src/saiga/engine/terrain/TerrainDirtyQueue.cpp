/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainDirtyQueue.h"

#include <algorithm>

namespace Saiga
{
namespace Engine
{
namespace
{
// Overlapping or touching (adjacent samples)
bool touches(const SampleRect& a, const SampleRect& b)
{
    return a.i0 <= b.i1 + 1 && b.i0 <= a.i1 + 1 && a.k0 <= b.k1 + 1 && b.k0 <= a.k1 + 1;
}
}  // namespace

void TerrainDirtyQueue::merge(List& l, SampleRect box)
{
    if (box.empty()) return;
    std::vector<SampleRect>& list = l.boxes;
    // An active front box is neither grown nor absorbed
    const size_t first = l.frontActive ? 1 : 0;
    auto grow          = [](SampleRect& m, const SampleRect& b)
    {
        m.i0 = std::min(m.i0, b.i0);
        m.k0 = std::min(m.k0, b.k0);
        m.i1 = std::max(m.i1, b.i1);
        m.k1 = std::max(m.k1, b.k1);
    };
    // The first box it touches grows; the grown box may touch further ones, before or after it (merged into it, in
    // place, until none touches)
    for (size_t i = first; i < list.size(); ++i)
    {
        if (!touches(list[i], box)) continue;
        grow(list[i], box);
        for (size_t j = first; j < list.size();)
        {
            if (j != i && touches(list[i], list[j]))
            {
                grow(list[i], list[j]);
                list.erase(list.begin() + std::ptrdiff_t(j));
                if (j < i) --i;
                j = first;
            }
            else
                ++j;
        }
        return;
    }
    list.push_back(box);
}

size_t TerrainDirtyQueue::size() const
{
    size_t n = tiles.boxes.size();
    for (const auto& kv : pyramids) n += kv.second.boxes.size();
    return n;
}

bool TerrainDirtyQueue::empty(SpaceId viewSpace) const
{
    auto it = pyramids.find(viewSpace);
    return tiles.boxes.empty() && (it == pyramids.end() || it->second.boxes.empty());
}

}  // namespace Engine
}  // namespace Saiga
