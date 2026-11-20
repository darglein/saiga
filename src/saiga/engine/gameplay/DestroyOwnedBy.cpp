/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// destroyOwnedBy (S16b). Reads: NetOwner (through ownedBy), Parent. Destroys the peer's roots.

#include "saiga/engine/gameplay/DestroyOwnedBy.h"

#include "saiga/engine/components/Parent.h"
#include "saiga/engine/world/World.h"

#include <vector>

namespace Saiga
{
namespace Engine
{
size_t destroyOwnedBy(World& host, uint8_t peer)
{
    SAIGA_ASSERT(host.isHost(), "destroyOwnedBy: on the host");
    SAIGA_ASSERT(peer != 0, "destroyOwnedBy: peer 0 is the host");
    SAIGA_ASSERT(!host.inStep(), "destroyOwnedBy: between steps");
    std::vector<Entity> roots;
    host.ownedBy<const Transform>(peer).each(
        [&](Entity e, const Transform&)
        {
            if (!host.has<Parent>(e)) roots.push_back(e);
        });
    for (Entity e : roots)
        if (host.alive(e)) host.destroy(e);
    return roots.size();
}

}  // namespace Engine
}  // namespace Saiga
