/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// destroyOwnedBy: the host's removal of a leaving peer's entities (S16b; a mechanism the game's leave policy calls,
// R-MECH-1).

#pragma once

#include "saiga/export.h"

#include <cstddef>
#include <cstdint>

namespace Saiga
{
namespace Engine
{
class World;

// Host, between steps: destroys every root owned by peer (NetOwner, §13.3) with its descendants (onDestroy runs; the
// clients get the Destroy). Host-owned entities (vehicles a peer drove) stay. Returns the number of roots destroyed.
// Peer 0 (the host) is refused (asserted).
SAIGA_ENGINE_API size_t destroyOwnedBy(World& host, uint8_t peer);

}  // namespace Engine
}  // namespace Saiga
