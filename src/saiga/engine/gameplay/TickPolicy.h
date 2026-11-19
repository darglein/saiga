/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

namespace Saiga
{
namespace Engine
{
/**
 * How often game objects tick, by grid-cell rings around the players (docs/engine_design.md §8.4; game policy,
 * R-MECH-1). The x/z plane is divided into square cells of ringSize meters; the ring of an object is the Chebyshev
 * distance in cells between its cell and the nearest player's cell in the same space:
 *   ring <= nearRing  every step (default: the 3 x 3 cells around the player, 96 m)
 *   ring <= midRing   every midInterval seconds (default: 13 x 13 cells, about 200 m)
 *   otherwise         every farInterval seconds (also: another space, or no player)
 * WorldConfig::ticking is the default; a class overrides the rings and intervals with GameObject::tickPolicy() (the
 * cell size is always the world's).
 */
struct TickPolicy
{
    double ringSize    = 32;
    int nearRing       = 1;
    int midRing        = 6;
    double midInterval = 0.1;
    double farInterval = 0.5;
};

}  // namespace Engine
}  // namespace Saiga
