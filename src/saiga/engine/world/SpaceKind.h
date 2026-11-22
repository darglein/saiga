/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SpaceKind (docs/engine_design.md §7.9, §12.4; S20, S21; its own header from S23a): the kind of a space template, in
// a header that generator code can include (content code never includes the template header itself).

#pragma once

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// OpenWorld: a height-map space (terrain from its generator, flat without one); Enclosed: no terrain, its content is
// authored (prefabs) around the site origin (S21, S23a)
enum class SpaceKind : uint8_t
{
    OpenWorld,
    Enclosed
};

}  // namespace Engine
}  // namespace Saiga
