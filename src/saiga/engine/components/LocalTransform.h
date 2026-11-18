/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"

namespace Saiga
{
namespace Engine
{
// The pose of a child relative to its Parent (docs/engine_design.md §7.4). Without it a child sits at its parent's
// pose. World pose = parent position + parent rotation * (parent scale * position), rotation = parent rotation *
// rotation, scale = parent scale * scale (component-wise).
struct LocalTransform
{
    static constexpr const char* name = "LocalTransform";
    static constexpr ComponentId id   = ComponentId(EngineComponent::LocalTransform);
    static constexpr uint32_t version = 1;

    Vec3 position = Vec3::Zero();
    Quat rotation = Quat::Identity();
    Vec3 scale    = Vec3::Ones();

    template <class V>
    void visit(V& v)
    {
        v("position", position);
        v("rotation", rotation);
        v("scale", scale);
    }
};

}  // namespace Engine
}  // namespace Saiga
