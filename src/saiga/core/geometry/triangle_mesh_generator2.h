/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/model/UnifiedModel.h"

#include "cone.h"
#include "plane.h"
#include "sphere.h"

#include <memory>

#include "triangle_mesh.h"
namespace Saiga
{
SAIGA_CORE_API UnifiedModel FullScreenQuad();


SAIGA_CORE_API UnifiedModel UVSphereMesh(const Sphere& sphere, int rings, int sectors);

SAIGA_CORE_API UnifiedModel IcoSphereMesh(const Sphere& sphere, int resolution);

SAIGA_CORE_API UnifiedModel CylinderMesh(float radius, float height, int sectors);
SAIGA_CORE_API UnifiedModel ConeMesh(const Cone& cone, int sectors);


SAIGA_CORE_API UnifiedModel PlaneMesh(const Plane& plane);

SAIGA_CORE_API UnifiedModel BoxMesh(const AABB& box);

SAIGA_CORE_API UnifiedModel SkyboxMesh(const AABB& box);

}  // namespace Saiga
