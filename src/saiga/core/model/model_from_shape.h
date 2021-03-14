/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/all.h"
#include "saiga/core/model/UnifiedModel.h"



namespace Saiga
{
SAIGA_CORE_API UnifiedModel FullScreenQuad();


SAIGA_CORE_API UnifiedModel UVSphereMesh(const Sphere& sphere, int rings, int sectors);

SAIGA_CORE_API UnifiedModel IcoSphereMesh(const Sphere& sphere, int resolution);

SAIGA_CORE_API UnifiedModel CylinderMesh(float radius, float height, int sectors);
SAIGA_CORE_API UnifiedModel ConeMesh(const Cone& cone, int sectors);


SAIGA_CORE_API UnifiedModel PlaneMesh(const Plane& plane);

SAIGA_CORE_API UnifiedModel BoxMesh(const AABB& box);

// LineMesh!!!
SAIGA_CORE_API UnifiedModel GridBoxMesh(const AABB& box, ivec3 steps);

SAIGA_CORE_API UnifiedModel SkyboxMesh(const AABB& box);


SAIGA_CORE_API UnifiedModel CheckerBoardPlane(const ivec2& size, float quadSize, const vec4& color1,
                                              const vec4& color2);



}  // namespace Saiga
