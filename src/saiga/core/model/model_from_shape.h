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
// ============= Triangle Meshes =============

SAIGA_CORE_API UnifiedModel FullScreenQuad();


SAIGA_CORE_API UnifiedModel UVSphereMesh(const Sphere& sphere, int rings, int sectors);

SAIGA_CORE_API UnifiedModel IcoSphereMesh(const Sphere& sphere, int resolution);

SAIGA_CORE_API UnifiedModel CylinderMesh(float radius, float height, int sectors);
SAIGA_CORE_API UnifiedModel ConeMesh(const Cone& cone, int sectors);


SAIGA_CORE_API UnifiedModel PlaneMesh(const Plane& plane);

SAIGA_CORE_API UnifiedModel BoxMesh(const AABB& box);



SAIGA_CORE_API UnifiedModel SkyboxMesh(const AABB& box);


SAIGA_CORE_API UnifiedModel CheckerBoardPlane(const ivec2& size, float quadSize, const vec4& color1,
                                              const vec4& color2);


// ============= Line Meshes =============

SAIGA_CORE_API UnifiedModel GridBoxLineMesh(const AABB& box, ivec3 steps);


/**
 * Simple debug grid placed into the x-z plane with y=0.
 *
 * dimension: number of lines in x and z direction.
 * spacing:   distance between lines
 */
SAIGA_CORE_API UnifiedModel GridPlaneLineMesh(const ivec2& dimension, const vec2& spacing);

/**
 * Debug camera frustum.
 * Created by backprojecting the "far-plane corners" of the unit cube.
 * p = inv(proj) * corner
 *
 *
 * @param farPlaneLimit Distance at which the far plane should be drawn. -1 uses the original far plane.
 */
SAIGA_CORE_API UnifiedModel FrustumLineMesh(const mat4& proj, float farPlaneDistance, bool vulkanTransform);

/**
 * Similar to above but uses a computer vision K matrix and the image dimensions.
 */
SAIGA_CORE_API UnifiedModel FrustumCVLineMesh(const mat3& K, float farPlaneDistance, int w, int h);


/**
 * Initializes a K matrix with a 90 degree FOV and creates the frustum.
 */
SAIGA_CORE_API UnifiedModel FrustumLineMesh(const Frustum& frustum);


// Create a simple 2.5D heighmap from an image.
// One vertex is created for every pixel and connected by triangles.
SAIGA_CORE_API UnifiedModel SimpleHeightmap(const ImageView<uint16_t> image, float height_scale, float horizontal_scale,
                                            bool translate_to_origin);



}  // namespace Saiga
