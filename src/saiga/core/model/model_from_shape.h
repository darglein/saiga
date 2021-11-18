/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/all.h"
#include "saiga/core/model/UnifiedMesh.h"



namespace Saiga
{
// ============= Triangle Meshes =============

SAIGA_CORE_API UnifiedMesh FullScreenQuad();


SAIGA_CORE_API UnifiedMesh UVSphereMesh(const Sphere& sphere, int rings, int sectors);

SAIGA_CORE_API UnifiedMesh IcoSphereMesh(const Sphere& sphere, int resolution);

SAIGA_CORE_API UnifiedMesh CylinderMesh(float radius, float height, int sectors);
SAIGA_CORE_API UnifiedMesh ConeMesh(const Cone& cone, int sectors);

// Creates a triangle mesh from 3 colored cylinders
SAIGA_CORE_API UnifiedMesh CoordinateSystemMesh(float scale = 1, bool add_sphere = false);


SAIGA_CORE_API UnifiedMesh PlaneMesh(const Plane& plane);

SAIGA_CORE_API UnifiedMesh BoxMesh(const AABB& box);



SAIGA_CORE_API UnifiedMesh SkyboxMesh(const AABB& box);


SAIGA_CORE_API UnifiedMesh CheckerBoardPlane(const ivec2& size, float quadSize, const vec4& color1, const vec4& color2);


// ============= Line Meshes =============

SAIGA_CORE_API UnifiedMesh GridBoxLineMesh(const AABB& box, ivec3 steps = ivec3(1, 1, 1));


/**
 * Simple debug grid placed into the x-z plane with y=0.
 *
 * dimension: number of lines in x and z direction.
 * spacing:   distance between lines
 */
SAIGA_CORE_API UnifiedMesh GridPlaneLineMesh(const ivec2& dimension, const vec2& spacing);

/**
 * Debug camera frustum.
 * Created by backprojecting the "far-plane corners" of the unit cube.
 * p = inv(proj) * corner
 *
 *
 * @param farPlaneLimit Distance at which the far plane should be drawn. -1 uses the original far plane.
 */
SAIGA_CORE_API UnifiedMesh FrustumLineMesh(const mat4& proj, float farPlaneDistance, bool vulkanTransform);

/**
 * Similar to above but uses a computer vision K matrix and the image dimensions.
 */
SAIGA_CORE_API UnifiedMesh FrustumCVLineMesh(const mat3& K, float farPlaneDistance, int w, int h);


/**
 * Initializes a K matrix with a 90 degree FOV and creates the frustum.
 */
SAIGA_CORE_API UnifiedMesh FrustumLineMesh(const Frustum& frustum);


// Create a simple 2.5D heighmap from an image.
// One vertex is created for every pixel and connected by triangles.
SAIGA_CORE_API UnifiedMesh SimpleHeightmap(const ImageView<uint16_t> image, float height_scale, float horizontal_scale,
                                           bool translate_to_origin);



}  // namespace Saiga
