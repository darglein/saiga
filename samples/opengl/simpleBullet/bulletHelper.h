/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/core/geometry/triangle.h"

#include "btBulletDynamicsCommon.h"

#include <vector>


namespace Saiga
{
/**
 * Creates a collision shape from a trianglemesh.
 * This shape can only be used for static objects.
 */
void createCollisionShape(std::vector<Triangle>& mesh, btBvhTriangleMeshShape*& outShape, btTriangleMesh*& outMesh);

/**
 * Creates a convex collision shape from a triangle mesh.
 * Can be used for dynamic and static objects.
 */
void createConvexCollisionShape(std::vector<Triangle>& mesh, btConvexTriangleMeshShape*& outShape,
                                btTriangleMesh*& outMesh);

/**
 * Creates a rigidbody with the collission shape and the given parameters.
 *
 */
btRigidBody* createRigidBody(btCollisionShape* collisionShape, float mass = 0.0f, vec3 position = make_vec3(0),
                             quat rotation = quat::Identity(), float friction = 1.0f);

void setRigidBodyState(btRigidBody* rigidBody, vec3 position, quat rotation);


/**
 * Conversion functions: glm->bullet
 */
inline btQuaternion toBT(const quat& q)
{
    vec4 v = quat_to_vec4(q);
    return btQuaternion(v[0], v[1], v[2], v[3]);
}
inline btVector3 toBT(const vec3& v)
{
    return btVector3(v[0], v[1], v[2]);
}

/**
 * Conversion functions: bullet->glm
 */
inline quat toGLM(const btQuaternion& q)
{
    return quat(q.w(), q.x(), q.y(), q.z());
}
inline vec3 toGLM(const btVector3& v)
{
    return vec3(v.x(), v.y(), v.z());
}

}  // namespace Saiga
