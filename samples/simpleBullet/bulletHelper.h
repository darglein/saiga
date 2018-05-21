/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "btBulletDynamicsCommon.h"

#include "saiga/geometry/triangle.h"
#include <vector>


namespace Saiga
{

/**
 * Creates a collision shape from a trianglemesh.
 * This shape can only be used for static objects.
 */
void createCollisionShape(std::vector<Triangle> &mesh,btBvhTriangleMeshShape* &outShape, btTriangleMesh* &outMesh);

/**
 * Creates a convex collision shape from a triangle mesh.
 * Can be used for dynamic and static objects.
 */
void createConvexCollisionShape(std::vector<Triangle> &mesh, btConvexTriangleMeshShape* &outShape, btTriangleMesh* &outMesh);

/**
 * Creates a rigidbody with the collission shape and the given parameters.
 *
 */
btRigidBody *createRigidBody(
        btCollisionShape *collisionShape,
        float mass = 0.0f,
        vec3 position=vec3(0), quat rotation=quat(1,0,0,0),
        float friction=1.0f
        );

void setRigidBodyState(btRigidBody* rigidBody, vec3 position, quat rotation);


/**
 * Conversion functions: glm->bullet
 */
inline btQuaternion toBT(const glm::quat& q){return btQuaternion(q.x,q.y,q.z,q.w);}
inline btVector3 toBT(const glm::vec3& v){return btVector3(v.x,v.y,v.z);}

/**
 * Conversion functions: bullet->glm
 */
inline glm::quat toGLM(const btQuaternion& q){return quat(q.w(),q.x(),q.y(),q.z());}
inline vec3 toGLM(const btVector3& v){return vec3(v.x(),v.y(),v.z());}

}
