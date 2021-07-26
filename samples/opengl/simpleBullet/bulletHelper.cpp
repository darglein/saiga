/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "bulletHelper.h"

#include <iostream>
namespace Saiga
{
void createCollisionShape(std::vector<Triangle>& mesh, btBvhTriangleMeshShape*& outShape, btTriangleMesh*& outMesh)
{
    if (mesh.size() == 0) return;

    outMesh = new btTriangleMesh();


    for (Triangle& t : mesh)
    {
        outMesh->addTriangle(toBT(t.a), toBT(t.b), toBT(t.c));
    }

    // collision shape
    outShape = new btBvhTriangleMeshShape(outMesh, true);
}

void createConvexCollisionShape(std::vector<Triangle>& mesh, btConvexTriangleMeshShape*& outShape,
                                btTriangleMesh*& outMesh)
{
    if (mesh.size() == 0) return;

    outMesh = new btTriangleMesh();


    for (Triangle& t : mesh)
    {
        outMesh->addTriangle(toBT(t.a), toBT(t.b), toBT(t.c));
    }

    // collision shape
    outShape = new btConvexTriangleMeshShape(outMesh, true);
}

btRigidBody* createRigidBody(btCollisionShape* collisionShape, float mass, vec3 position, quat rotation, float friction)
{
    if (!collisionShape)
    {
        std::cout << "createRigidbody: collision shape is null!" << std::endl;
    }

    // rigidbody is dynamic if and only if mass is non zero, otherwise static
    bool isDynamic = (mass != 0.f);

    btVector3 localInertia(0, 0, 0);

    if (isDynamic) collisionShape->calculateLocalInertia(mass, localInertia);


    btDefaultMotionState* motionState = new btDefaultMotionState(btTransform(toBT(rotation), toBT(position)));
    //    MyMotionState* motionState = new MyMotionState(btTransform(q,v),obj);

    btRigidBody::btRigidBodyConstructionInfo rigidBodyCI(mass, motionState, collisionShape, localInertia);

    rigidBodyCI.m_friction = friction;

    btRigidBody* rigidBody = new btRigidBody(rigidBodyCI);

    return rigidBody;
}

void setRigidBodyState(btRigidBody* rigidBody, vec3 position, quat rotation)
{
    btTransform T(toBT(rotation), toBT(position));
    rigidBody->setWorldTransform(T);
    rigidBody->getMotionState()->setWorldTransform(T);
}



}  // namespace Saiga
