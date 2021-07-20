/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "physics.h"

#include "btBulletDynamicsCommon.h"


namespace Saiga
{
BulletPhysics::BulletPhysics()
    : dispatcher(&collisionConfiguration), dynamicsWorld(&dispatcher, &broadPhase, &solver, &collisionConfiguration)
{
    dynamicsWorld.setGravity(btVector3(0, -10, 0));
    dynamicsWorld.setDebugDrawer(&debugDrawer);
}

BulletPhysics::~BulletPhysics()
{
    ///-----cleanup_start-----

    // remove the rigidbodies from the dynamics world and delete them
    for (int i = dynamicsWorld.getNumCollisionObjects() - 1; i >= 0; i--)
    {
        btCollisionObject* obj = dynamicsWorld.getCollisionObjectArray()[i];
        btRigidBody* body      = btRigidBody::upcast(obj);
        if (body && body->getMotionState())
        {
            delete body->getMotionState();
        }
        dynamicsWorld.removeCollisionObject(obj);
        delete obj;
    }

    // delete collision shapes
    for (int j = 0; j < collisionShapes.size(); j++)
    {
        btCollisionShape* shape = collisionShapes[j];
        collisionShapes[j]      = 0;
        delete shape;
    }
}

void BulletPhysics::update()
{
    dynamicsWorld.stepSimulation(1.f / 60.f, 10);
}

void BulletPhysics::render(Camera* cam)
{
    debugDrawer.render(&dynamicsWorld, cam);
}

}  // namespace Saiga
