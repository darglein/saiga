/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "btBulletDynamicsCommon.h"
#include "bulletDebugDrawer.h"
#include "bulletHelper.h"


namespace Saiga
{
class BulletPhysics
{
   public:
    // ================
    /// the default constraint solver. For parallel processing you can use a different solver (see
    /// Extras/BulletMultiThreaded)
    btSequentialImpulseConstraintSolver solver;

    /// btDbvtBroadphase is a good general purpose broadphase. You can also try out btAxis3Sweep.
    btDbvtBroadphase broadPhase;

    /// collision configuration contains default setup for memory, collision setup. Advanced users can create their own
    /// configuration.
    btDefaultCollisionConfiguration collisionConfiguration;

    /// use the default collision dispatcher. For parallel processing you can use a diffent dispatcher (see
    /// Extras/BulletMultiThreaded)
    btCollisionDispatcher dispatcher;

    btDiscreteDynamicsWorld dynamicsWorld;

    GLDebugDrawer debugDrawer;


    // keep track of the shapes, we release memory at exit.
    // make sure to re-use collision shapes among rigid bodies whenever possible!
    btAlignedObjectArray<btCollisionShape*> collisionShapes;

    BulletPhysics();
    ~BulletPhysics();


    void update();
    void render(Camera* cam);
};


}  // namespace Saiga
