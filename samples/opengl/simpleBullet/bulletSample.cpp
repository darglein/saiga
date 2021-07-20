/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "bulletSample.h"

#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/shader/shaderLoader.h"


Sample::Sample()
{
    cubeAsset = std::make_shared<ColoredAsset>(
        BoxMesh(AABB(vec3(-1, -1, -1), vec3(1, 1, 1))).SetVertexColor(vec4(0.7, 0.7, 0.7, 1)));

    initBullet();
    std::cout << "Program Initialized!" << std::endl;
}

Sample::~Sample()
{
    // We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
}

void Sample::initBullet()
{
    {
        btCollisionShape* groundShape = new btBoxShape(btVector3(btScalar(20.), btScalar(20.), btScalar(20.)));

        physics.collisionShapes.push_back(groundShape);

        btRigidBody* body = createRigidBody(groundShape, 0, vec3(0, -20, 0));

        // add the body to the dynamics world
        physics.dynamicsWorld.addRigidBody(body);
    }

    {
        // create a dynamic rigidbody

        btCollisionShape* colShape = new btBoxShape(btVector3(1, 1, 1));
        // btCollisionShape* colShape = new btSphereShape(btScalar(1.));
        physics.collisionShapes.push_back(colShape);


        for (int i = 0; i < 100; ++i)
        {
            btRigidBody* body =
                createRigidBody(colShape, 1, vec3(0, 10, 0) + linearRand(vec3(-5, 0, -5), vec3(5, 30, 5)));
            physics.dynamicsWorld.addRigidBody(body);

            PhysicAssetObject pao;
            pao.rigidBody = body;
            pao.asset     = cubeAsset;
            cubes.push_back(pao);
        }
    }
}

void Sample::update(float dt)
{
    SampleWindowDeferred::update(dt);

    physics.update();

    for (auto& objects : cubes)
    {
        objects.loadFromRigidbody();
        objects.calculateModel();
    }
}



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
