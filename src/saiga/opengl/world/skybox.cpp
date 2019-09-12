/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/skybox.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"

namespace Saiga
{
Skybox::Skybox()
{
    AABB bb(vec3(-1), vec3(1));
    auto sb = TriangleMeshGenerator::createSkyboxMesh(bb);
    //    sb->createBuffers(mesh);
    mesh.fromMesh(*sb);
}

void Skybox::setPosition(const vec3& p)
{
    col(model, 3) = vec4(p[0], 0, p[2], 1);
}

void Skybox::setDistance(float d)
{
    col(model, 0)[0] = d;
    col(model, 1)[1] = d;
    col(model, 2)[2] = d;
}


void Skybox::render(Camera* cam)
{
    shader->bind();
    shader->uploadModel(model);
    shader->uploadTexture(cube_texture.get());
    mesh.bindAndDraw();
    cube_texture->unbind();

    shader->unbind();
}

}  // namespace Saiga
