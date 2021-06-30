/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/skybox.h"

#include "saiga/core/model/model_from_shape.h"

namespace Saiga
{
Skybox::Skybox() : mesh(SkyboxMesh(AABB(make_vec3(-1), make_vec3(1)))) {}

void Skybox::setPosition(const vec3& p)
{
    model.col(3) = vec4(p[0], 0, p[2], 1);
}

void Skybox::setDistance(float d)
{
    model(0, 0) = d;
    model(1, 1) = d;
    model(2, 2) = d;
}


void Skybox::render(Camera* cam)
{
    if(shader->bind())
    {
        shader->uploadModel(model);
        shader->uploadModel(mat4::Identity());
        shader->uploadTexture(cube_texture.get());
        mesh.BindAndDraw();
        cube_texture->unbind();

        shader->unbind();
    }
}

}  // namespace Saiga
