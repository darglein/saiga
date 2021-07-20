/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/skybox.h"

#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
Skybox::Skybox(std::shared_ptr<Texture> texture, const std::string mapping) : mesh(FullScreenQuad()), texture(texture)
{
    if (mapping == "spherical")
    {
        shader = shaderLoader.load<MVPTextureShader>("geometry/skybox_image.glsl");
        type   = 0;
    }
}


void Skybox::render(Camera* cam)
{
    SAIGA_ASSERT(shader);
    if (shader->bind())
    {
        // shader->uploadModel(model);
        // shader->uploadModel(mat4::Identity());
        if (type == 0)
        {
            shader->uploadTexture(texture.get());
            mesh.BindAndDraw();
            // cube_texture->unbind();
        }
        shader->unbind();
    }
}

}  // namespace Saiga
