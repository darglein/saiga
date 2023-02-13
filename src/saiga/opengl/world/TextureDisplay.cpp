/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TextureDisplay.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/all.h"

namespace Saiga
{
TextureDisplay::TextureDisplay() : buffer(FullScreenQuad())
{
    shader = shaderLoader.load<MVPTextureShader>("post_processing/imagedisplay.glsl");
}

/*0: no rotation, 1: right rotation, -1: left rotation*/
void TextureDisplay::render(TextureBase* texture, const ivec2& position, const ivec2& size, bool flip_y, int rotate_result)
{
    ViewPort vp;
    vp.position = position;
    vp.size     = size;

    setViewPort(vp);


    if (shader->bind())
    {
        shader->upload(0, (int)flip_y);
        shader->upload(1, rotate_result);
        shader->uploadTexture(texture);
        buffer.BindAndDraw();

        shader->unbind();
    }
}

}  // namespace Saiga
