/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TextureDisplay.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/all.h"

namespace Saiga
{
TextureDisplay::TextureDisplay()
{
    auto tm = TriangleMeshGenerator::createFullScreenQuadMesh();
    buffer.fromMesh(*tm);
    shader = shaderLoader.load<MVPTextureShader>("post_processing/imagedisplay.glsl");
}

void TextureDisplay::render(TextureBase* texture, const ivec2& position, const ivec2& size)
{
    ViewPort vp;
    vp.position = position;
    vp.size     = size;

    setViewPort(vp);


    shader->bind();

    shader->uploadTexture(texture);
    buffer.bindAndDraw();

    shader->unbind();
}

}  // namespace Saiga
