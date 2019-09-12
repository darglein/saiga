/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "LineSoup.h"

#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
LineSoup::LineSoup()
{
    shader = shaderLoader.load<MVPShader>("colored_points.glsl");
    buffer.setDrawMode(GL_LINES);
}

void LineSoup::render(Camera* cam)
{
    if (buffer.getVAO() == 0) return;
    glLineWidth(lineWidth);
    shader->bind();

    shader->uploadModel(model);

    buffer.bindAndDraw();

    shader->unbind();
}

void LineSoup::updateBuffer()
{
    SAIGA_ASSERT(lines.size() % 2 == 0);
    if (lines.size() > 0) buffer.set(lines, GL_STATIC_DRAW);
}


}  // namespace Saiga
