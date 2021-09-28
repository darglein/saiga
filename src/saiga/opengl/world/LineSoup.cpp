/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "LineSoup.h"

#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{


template <>
void VertexBuffer<PointVertex>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PointVertex), NULL);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(PointVertex), (void*)(3 * sizeof(GLfloat)));
}
LineSoup::LineSoup()
{
    shader = shaderLoader.load<MVPShader>("geometry/colored_points.glsl");
    buffer.setDrawMode(GL_LINES);
}

void LineSoup::render(Camera* cam)
{
    if (buffer.getVAO() == 0) return;
    glLineWidth(lineWidth);
    if(shader->bind())
    {
        shader->uploadModel(model);

        buffer.bindAndDraw();

        shader->unbind();
    }
}

void LineSoup::updateBuffer()
{
    SAIGA_ASSERT(lines.size() % 2 == 0);
    if (lines.size() > 0) buffer.set(lines, GL_STATIC_DRAW);
}


}  // namespace Saiga
