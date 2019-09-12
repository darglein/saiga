/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/pointCloud.h"

#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
GLPointCloud::GLPointCloud()
{
    shader = shaderLoader.load<MVPShader>("colored_points.glsl");
    buffer.setDrawMode(GL_POINTS);
}

void GLPointCloud::render(Camera* cam)
{
    if (buffer.getVAO() == 0) return;
    glPointSize(pointSize);
    shader->bind();

    shader->uploadModel(model);

    buffer.bindAndDraw();

    shader->unbind();
}

void GLPointCloud::updateBuffer()
{
    if (points.size() > 0)
    {
        buffer.set(points, GL_STATIC_DRAW);
    }
}


template <>
void VertexBuffer<PointVertex>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PointVertex), NULL);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(PointVertex), (void*)(3 * sizeof(GLfloat)));
}

}  // namespace Saiga
