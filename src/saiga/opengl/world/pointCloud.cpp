/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/pointCloud.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
GLPointCloud::GLPointCloud()
{
    shader_simple = shaderLoader.load<MVPShader>("geometry/colored_points.glsl");
    shader_geometry = shaderLoader.load<MVPShader>("geometry/colored_points_geometry.glsl");

    buffer.setDrawMode(GL_POINTS);
}

void GLPointCloud::render(Camera* cam)
{
    if (buffer.getVAO() == 0) return;
    glPointSize(screen_point_size);

    auto shader = splat_geometry ? shader_geometry : shader_simple;

    shader->bind();

    if(splat_geometry)
    {
        shader->upload(0, world_point_size);
    }

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
void GLPointCloud::imgui()
{
    ImGui::InputFloat("screen_point_size", &screen_point_size);
    ImGui::InputFloat("world_point_size", &world_point_size);



    ImGui::Checkbox("splat_geometry", &splat_geometry);
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
