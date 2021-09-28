/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/pointCloud.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
GLPointCloud::GLPointCloud(const Saiga::UnifiedMesh& point_cloud_mesh) : buffer(point_cloud_mesh, GL_POINTS)
{
    shader_simple   = shaderLoader.load<MVPShader>("geometry/colored_points.glsl");
    shader_geometry = shaderLoader.load<MVPShader>("geometry/colored_points_geometry.glsl");
}

void GLPointCloud::render(Camera* cam)
{
    glPointSize(screen_point_size);

    auto shader = splat_geometry ? shader_geometry : shader_simple;

    if (shader->bind())
    {
        if (splat_geometry)
        {
            shader->upload(0, world_point_size);
        }

        shader->uploadModel(model);

        buffer.BindAndDraw();

        shader->unbind();
    }
}

void GLPointCloud::imgui()
{
    ImGui::InputFloat("screen_point_size", &screen_point_size);
    ImGui::InputFloat("world_point_size", &world_point_size);



    ImGui::Checkbox("splat_geometry", &splat_geometry);
}



}  // namespace Saiga
