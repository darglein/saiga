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
    ShaderPart::ShaderCodeInjections injection;
    injection.emplace_back(GL_GEOMETRY_SHADER, "#define SHADOW", 2);
    injection.emplace_back(GL_FRAGMENT_SHADER, "#define SHADOW", 2);
    injection.emplace_back(GL_FRAGMENT_SHADER, "#define WRITE_DEPTH", 2);

    shader_default        = shaderLoader.load<MVPShader>("geometry/point_cloud_default.glsl");
    shader_default_shadow = shaderLoader.load<MVPShader>("geometry/point_cloud_default.glsl", injection);

    shader_sphere        = shaderLoader.load<MVPShader>("geometry/point_cloud_sphere.glsl");
    shader_sphere_shadow = shaderLoader.load<MVPShader>("geometry/point_cloud_sphere.glsl", injection);

    shader_disc = shaderLoader.load<MVPShader>("geometry/point_cloud_disc.glsl");
}

void GLPointCloud::render(RenderInfo render_info)
{
    std::shared_ptr<MVPShader> shader;
    switch (render_type)
    {
        case PointRenderType::DEFAULT:
            shader = render_info.render_pass == RenderPass::Shadow ? shader_default_shadow : shader_default;
            glPointSize(point_size);
            break;
        case PointRenderType::SPHERE:
            shader = render_info.render_pass == RenderPass::Shadow ? shader_sphere_shadow : shader_sphere;
            break;
        case PointRenderType::ORIENTED_DISC:
            shader = shader_disc;
            break;
    }

    if (shader->bind())
    {
        if (render_type != PointRenderType::DEFAULT)
        {
            shader->upload(0, point_radius);
        }
        shader->upload(1, (int)cull_backface);
        shader->uploadModel(model);
        buffer.BindAndDraw();
        shader->unbind();
    }
}

void GLPointCloud::imgui()
{
    ImGui::Checkbox("cull_backface", &cull_backface);
    ImGui::InputFloat("point_size", &point_size);
    ImGui::InputFloat("point_radius", &point_radius);
    ImGui::SliderInt("render_type", (int*)&render_type, 0, 2);
}



}  // namespace Saiga
