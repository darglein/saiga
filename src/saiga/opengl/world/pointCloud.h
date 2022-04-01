/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/object3d.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/CubeTexture.h"
#include "saiga/opengl/texture/Texture.h"
#include "saiga/opengl/vertex.h"

namespace Saiga
{

class SAIGA_OPENGL_API GLPointCloud : public Object3D
{
   public:
    enum class PointRenderType
    {
        // Default gl_points rendering.
        // Is affected by point_size
        DEFAULT = 0,

        // Render each point as sphere.
        // Is affected by point_radius
        SPHERE,

        // Render each point as normal-aligned disc
        // Is affected by point_radius
        ORIENTED_DISC,
    };
    PointRenderType render_type = PointRenderType::DEFAULT;

    float point_size   = 3;
    float point_radius = 0.1;

    // call points based on the normal
    bool cull_backface = true;

    std::shared_ptr<MVPShader> shader_default, shader_default_shadow, shader_sphere, shader_sphere_shadow, shader_disc;
    UnifiedMeshBuffer buffer;


    GLPointCloud(const Saiga::UnifiedMesh& point_cloud_mesh);
    void render(RenderInfo render_info);
    void updateBuffer();
    void imgui();
};



}  // namespace Saiga
