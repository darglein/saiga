/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/line_mesh.h"

namespace Saiga
{
class SAIGA_CORE_API LineModelColored
{
   public:
    LineMesh<VertexNC, uint32_t> mesh;

    void createGrid(int numX, int numY, float quadSize = 1.0f, vec4 color = make_vec4(0.5));
    void createFrustum(const mat4& proj, float farPlaneLimit = -1, const vec4& color = make_vec4(0.5),
                       bool vulkanTransform = false);

    void createFrustumCV(float farPlaneLimit, const vec4& color, int w, int h);
    void createFrustumCV(const mat3& K, float farPlaneLimit, const vec4& color, int w, int h);
};



}  // namespace Saiga
