/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/geometry/line_mesh.h"

namespace Saiga {


class SAIGA_GLOBAL LineModelColored
{
public:
    LineMesh<VertexNC,uint32_t> mesh;

    void createGrid(int numX, int numY, float quadSize=1.0f, vec4 color = vec4(0.5));
    void createFrustum(const mat4& proj, float farPlaneLimit = -1, const vec4& color=vec4(0.5), bool vulkanTransform = false);
};



}
