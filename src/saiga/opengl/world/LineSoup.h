/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "pointCloud.h"

namespace Saiga
{
/**
 * Each lines consists of 2 vertices (no line strip!!)
 * That means num_lines = num_vertices / 2
 *
 * @brief The LineSoup class
 */
class SAIGA_OPENGL_API LineSoup : public Object3D
{
   public:
    std::vector<PointVertex> lines;
    int lineWidth = 1;

    LineSoup();
    void render(Camera* cam);
    void updateBuffer();

   private:
    std::shared_ptr<MVPShader> shader;
    VertexBuffer<PointVertex> buffer;
};


}  // namespace Saiga
