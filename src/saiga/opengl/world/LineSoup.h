/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/object3d.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/CubeTexture.h"
#include "saiga/opengl/texture/Texture.h"
#include "saiga/opengl/vertex.h"

namespace Saiga
{

struct SAIGA_OPENGL_API PointVertex
{
    vec3 position;
    vec3 color;
};


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

template <>
SAIGA_OPENGL_API void VertexBuffer<PointVertex>::setVertexAttributes();

}  // namespace Saiga
