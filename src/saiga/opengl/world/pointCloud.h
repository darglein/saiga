/**
 * Copyright (c) 2017 Darius Rückert
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



class SAIGA_OPENGL_API GLPointCloud : public Object3D
{
   public:
    float pointSize = 3;
    std::vector<PointVertex> points;

    GLPointCloud();
    void render(Camera* cam);
    void updateBuffer();

   private:
    std::shared_ptr<MVPShader> shader;
    VertexBuffer<PointVertex> buffer;
};


template <>
SAIGA_OPENGL_API void VertexBuffer<PointVertex>::setVertexAttributes();

}  // namespace Saiga
