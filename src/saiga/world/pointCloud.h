/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/vertex.h"
#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/texture/cube_texture.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/rendering/object3d.h"

namespace Saiga {

struct SAIGA_GLOBAL PointVertex {
    vec3 position;
    vec3 color;
};



class SAIGA_GLOBAL GLPointCloud : public Object3D
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


template<>
SAIGA_GLOBAL void VertexBuffer<PointVertex>::setVertexAttributes();

}
