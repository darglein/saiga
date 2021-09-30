/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/vertex.h"
#include "saiga/core/model/UnifiedMesh.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/opengl/instancedBuffer.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/templatedBuffer.h"

#include <iostream>
#include <vector>

namespace Saiga
{
class SAIGA_OPENGL_API UnifiedMeshBuffer
{
   public:
    UnifiedMeshBuffer(UnifiedMesh mesh, GLenum draw_mode = GL_TRIANGLES);
    ~UnifiedMeshBuffer();

    UnifiedMeshBuffer(const UnifiedMeshBuffer&) = delete;
    UnifiedMeshBuffer& operator=(UnifiedMeshBuffer const&) = delete;

    void Bind();
    void Unbind();
    void Draw(int offset = 0, int count = std::numeric_limits<int>::max());

    void BindAndDraw()
    {
        Bind();
        Draw();
        Unbind();
    }

    TemplatedBuffer<uint32_t> indices   = {GL_ELEMENT_ARRAY_BUFFER};
    TemplatedBuffer<vec3> position      = {GL_ARRAY_BUFFER};
    TemplatedBuffer<vec3> normal        = {GL_ARRAY_BUFFER};
    TemplatedBuffer<vec4> color         = {GL_ARRAY_BUFFER};
    TemplatedBuffer<vec2> tc            = {GL_ARRAY_BUFFER};
    TemplatedBuffer<BoneInfo> bone_info = {GL_ARRAY_BUFFER};

    // true on indexed face set.
    // false if vertex array is directly drawn.
    bool is_indexed  = true;

    int num_elements = 0;
    int indices_per_element;
    GLuint gl_vao = 0;
    GLenum draw_mode;
};


}  // namespace Saiga
