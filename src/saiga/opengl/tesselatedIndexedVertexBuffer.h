/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/core/util/assert.h"

namespace Saiga
{
/**
 *  Requires OpenGL 4.0+
 */

template <class vertex_t, class index_t>
class TesselatedIndexedVertexBuffer : public IndexedVertexBuffer<vertex_t, index_t>
{
   protected:
    typedef IndexedVertexBuffer<vertex_t, index_t> ivbuffer_t;

    int patchSize = -1;  // number of vertices per patch

   public:
    void setPatchSize(int size);

    void bindAndDraw() const;
    void bind() const;
};


template <class vertex_t, class index_t>
void TesselatedIndexedVertexBuffer<vertex_t, index_t>::setPatchSize(int patchSize)
{
    this->patchSize = patchSize;
    this->draw_mode = GL_PATCHES;
}

template <class vertex_t, class index_t>
void TesselatedIndexedVertexBuffer<vertex_t, index_t>::bindAndDraw() const
{
    bind();
    ivbuffer_t::draw();
    ivbuffer_t::unbind();
}

template <class vertex_t, class index_t>
void TesselatedIndexedVertexBuffer<vertex_t, index_t>::bind() const
{
    SAIGA_ASSERT(patchSize > 0);
    SAIGA_ASSERT(this->draw_mode == GL_PATCHES);
    ivbuffer_t::bind();
    glPatchParameteri(GL_PATCH_VERTICES, patchSize);
    assert_no_glerror();
}

}  // namespace Saiga
