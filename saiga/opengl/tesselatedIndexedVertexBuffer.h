#pragma once

#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/util/assert.h"

/**
 *  Requires OpenGL 4.0+
 */

template<class vertex_t, class index_t>
class TesselatedIndexedVertexBuffer : public IndexedVertexBuffer<vertex_t,index_t>{
protected:
    typedef IndexedVertexBuffer<vertex_t,index_t> ivbuffer_t;

    int patchSize = -1; //number of vertices per patch

public:
    void setPatchSize(int size);

    void bindAndDraw() const;
    void bind() const;



};


template<class vertex_t, class index_t>
void TesselatedIndexedVertexBuffer<vertex_t,index_t>::setPatchSize(int patchSize)
{
    this->patchSize = patchSize;
    this->draw_mode = GL_PATCHES;
}

template<class vertex_t, class index_t>
void TesselatedIndexedVertexBuffer<vertex_t,index_t>::bindAndDraw() const{
    bind();
    ivbuffer_t::draw();
    ivbuffer_t::unbind();
}

template<class vertex_t, class index_t>
void TesselatedIndexedVertexBuffer<vertex_t,index_t>::bind() const{
    assert(patchSize > 0);
    assert(this->draw_mode == GL_PATCHES);
    ivbuffer_t::bind();
    glPatchParameteri(GL_PATCH_VERTICES, patchSize);
    assert_no_glerror();
}
