#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/opengl/vertexBuffer.h"
#include "saiga/opengl/indexBuffer.h"
#include <iostream>


template<class vertex_t, class index_t>
class IndexedVertexBuffer : public VertexBuffer<vertex_t>, public IndexBuffer<index_t>{
public:
    typedef VertexBuffer<vertex_t> vbuffer_t;
    typedef IndexBuffer<index_t> ibuffer_t;


    void bind() const;
    void unbind() const;

    void bindAndDraw() const;
    void draw() const;
    void draw(unsigned int length, void* offset) const;


    void drawInstanced(int instances) const;
    void drawInstanced(int instances, int offset, int length) const;

    void set(std::vector<vertex_t> &vertices,std::vector<index_t> &indices);
    void set(vertex_t* vertices,int vertex_count,index_t* indices,int index_count);
};

template<class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t,index_t>::bindAndDraw() const{
    bind();
    draw();
    unbind();
}

template<class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t,index_t>::draw() const{
    draw(ibuffer_t::index_count,nullptr);
}

template<class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t,index_t>::draw(unsigned int length, void *offset) const{
    glDrawElements( vbuffer_t::draw_mode, length, ibuffer_t::GLType::value, offset );
    assert_no_glerror();
}


template<class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t,index_t>::drawInstanced(int instances) const
{
    drawInstanced(instances,0,ibuffer_t::index_count);
    assert_no_glerror();
}

template<class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t,index_t>::drawInstanced(int instances, int offset, int length) const
{
    glDrawElementsInstanced(vbuffer_t::draw_mode,length,ibuffer_t::GLType::value,(void*)offset,instances);
    assert_no_glerror();
}



template<class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t,index_t>::bind() const{
    vbuffer_t::bind();
    //    ibuffer_t::bind();
}

template<class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t,index_t>::unbind() const{
    vbuffer_t::unbind();
    //    ibuffer_t::unbind();
}


template<class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t,index_t>::set(std::vector<vertex_t> &vertices, std::vector<index_t> &indices){
    set(&vertices[0],vertices.size(),&indices[0],indices.size());
}

template<class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t,index_t>::set(vertex_t *vertices, int _vertex_count, index_t *indices, int _index_count){
    vbuffer_t::set(vertices,_vertex_count);
    ibuffer_t::set(indices,_index_count);

    //The ELEMENT_ARRAY_BUFFER_BINDING is part of VAO state.
    //adds index buffer to vao state
    vbuffer_t::bind();
    ibuffer_t::bind();
    vbuffer_t::unbind();
    ibuffer_t::unbind();
}
