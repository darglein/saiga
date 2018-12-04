/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/geometry/triangle_mesh.h"
#include "saiga/opengl/indexBuffer.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/vertexBuffer.h"

namespace Saiga
{
template <class vertex_t, class index_t>
class IndexedVertexBuffer : public VertexBuffer<vertex_t>, public IndexBuffer<index_t>
{
   public:
    typedef VertexBuffer<vertex_t> vbuffer_t;
    typedef IndexBuffer<index_t> ibuffer_t;


    void bind() const;
    void unbind() const;
    void deleteGLBuffer();

    void bindAndDraw() const;
    void draw() const;
    void draw(unsigned int length, int offset) const;


    void drawInstanced(int instances) const;
    void drawInstanced(int instances, int offset, int length) const;

    void set(std::vector<vertex_t>& vertices, std::vector<index_t>& indices, GLenum usage);
    void set(vertex_t* vertices, int vertex_count, index_t* indices, int index_count, GLenum usage);

    /*
     * Creates OpenGL buffer from indices and vertices
     * 'buffer' is now ready to draw.
     */

    void fromMesh(TriangleMesh<vertex_t, index_t>& mesh, GLenum usage = GL_STATIC_DRAW);

    template <typename buffer_vertex_t, typename buffer_index_t>
    void fromMesh(TriangleMesh<buffer_vertex_t, buffer_index_t>& mesh, GLenum usage = GL_STATIC_DRAW);

    /*
     * Updates OpenGL buffer with the data currently saved in this mesh
     * see VertexBuffer::updateVertexBuffer for more details
     */

    void updateFromMesh(TriangleMesh<vertex_t, index_t>& buffer, int vertex_count, int vertex_offset);
};

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::bindAndDraw() const
{
    bind();
    draw();
    unbind();
}

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::draw() const
{
    draw(ibuffer_t::getElementCount(), 0);
}

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::draw(unsigned int length, int offset) const
{
    glDrawElements(vbuffer_t::draw_mode, length, ibuffer_t::GLType::value, (void*)(intptr_t)(offset * sizeof(index_t)));
    assert_no_glerror();
}


template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::drawInstanced(int instances) const
{
    drawInstanced(instances, 0, ibuffer_t::getElementCount());
    assert_no_glerror();
}

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::drawInstanced(int instances, int offset, int length) const
{
    glDrawElementsInstanced(vbuffer_t::draw_mode, length, ibuffer_t::GLType::value, (void*)(intptr_t)offset, instances);
    assert_no_glerror();
}



template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::bind() const
{
    vbuffer_t::bind();
    //    ibuffer_t::bind();
}

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::unbind() const
{
    vbuffer_t::unbind();
    //    ibuffer_t::unbind();
}

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::deleteGLBuffer()
{
    vbuffer_t::deleteGLBuffer();
    ibuffer_t::deleteGLBuffer();
}



template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::set(std::vector<vertex_t>& vertices, std::vector<index_t>& indices,
                                                 GLenum usage)
{
    set(&vertices[0], vertices.size(), &indices[0], indices.size(), usage);
}

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::set(vertex_t* vertices, int _vertex_count, index_t* indices,
                                                 int _index_count, GLenum usage)
{
    vbuffer_t::set(vertices, _vertex_count, usage);
    ibuffer_t::set(indices, _index_count, usage);

    // The ELEMENT_ARRAY_BUFFER_BINDING is part of VAO state.
    // adds index buffer to vao state
    vbuffer_t::bind();
    ibuffer_t::bind();
    vbuffer_t::unbind();
    ibuffer_t::unbind();
}


template <typename vertex_t, typename index_t>
void IndexedVertexBuffer<vertex_t, index_t>::fromMesh(TriangleMesh<vertex_t, index_t>& mesh, GLenum usage)
{
    if (mesh.faces.empty() || mesh.vertices.empty()) return;
    std::vector<index_t> indices(mesh.faces.size() * 3);
    std::memcpy(&indices[0], &mesh.faces[0], mesh.faces.size() * sizeof(index_t) * 3);
    set(mesh.vertices, indices, usage);
    this->setDrawMode(GL_TRIANGLES);
}

template <typename buffer_vertex_t, typename buffer_index_t>
template <typename vertex_t, typename index_t>
void IndexedVertexBuffer<buffer_vertex_t, buffer_index_t>::fromMesh(TriangleMesh<vertex_t, index_t>& mesh, GLenum usage)
{
    if (mesh.faces.empty() || mesh.vertices.empty()) return;
    std::vector<index_t> indices(mesh.faces.size() * 3);
    std::memcpy(&indices[0], &mesh.faces[0], mesh.faces.size() * sizeof(index_t) * 3);

    // convert index_t to buffer_index_t
    std::vector<buffer_index_t> bufferIndices(indices.begin(), indices.end());

    // convert vertex_t to buffer_vertex_t
    std::vector<buffer_vertex_t> bufferVertices(mesh.vertices.begin(), mesh.vertices.end());

    set(bufferVertices, bufferIndices, usage);
    this->setDrawMode(GL_TRIANGLES);
}


template <typename vertex_t, typename index_t>
void IndexedVertexBuffer<vertex_t, index_t>::updateFromMesh(TriangleMesh<vertex_t, index_t>& mesh, int vertex_count,
                                                            int vertex_offset)
{
    SAIGA_ASSERT((int)mesh.vertices.size() >= vertex_offset + vertex_count);
    VertexBuffer<vertex_t>::updateBuffer(&mesh.vertices[vertex_offset], vertex_count, vertex_offset);
}

}  // namespace Saiga
