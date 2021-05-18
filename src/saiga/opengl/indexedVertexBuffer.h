/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/LineMesh.h"
#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/core/model/UnifiedModel.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/opengl/UnifiedMeshBuffer.h"
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

    /**
     * @brief Draw a few indices from this buffer. See glDrawElements for more informations.
     * @param length number of indices to draw. -1 to render the complete buffer
     * @param offset
     */
    void draw(int length = -1, int offset = 0) const;


    /**
     * @brief drawInstanced
     * Draw multiple instances of this buffer. It is recommended to have an instancedBuffer bound to the VAO with
     * VertexBuffer::addInstancedBuffer.
     *
     * See glDrawElementsInstancedBaseInstance for more informations.
     *
     * @param instanceCount Number of instances to draw
     * @param baseInstance Start instance offset into the instanceBuffers
     * @param indexOffset Start Vertex Index for mesh rendering
     * @param indexCount Number of indices to render. -1 to render the complete buffer
     */
    void drawInstanced(int instanceCount, int baseInstance = 0, int indexOffset = 0, int indexCount = -1) const;

    void set(ArrayView<vertex_t> vertices, ArrayView<index_t> indices, GLenum usage);

    /*
     * Creates OpenGL buffer from indices and vertices
     * 'buffer' is now ready to draw.
     */
    void fromMesh(TriangleMesh<vertex_t, index_t>& mesh, GLenum usage = GL_STATIC_DRAW);

    template <typename buffer_vertex_t, typename buffer_index_t>
    void fromMesh(TriangleMesh<buffer_vertex_t, buffer_index_t>& mesh, GLenum usage = GL_STATIC_DRAW);

    /*
     * Creates OpenGL buffer from indices and vertices
     * 'buffer' is now ready to draw.
     */
    void fromMesh(LineMesh<vertex_t, index_t>& mesh, GLenum usage = GL_STATIC_DRAW);



    void fromMesh(const UnifiedMesh& model, GLenum usage = GL_STATIC_DRAW);

    /*
     * Updates OpenGL buffer with the data currently saved in this mesh
     * see VertexBuffer::updateVertexBuffer for more details
     */
    void updateFromMesh(TriangleMesh<vertex_t, index_t>& buffer, int vertex_count, int vertex_offset);

    int IndexCount() { return ibuffer_t::Size(); }

    int VertexCount() { return vbuffer_t::Size(); }
};

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::bindAndDraw() const
{
    bind();
    draw();
    unbind();
}

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::draw(int length, int offset) const
{
    glDrawElements(vbuffer_t::draw_mode, length < 0 ? ibuffer_t::Size() : length, ibuffer_t::GLType::value,
                   (void*)(intptr_t)(offset * sizeof(index_t)));
    assert_no_glerror();
}



template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::drawInstanced(int instanceCount, int baseInstance, int indexOffset,
                                                           int indexCount) const
{
    glDrawElementsInstancedBaseInstance(
        vbuffer_t::draw_mode, indexCount < 0 ? ibuffer_t::Size() : indexCount, ibuffer_t::GLType::value,
        (void*)(intptr_t)(indexOffset * sizeof(index_t)), instanceCount, baseInstance);
    assert_no_glerror();
}


template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::bind() const
{
    vbuffer_t::bind();
}

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::unbind() const
{
    vbuffer_t::unbind();
}

template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::deleteGLBuffer()
{
    vbuffer_t::deleteGLBuffer();
    ibuffer_t::deleteGLBuffer();
}



template <class vertex_t, class index_t>
void IndexedVertexBuffer<vertex_t, index_t>::set(ArrayView<vertex_t> vertices, ArrayView<index_t> indices, GLenum usage)
{
    vbuffer_t::set(vertices, usage);
    ibuffer_t::create(indices, usage);

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
void IndexedVertexBuffer<vertex_t, index_t>::fromMesh(LineMesh<vertex_t, index_t>& mesh, GLenum usage)
{
    ArrayView<index_t> indices((index_t*)mesh.lines.data(), mesh.lines.size() * 2);
    set(mesh.vertices, indices, usage);
    this->setDrawMode(GL_LINES);
}


template <typename vertex_t, typename index_t>
void IndexedVertexBuffer<vertex_t, index_t>::fromMesh(const UnifiedMesh& model, GLenum usage)
{
    auto mesh = model.Mesh<vertex_t, index_t>();
    fromMesh(mesh, usage);
}


template <typename vertex_t, typename index_t>
void IndexedVertexBuffer<vertex_t, index_t>::updateFromMesh(TriangleMesh<vertex_t, index_t>& mesh, int vertex_count,
                                                            int vertex_offset)
{
    SAIGA_ASSERT((int)mesh.vertices.size() >= vertex_offset + vertex_count);
    VertexBuffer<vertex_t>::updateBuffer(&mesh.vertices[vertex_offset], vertex_count, vertex_offset);
}

}  // namespace Saiga
