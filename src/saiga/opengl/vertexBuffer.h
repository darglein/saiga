/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/vertex.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/opengl/instancedBuffer.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/templatedBuffer.h"

#include <iostream>
#include <vector>

namespace Saiga
{
/*
 *  A combination of a gl_array_buffer and a vertex array object (vao).
 * The VertexBuffer stores raw vertex data.
 *  If your mesh is indexed use IndexedVertexBuffer instead.
 *
 * You can add additional buffers to the vao like this:
 *
 * VertexBuffer<Particle> particleBuffer;
 * TemplatedBuffer<vec4> colorBuffer;
 *
 * //init both buffers with data
 * //...
 *
 * //Add the color buffer, which is a simple GL_ARRAY_BUFFER, to the location 3
 * //of the particleBuffer VAO.
 * particleBuffer.bind();
 * colorBuffer.bind();
 * glEnableVertexAttribArray( 3 );
 * glVertexAttribPointer(3,4, GL_FLOAT, GL_FALSE, sizeof(vec4), NULL );
 * particleBuffer.unbind();
 *
 *
 *
 */

template <class vertex_t>
class VertexBuffer : public TemplatedBuffer<vertex_t>
{
   public:
    /*
     *  Create VertexBuffer object.
     *  Does not create any OpenGL buffers.
     *  use set() to initialized buffers.
     */
    VertexBuffer() : TemplatedBuffer<vertex_t>(GL_ARRAY_BUFFER) {}
    ~VertexBuffer() { deleteGLBuffer(); }


    /*
     *  Creates OpenGL buffers and uploads data.
     *  'vertices' can be deleted after that.
     *
     *  A VBO and a VAO will be created and initialized.
     */
    void set(ArrayView<vertex_t> vertices, GLenum usage);

    /*
     *  Updates the existing OpenGL buffer.
     *  Have to be called after 'set()'.
     *
     *  Replaces the vertex at 'vertex_offset' and the following 'vertex_count'
     *  vertices in the current buffer by uploading them to OpenGL.
     *  'vertices' can be deleted after that.
     */

    //    void updateVertexBuffer(vertex_t* vertices,int vertex_count, int vertex_offset);

    /*
     *  Deletes all OpenGL buffers.
     *  Will be called by the destructor automatically
     */
    void deleteGLBuffer();

    /*
     *  Binds/Unbinds OpenGL buffers.
     */
    void bind() const;
    void unbind() const;

    /**
     *  Adds an instanced buffer to this vertex array object.
     *  While rendering the instanced buffer does not have to be bound again.
     *
     * 'location' is the shader location of the instanced uniform. With the following example shader you have to pass in
     * '4'.
     * Example Shader:
     *
     * //...
     * layout(location = 4) in mat4 instanceModel;
     * //...
     * gl_Position = viewProj * instanceModel * vec4(in_position, 1);
     */
    template <typename data_t>
    void addInstancedBuffer(InstancedBuffer<data_t>& buffer, int location, int divisor = 1);

    /*
     *  Draws the vertex array in the specified draw mode.
     *  Uses consecutive vertices to form the specified primitive.
     *  E.g. if the draw mode is GL_TRIANGLES
     *  1.Triangle =  (Vertex 0, Vertex 1, Vertex 2)
     *  2.Triangle =  (Vertex 3, Vertex 4, Vertex 5)
     *  ...
     *
     *  If you want to use one vertex for multiple faces use IndexedVertexBuffer instead.
     */
    void draw(int startVertex = 0, int count = -1) const;

    void drawInstanced(int instanceCount, int indexOffset = 0, int indexCount = -1) const;


    /*
     *  1. bind()
     *  2. draw()
     *  3. unbind()
     */
    void bindAndDraw() const;

    /*
     *  Set draw type of primitives.
     *
     *  Allowed values:
     *  GL_POINTS, GL_LINE_STRIP, GL_LINE_LOOP, GL_LINES, GL_LINE_STRIP_ADJACENCY,
     *  GL_LINES_ADJACENCY, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_TRIANGLES,
     *  GL_TRIANGLE_STRIP_ADJACENCY, GL_TRIANGLES_ADJACENCY, GL_PATCHES
     */
    void setDrawMode(GLenum _draw_mode) { draw_mode = _draw_mode; }
    GLenum getDrawMode() { return draw_mode; }

    GLuint getVBO() { return TemplatedBuffer<vertex_t>::buffer; }
    GLuint getVAO() { return gl_vao; }

    /**
     * Adds an external buffer to this VAO.
     * Usefull for 'structure of arrays' vertex rendering.
     */
    void addExternalBuffer(Buffer& buffer, GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride,
                           const void* pointer = nullptr);

   protected:
    GLenum draw_mode;
    GLuint gl_vao = 0;

    /*
     *  Tells OpenGL how to handle the vertices.
     *  If you are not using the default Vertex types, you have to
     *  specialize this methode.
     *
     *  A specialized methode should look like this:
     *
     *  //Enable attributes
     *  glEnableVertexAttribArray( 0 );
     *  //Set Attribute Pointers
     *  glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), NULL );
     *  ...
     */

    void setVertexAttributes();
};



template <class vertex_t>
void VertexBuffer<vertex_t>::bindAndDraw() const
{
    bind();
    draw();
    unbind();
}



template <class vertex_t>
void VertexBuffer<vertex_t>::set(ArrayView<vertex_t> vertices, GLenum _usage)
{
    //    this->vertex_count = _vertex_count;

    deleteGLBuffer();
    assert_no_glerror();

    TemplatedBuffer<vertex_t>::create(vertices, _usage);
    //    createGLBuffer(vertices,_vertex_count * sizeof(vertex_t),usage);

    // create VAO and init
    glGenVertexArrays(1, &gl_vao);
    SAIGA_ASSERT(gl_vao);

    glBindVertexArray(gl_vao);
    assert_no_glerror();

    Buffer::bind();

    setVertexAttributes();

    glBindVertexArray(0);
    glBindBuffer(TemplatedBuffer<vertex_t>::target, 0);

    assert_no_glerror();
}

template <class vertex_t>
void VertexBuffer<vertex_t>::deleteGLBuffer()
{
    // glDeleteBuffers silently ignores 0's and names that do not correspond to existing buffer objects
    TemplatedBuffer<vertex_t>::deleteGLBuffer();
    if (gl_vao)
    {
        glDeleteVertexArrays(1, &gl_vao);
        gl_vao = 0;
        assert_no_glerror();
    }
}


template <class vertex_t>
void VertexBuffer<vertex_t>::bind() const
{
    glBindVertexArray(gl_vao);
    assert_no_glerror();
}

template <class vertex_t>
void VertexBuffer<vertex_t>::unbind() const
{
    glBindVertexArray(0);
    assert_no_glerror();
}

template <class vertex_t>
template <typename data_t>
void VertexBuffer<vertex_t>::addInstancedBuffer(InstancedBuffer<data_t>& buffer, int location, int divisor)
{
    bind();
    buffer.setAttributes(location, divisor);
    unbind();
}

template <class vertex_t>
void VertexBuffer<vertex_t>::draw(int startVertex, int count) const
{
    glDrawArrays(draw_mode, startVertex, count < 0 ? TemplatedBuffer<vertex_t>::Size() : count);
    assert_no_glerror();
}


template <class vertex_t>
void VertexBuffer<vertex_t>::drawInstanced(int instanceCount, int indexOffset, int indexCount) const
{
    glDrawArraysInstanced(draw_mode, indexOffset,
                          indexCount < 0 ? TemplatedBuffer<vertex_t>::Size() : indexCount, instanceCount);
    assert_no_glerror();
}

template <class vertex_t>
void VertexBuffer<vertex_t>::addExternalBuffer(Buffer& buffer, GLuint index, GLint size, GLenum type,
                                               GLboolean normalized, GLsizei stride, const void* pointer)
{
    bind();
    buffer.bind();
    glEnableVertexAttribArray(index);
    glVertexAttribPointer(index, size, type, normalized, stride, pointer);
}



//=========================================================================

template <class vertex_t>
void VertexBuffer<vertex_t>::setVertexAttributes()
{
    std::cerr
        << "Warning: I don't know how to bind this Vertex Type. Please use the vertices in vertex.h or write your own "
           "bind function"
        << std::endl;
    std::cerr << "If you want to write your own bind function use this template:" << std::endl;
    std::cerr << "\ttemplate<>" << std::endl;
    std::cerr << "\tvoid VertexBuffer<YOUR_VERTEX_TYPE>::bindVertices(){" << std::endl;
    std::cerr << "\t\t//bind code" << std::endl;
    std::cerr << "\t}" << std::endl;
    SAIGA_ASSERT(0);
}



template <>
SAIGA_OPENGL_API void VertexBuffer<Vertex>::setVertexAttributes();
template <>
SAIGA_OPENGL_API void VertexBuffer<VertexN>::setVertexAttributes();
template <>
SAIGA_OPENGL_API void VertexBuffer<VertexC>::setVertexAttributes();
template <>
SAIGA_OPENGL_API void VertexBuffer<VertexNT>::setVertexAttributes();
template <>
SAIGA_OPENGL_API void VertexBuffer<VertexNC>::setVertexAttributes();


}  // namespace Saiga
