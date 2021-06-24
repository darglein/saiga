/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/error.h"
#include "saiga/opengl/opengl.h"

#include <vector>

namespace Saiga
{
/**
 * @brief The Buffer class
 * A basic wrapper for an OpenGL 'Mutable Buffer Object'.
 * See https://www.opengl.org/wiki/Buffer_Object for more information on Buffer Objects.
 *
 * A Buffer Object is basically a piece of memory (=buffer) on the GPU.
 * Updating the Buffer ( with updateBuffer() ) transfers the data to the allocated GPU memory.
 *
 * The typical usage is storing rendering data on the GPU, to save CPU-GPU bandwith.
 * The most common application is storing the Vertex-Data of a model (see VertexBuffer).
 */

class SAIGA_OPENGL_API Buffer
{
   public:
    GLuint buffer = 0;        // opengl id
    GLuint size   = 0;        // size of the buffer in bytes
    GLenum target = GL_NONE;  // opengl target. example: GL_ARRAY_BUFFER
    GLenum usage  = GL_NONE;


    Buffer(GLenum _target);
    ~Buffer();

    // copy and swap idiom
    // http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
    Buffer(Buffer const& other);
    Buffer& operator=(Buffer other);
    friend void swap(Buffer& first, Buffer& second);

    // usage:
    //    Specifies the expected usage pattern of the data store. The symbolic constant must be
    //    GL_STREAM_DRAW, GL_STREAM_READ, GL_STREAM_COPY,
    //    GL_STATIC_DRAW, GL_STATIC_READ, GL_STATIC_COPY,
    //    GL_DYNAMIC_DRAW, GL_DYNAMIC_READ, or GL_DYNAMIC_COPY.
    void createGLBuffer(const void* data = nullptr, unsigned int size = 0, GLenum usage = GL_STATIC_DRAW);
    void deleteGLBuffer();

    void resize(int new_size);

    void fill2(const void* data, unsigned int size, GLenum usage = GL_STATIC_DRAW);

    void updateBuffer(const void* data, unsigned int size, unsigned int offset);
    void getBuffer(void* out_data, unsigned int _size, unsigned int offset) const;
    void bind() const;
    void bind(GLuint bindingPoint) const;

    /*
     * glMapBuffer and glMapNamedBuffer map the entire data store of a specified buffer object into the client's address
     * space. The data can then be directly read and/or written relative to the returned pointer, depending on the
     * specified access policy.
     */
    void* mapBuffer(GLenum access = GL_READ_WRITE);
    void unmapBuffer();

    GLuint getBufferObject() { return buffer; }
};

inline Buffer::Buffer(GLenum _target) : target(_target) {}

inline Buffer::~Buffer()
{
    deleteGLBuffer();
}

inline Buffer::Buffer(Buffer const& other) : target(other.target)
{
    std::vector<unsigned char> data(other.size);
    other.getBuffer(data.data(), other.size, 0);

    createGLBuffer(data.data(), other.size, other.usage);
}

inline Buffer& Buffer::operator=(Buffer other)
{
    swap(*this, other);
    return *this;
}

inline void swap(Buffer& first, Buffer& second)
{
    using std::swap;
    swap(first.buffer, second.buffer);
    swap(first.size, second.size);
    swap(first.target, second.target);
    swap(first.usage, second.usage);
}

inline void Buffer::createGLBuffer(const void* data, unsigned int _size, GLenum _usage)
{
    size  = _size;
    usage = _usage;

    deleteGLBuffer();
    glGenBuffers(1, &buffer);
    assert_no_glerror();
    bind();
    glBufferData(target, size, data, usage);

    assert_no_glerror();
}

inline void Buffer::deleteGLBuffer()
{
    if (buffer)
    {
        glDeleteBuffers(1, &buffer);
        buffer = 0;
        assert_no_glerror();
    }
}

inline void Buffer::fill2(const void* data, unsigned int _size, GLenum _usage)
{
    size  = _size;
    usage = _usage;

    if (buffer)
    {
        bind();
        glBufferData(target, size, data, usage);
    }
    else
    {
        createGLBuffer(data, size, usage);
    }
    assert_no_glerror();
}

inline void Buffer::updateBuffer(const void* data, unsigned int _size, unsigned int offset)
{
    if (_size == 0) return;
    SAIGA_ASSERT(offset + _size <= size);
    bind();
    glBufferSubData(target, offset, _size, data);
    assert_no_glerror();
}

inline void Buffer::getBuffer(void* out_data, unsigned int _size, unsigned int offset) const
{
    SAIGA_ASSERT(offset + _size <= size);
    bind();
    glGetBufferSubData(target, offset, _size, out_data);
    assert_no_glerror();
}

inline void Buffer::bind() const
{
    glBindBuffer(target, buffer);
    assert_no_glerror();
}


inline void Buffer::bind(GLuint bindingPoint) const
{
    glBindBufferBase(target, bindingPoint, buffer);
    assert_no_glerror();
}

inline void* Buffer::mapBuffer(GLenum access)
{
    void* ptr = glMapBuffer(target, access);
    assert_no_glerror();
    return ptr;
}

inline void Buffer::unmapBuffer()
{
    glUnmapBuffer(target);
    assert_no_glerror();
}
inline void Buffer::resize(int new_size)
{
    SAIGA_ASSERT(usage != GL_NONE);
    if (new_size > size)
    {
        size = new_size;
        bind();
        glBufferData(target, size, nullptr, usage);
    }
}

}  // namespace Saiga
