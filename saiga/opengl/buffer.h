#pragma once

#include "saiga/opengl/opengl.h"

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

class SAIGA_GLOBAL Buffer{
public:
    GLuint buffer = 0; //opengl id
    GLuint size = 0; //size of the buffer in bytes
    GLenum target ; //opengl target. example: GL_ARRAY_BUFFER

    Buffer(GLenum _target );
    ~Buffer();

    void createGLBuffer(void* data=nullptr,unsigned int size=0, GLenum usage=GL_DYNAMIC_DRAW);
    void deleteGLBuffer();

    void updateBuffer(void* data, unsigned int size, unsigned int offset);

    void bind() const;

    /*
     * glMapBuffer and glMapNamedBuffer map the entire data store of a specified buffer object into the client's address space.
     * The data can then be directly read and/or written relative to the returned pointer, depending on the specified access policy.
     */
    void* mapBuffer(GLenum access=GL_READ_WRITE);
    void unmapBuffer();

};

inline Buffer::Buffer(GLenum _target):target(_target)
{

}

inline Buffer::~Buffer()
{
    deleteGLBuffer();
}

inline void Buffer::createGLBuffer(void *data, unsigned int size, GLenum usage)
{
    glGenBuffers( 1, &buffer );
    bind();
    glBufferData(target, size, data, usage);

    this->size = size;


}

inline void Buffer::deleteGLBuffer()
{
    if(buffer){
        glDeleteBuffers( 1, &buffer );
        buffer = 0;
    }
}

inline void Buffer::updateBuffer(void *data, unsigned int size, unsigned int offset)
{
    bind();
    glBufferSubData(target,offset,size,data);
}

inline void Buffer::bind() const
{
    glBindBuffer( target, buffer );
}

inline void *Buffer::mapBuffer(GLenum access)
{
    return glMapBuffer(target,access);
}

inline void Buffer::unmapBuffer()
{
    glUnmapBuffer(target);
}
