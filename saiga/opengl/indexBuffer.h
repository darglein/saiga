#pragma once

#include <type_traits>
#include "saiga/opengl/buffer.h"


#include <vector>

/**
 * Converting the template argument to a GLenum with template spezialisation.
 * Only GL_UNSIGNED_BYTE,GL_UNSIGNED_SHORT and GL_UNSIGNED_INT are allowed by the GL specification.
 * Using IndexBuffer with another type will result in compile time errors.
 */

template<class index_t> struct IndexGLType;

template<> struct IndexGLType<GLubyte>
{ static const GLenum value = GL_UNSIGNED_BYTE;};
template<> struct IndexGLType<GLushort>
{ static const GLenum value = GL_UNSIGNED_SHORT;};
template<> struct IndexGLType<GLuint>
{ static const GLenum value = GL_UNSIGNED_INT;};

template<class index_t>
class IndexBuffer : public Buffer{
    static_assert(std::is_integral<index_t>::value && std::is_unsigned<index_t>::value,
                   "Only unsigned integral types allowed!");
    static_assert(sizeof(index_t)==1 || sizeof(index_t)==2 || sizeof(index_t)==4,
                   "Only 1,2 and 4 byte index types allowed!");
public:
    int index_count;
    typedef IndexGLType<index_t> GLType;

    IndexBuffer(): Buffer(GL_ELEMENT_ARRAY_BUFFER){}
    ~IndexBuffer(){}

    void set(std::vector<index_t> &indices);
    void set(index_t* indices,int index_count);

    void unbind() const;

};




template<class index_t>
void IndexBuffer<index_t>::set(std::vector<index_t> &indices){
    set(&indices[0],indices.size());
}

template<class index_t>
void IndexBuffer<index_t>::set(index_t* indices,int index_count){

    this->index_count = index_count;

    createGLBuffer(indices,index_count * sizeof(index_t),GL_STATIC_DRAW);

}

template<class index_t>
void IndexBuffer<index_t>::unbind() const{
    //Binding 0 is deprecated!

    // Application-generated object names - the names of all object types, such as buffer, query, and texture objects,
    // must be generated using the corresponding Gen* commands. Trying to bind an object name not returned by a Gen*command will
    // result in an INVALID_OPERATION error. This behavior is already the case for framebuffer, renderbuffer, and vertex array objects.
    // Object types which have default objects (objects named zero), such as vertex array, framebuffer, and texture objects,
    // may also bind the default object, even though it is not returned by Gen*.

    //      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
}
