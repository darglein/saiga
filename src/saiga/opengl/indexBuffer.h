/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/templatedBuffer.h"

#include <vector>

#include <type_traits>

namespace Saiga
{
/**
 * Converting the template argument to a GLenum with template spezialisation.
 * Only GL_UNSIGNED_BYTE,GL_UNSIGNED_SHORT and GL_UNSIGNED_INT are allowed by the GL specification.
 * Using IndexBuffer with another type will result in compile time errors.
 */

template <class index_t>
struct IndexGLType;

// on desktop pcs unsigned_short is faster than unsigned_byte
// using unsigned_byte can create performance warnings on ati cards.
template <>
struct IndexGLType<GLubyte>
{
    static const GLenum value = GL_UNSIGNED_BYTE;
};
template <>
struct IndexGLType<GLushort>
{
    static const GLenum value = GL_UNSIGNED_SHORT;
};
template <>
struct IndexGLType<GLuint>
{
    static const GLenum value = GL_UNSIGNED_INT;
};

template <class index_t>
class IndexBuffer : public TemplatedBuffer<index_t>
{
    static_assert(std::is_integral<index_t>::value && std::is_unsigned<index_t>::value,
                  "Only unsigned integral types allowed!");
    static_assert(sizeof(index_t) == 1 || sizeof(index_t) == 2 || sizeof(index_t) == 4,
                  "Only 1,2 and 4 byte index types allowed!");

   public:
    typedef IndexGLType<index_t> GLType;

    IndexBuffer() : TemplatedBuffer<index_t>(GL_ELEMENT_ARRAY_BUFFER) {}
    ~IndexBuffer() {}

    void unbind() const;
};



template <class index_t>
inline void IndexBuffer<index_t>::unbind() const
{
    // Binding 0 is deprecated!

    // Application-generated object names - the names of all object types, such as buffer, query, and texture objects,
    // must be generated using the corresponding Gen* commands. Trying to bind an object name not returned by a
    // Gen*command will result in an INVALID_OPERATION error. This behavior is already the case for framebuffer,
    // renderbuffer, and vertex array objects. Object types which have default objects (objects named zero), such as
    // vertex array, framebuffer, and texture objects, may also bind the default object, even though it is not returned
    // by Gen*.

    //      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
}

}  // namespace Saiga
