/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/buffer.h"
#include "saiga/core/math/math.h"

namespace Saiga
{
/**
 * Generic class for an instanced buffer used for instance rendering.
 *
 * Every instace gets an object of data_t type.
 * For example, if every instance should be rendered with a different model matrix,
 * an instanced buffer of type InstancedBuffer<mat4> should be used.
 *
 * For all other types the method void setAttributes(int location, int divisor=1); must be defined.
 *
 * Remember: The attribute pointers and divisors are part of the vao state, so setAttributes does not have to be called
 * every frame.
 */


template <typename data_t>
class InstancedBuffer : public Buffer
{
   public:
    int elements;

    InstancedBuffer() : Buffer(GL_ARRAY_BUFFER) {}
    ~InstancedBuffer() {}

    void createGLBuffer(unsigned int elements = 0);
    void updateBuffer(void* data, unsigned int elements, unsigned int offset);

    void setAttributes(int location, int divisor = 1);
};


template <typename data_t>
void InstancedBuffer<data_t>::createGLBuffer(unsigned int elements)
{
    Buffer::createGLBuffer(nullptr, elements * sizeof(data_t), GL_DYNAMIC_DRAW);
    this->elements = elements;
}

template <typename data_t>
void InstancedBuffer<data_t>::updateBuffer(void* data, unsigned int elements, unsigned int offset)
{
    Buffer::updateBuffer(data, elements * sizeof(data_t), offset * sizeof(data_t));
}



template <>
inline void InstancedBuffer<mat4>::setAttributes(int location, int divisor)
{
    Buffer::bind();

    for (unsigned int i = 0; i < 4; i++)
    {
        glEnableVertexAttribArray(location + i);
        glVertexAttribPointer(location + i, 4, GL_FLOAT, GL_FALSE, sizeof(mat4),
                              (const GLvoid*)(sizeof(GLfloat) * i * 4));
        glVertexAttribDivisor(location + i, divisor);
    }
    assert_no_glerror();
}

}  // namespace Saiga
