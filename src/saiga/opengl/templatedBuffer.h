/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/opengl/buffer.h"

#include <vector>

namespace Saiga
{
template <typename T>
class TemplatedBuffer : public Buffer
{
   public:
    TemplatedBuffer(GLenum _target) : Buffer(_target) {}
    ~TemplatedBuffer() {}

    void set(ArrayView<T> data, GLenum usage);


    void fill(const T* data, int count, GLenum usage = GL_STATIC_DRAW);

    void updateBuffer(T* data, int count, int offset);

    int getElementCount() const { return Buffer::size / sizeof(T); }
};



template <typename T>
void TemplatedBuffer<T>::set(ArrayView<T> data, GLenum _usage)
{
    Buffer::createGLBuffer(data.data(), data.size() * sizeof(T), _usage);
}

template <typename T>
void TemplatedBuffer<T>::fill(const T* data, int count, GLenum usage)
{
    Buffer::fill(data, count * sizeof(T), usage);
}


template <typename T>
void TemplatedBuffer<T>::updateBuffer(T* data, int count, int offset)
{
    Buffer::updateBuffer(data, count * sizeof(T), offset * sizeof(T));
}

}  // namespace Saiga
