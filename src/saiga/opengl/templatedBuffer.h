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

    void create(ArrayView<T> data, GLenum usage) { Buffer::fill2(data.data(), data.size() * sizeof(T), usage); }

    void update(T* data, int count, int offset = 0)
    {
        Buffer::updateBuffer(data, count * sizeof(T), offset * sizeof(T));
    }

    void update(ArrayView<T> data, int offset = 0)
    {
        Buffer::updateBuffer(data.data(), data.size() * sizeof(T), offset * sizeof(T));
    }

    int Size() const { return Buffer::size / sizeof(T); }
};



}  // namespace Saiga
