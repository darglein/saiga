/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/opengl.h"

namespace Saiga
{
template <class buffer_t>
class CombinedBuffer
{
   public:
    int count  = 0;
    int offset = 0;

    buffer_t* buffer;

    void draw() const;
    void bindAndDraw() const;
};

template <class buffer_t>
void CombinedBuffer<buffer_t>::draw() const
{
    buffer->draw(count, offset);
}

template <class buffer_t>
void CombinedBuffer<buffer_t>::bindAndDraw() const
{
    buffer->bind();
    buffer->draw(count, offset);
    buffer->unbind();
}

}  // namespace Saiga
