/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/mouse.h"

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
Mouse mouse;


Mouse::Mouse() : Keyboard(32) {}


void Mouse::setPosition(const ivec2& value)
{
    position = value;
}

}  // namespace Saiga
