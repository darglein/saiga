/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/mouse.h"
#include "saiga/util/assert.h"
#include "internal/noGraphicsAPI.h"

namespace Saiga {

Mouse mouse;


Mouse::Mouse() : Keyboard(32)
{
}


void Mouse::setPosition(const glm::ivec2 &value)
{
    position = value;
}

}
