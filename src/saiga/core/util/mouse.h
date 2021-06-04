/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include <map>
#include <saiga/core/util/keyboard.h>
#include <vector>

namespace Saiga
{
class SAIGA_CORE_API Mouse : public Keyboard
{
   protected:
    ivec2 position;

   public:
    Mouse();

    ivec2 getPosition() { return position; }
    int getX() { return position[0]; }
    int getY() { return position[1]; }


    // should not be called by applications
    void setPosition(const ivec2& value);
};

extern SAIGA_CORE_API Mouse mouse;

}  // namespace Saiga
