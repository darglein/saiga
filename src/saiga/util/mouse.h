/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/util/glm.h"
#include <saiga/util/keyboard.h>

#include <map>
#include <vector>

namespace Saiga {

class SAIGA_GLOBAL Mouse : public Keyboard{
protected:
    glm::ivec2 position;
public:
    Mouse();

    glm::ivec2 getPosition() { return position; }
    int getX() { return position.x; }
    int getY() { return position.y; }


    //should not be called by applications
    void setPosition(const glm::ivec2 &value);
};

extern SAIGA_GLOBAL Mouse mouse;

}
