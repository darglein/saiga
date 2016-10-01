#include "saiga/util/mouse.h"
#include "saiga/util/assert.h"

Mouse mouse;


Mouse::Mouse() : Keyboard(32)
{
}


void Mouse::setPosition(const glm::ivec2 &value)
{
    position = value;
}