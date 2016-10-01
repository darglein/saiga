
#pragma once

#include <string>
#include <map>
#include <vector>
#include <saiga/config.h>
#include <saiga/util/glm.h>
#include <saiga/util/keyboard.h>


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


