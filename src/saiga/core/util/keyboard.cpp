/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/keyboard.h"

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include <iostream>

namespace Saiga
{
Keyboard keyboard(1024);


Keyboard::Keyboard(int numKeys)
{
    keystate.resize(numKeys, 0);
}

int Keyboard::getKeyState(int key)
{
    if (key >= 0 && key < (int)keystate.size())
    {
        return keystate[key];
    }
    else
    {
        std::cerr << "Keyboard::getKeyState Key not found: " << key << std::endl;
        return 0;
    }
}

bool Keyboard::getMappedKeyState(int mappedKey, const std::vector<int>& keymap)
{
    int key;
    if (mappedKey >= 0 && mappedKey < (int)keymap.size())
    {
        key = keymap[mappedKey];
    }
    else
    {
        std::cerr << "Keyboard::getMappedKeyState Keymap entry not found: " << mappedKey << std::endl;
        key = -1;
    }
    return getKeyState(key) > 0;
}


int Keyboard::setKeyState(int key, int state)
{
    if (key >= 0 && key < (int)keystate.size())
    {
        SAIGA_ASSERT(state == 0 || state == 1);
        int old       = keystate[key];
        keystate[key] = state;
        if (old ^ state) return key;
    }
    return -1;
}

void Keyboard::printKeyState()
{
    std::cout << "[";
    for (auto k : keystate)
    {
        std::cout << k << ",";
    }
    std::cout << "]" << std::endl;
}

}  // namespace Saiga
