/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/joystick.h"

#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include <iostream>

namespace Saiga
{
Joystick2 joystick;

Joystick2::Joystick2() : Keyboard(100)
{
    axis.resize(20, 0);
}

void Joystick2::setCount(int _axisCount, int _buttonCount)
{
    axisCount          = _axisCount;
    buttonCount        = _buttonCount;
    virtualButtonCount = buttonCount + axisCount * 2;
    axis.resize(axisCount);
    keystate.resize(virtualButtonCount);
}

void Joystick2::setAxisState(int ax, float state)
{
    SAIGA_ASSERT(state >= -1 && state <= 1);



    if (ax >= 0 && ax < (int)axis.size())
    {
        axis[ax] = state;
    }
    else
    {
        std::cerr << "Joystick::setAxisState Axis not found: " << ax << std::endl;
    }
}

int Joystick2::setVirtualAxisKeyState(int ax, float state)
{
    int key0   = buttonCount + 2 * ax;
    int key1   = buttonCount + 2 * ax + 1;
    int state0 = state > virtualButtonThreshold;
    int state1 = state < -virtualButtonThreshold;

    int r0 = setKeyState(key0, state0);
    int r1 = setKeyState(key1, state1);

    if (r0 != -1 || r1 != -1) return key0;

    return -1;
}

float Joystick2::getAxisState(int ax)
{
    if (ax >= 0 && ax < (int)axis.size())
        return axis[ax];
    else
    {
        std::cerr << "Joystick::getAxisState Axis not found: " << ax << std::endl;
        return 0.0f;
    }
}

float Joystick2::getMappedAxisState(int mappedKey, const std::vector<int>& keymap)
{
    int key;
    if (mappedKey >= 0 && mappedKey < (int)keymap.size())
    {
        key = keymap[mappedKey];
    }
    else
    {
        std::cerr << "Joystick::getMappedAxisState Keymap entry not found: " << mappedKey << std::endl;
        key = -1;
    }

    return getAxisState(key);
}

void Joystick2::printAxisState()
{
    std::cout << "[";
    for (auto k : axis)
    {
        std::cout << k << ",";
    }
    std::cout << "]" << std::endl;
}

}  // namespace Saiga
