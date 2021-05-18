/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <map>
#include <saiga/core/util/keyboard.h>
#include <vector>

namespace Saiga
{
class SAIGA_CORE_API Joystick2 : public Keyboard
{
   private:
    int axisCount          = 0;
    int buttonCount        = 0;
    int virtualButtonCount = 0;

    // for every axis two virtual buttons are created
    float virtualButtonThreshold = 0.5f;
    std::vector<float> axis;

    // if and axis value is +- this value it will be clamped to 0
    //    float axisClampThreshold = 0.01f;
   public:
    Joystick2();

    void setCount(int _axisCount, int _buttonCount);
    void setAxisState(int ax, float state);

    int setVirtualAxisKeyState(int ax, float state);

    float getAxisState(int key);

    // An additional mapping used to map actions to buttons.
    // Usage:
    // create an enum for the actions
    // create a map that maps the enum value to a key
    float getMappedAxisState(int mappedKey, const std::vector<int>& keymap);

    void printAxisState();
};



extern SAIGA_CORE_API Joystick2 joystick;

}  // namespace Saiga
