/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <map>
#include <vector>

namespace Saiga
{
class SAIGA_CORE_API Keyboard
{
   protected:
    // We are using vectors here for faster access.
    // The state of the individual keys. 0 = not pressed, 1 = pressed
    std::vector<int> keystate;

   public:
    Keyboard(int numKeys);
    int getKeyState(int key);

    // An additional mapping used to map actions to buttons.
    // Usage:
    // create an enum for the actions
    // create a map that maps the enum value to a key
    bool getMappedKeyState(int mappedKey, const std::vector<int>& keymap);

    // should not be called by applications
    // return true if the state has changed
    int setKeyState(int key, int state);

    void printKeyState();
};

extern SAIGA_CORE_API Keyboard keyboard;

}  // namespace Saiga
