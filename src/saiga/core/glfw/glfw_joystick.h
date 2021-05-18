/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/joystick.h"

namespace Saiga
{
struct SAIGA_CORE_API glfw_Joystick
{
   private:
    static int joystickId;

   public:
    static void update();

    static void enableFirstJoystick();

    static void joystick_callback(int joy, int event);


    static bool isEnabled() { return joystickId != -1; }
};

}  // namespace Saiga
