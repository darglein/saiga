/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/keyboard.h"

namespace Saiga
{
namespace Game
{
// Keyboard helpers of the tech demos (key codes are GLFW key codes; the samples pass them). Transitional: replaced by
// the engine's input action map (S13, docs/engine_design.md §8.3).
inline bool keyDown(int key)
{
    return key >= 0 && keyboard().getKeyState(key) > 0;
}

// -1 if only the negative key is down, +1 if only the positive one, else 0
inline double keyAxis(int negativeKey, int positiveKey)
{
    return (keyDown(positiveKey) ? 1.0 : 0.0) - (keyDown(negativeKey) ? 1.0 : 0.0);
}

}  // namespace Game
}  // namespace Saiga
