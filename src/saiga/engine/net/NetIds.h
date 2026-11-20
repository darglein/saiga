/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Network message IDs (docs/engine_design.md §13.6; S15): engine messages 0..63 (S16 adds them), game messages from 64.

#pragma once

#include <cstdint>

namespace Saiga
{
namespace Engine
{
using MessageId = uint16_t;

enum class EngineMessage : MessageId
{
    FirstGame = 64,
};

}  // namespace Engine
}  // namespace Saiga
