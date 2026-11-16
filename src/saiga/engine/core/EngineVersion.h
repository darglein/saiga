/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
// Version of the engine module. Bumped when the engine's save format or network protocol changes (from S11 / S26 the
// schema hash is used for compatibility; this is the human readable version).
SAIGA_ENGINE_API const char* engineVersion();

}  // namespace Engine
}  // namespace Saiga
