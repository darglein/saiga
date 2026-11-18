/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
// Like logPrintf, with the message prefixed by the entity: "e<index>:<generation> message"
SAIGA_ENGINE_API void logPrintfEntity(const char* category, Entity entity, const char* format, ...)
    SAIGA_ENGINE_PRINTF_FORMAT(3, 4);

}  // namespace Engine
}  // namespace Saiga

// SAIGA_LOG_E("ai", e, "flees from %u", attacker.index): a log line about an entity (docs/engine_design.md §17.1)
#define SAIGA_LOG_E(category, entity, ...) ::Saiga::Engine::logPrintfEntity(category, entity, __VA_ARGS__)
