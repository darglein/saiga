/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/core/LogRing.h"
#include "saiga/export.h"

// printf format checking of logPrintf (GCC, Clang)
#if defined(__GNUC__) || defined(__clang__)
#    define SAIGA_ENGINE_PRINTF_FORMAT(f, a) __attribute__((format(printf, f, a)))
#else
#    define SAIGA_ENGINE_PRINTF_FORMAT(f, a)
#endif

namespace Saiga
{
namespace Engine
{
// The process-wide log ring (docs/engine_design.md §17.1); the MCP tool logs(since) reads it (S14)
SAIGA_ENGINE_API LogRing& engineLog();

// Echo every line to stdout as "[category] message" (default on)
SAIGA_ENGINE_API void setLogEcho(bool echo);
SAIGA_ENGINE_API bool logEcho();

// Formats the message (printf style, any length) and adds it to engineLog()
SAIGA_ENGINE_API void logPrintf(const char* category, const char* format, ...) SAIGA_ENGINE_PRINTF_FORMAT(2, 3);

}  // namespace Engine
}  // namespace Saiga

// SAIGA_LOG("streaming", "chunk %d loaded", id): a line in the engine log ring and on stdout
#define SAIGA_LOG(category, ...) ::Saiga::Engine::logPrintf(category, __VA_ARGS__)
