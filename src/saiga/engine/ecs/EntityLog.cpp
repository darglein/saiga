/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SAIGA_LOG_E: log lines about an entity.

#include "saiga/engine/ecs/EntityLog.h"

#include <cstdarg>
#include <cstdio>
#include <string>

namespace Saiga
{
namespace Engine
{
void logPrintfEntity(const char* category, Entity entity, const char* format, ...)
{
    // Every va_start is ended before anything that can throw (allocations)
    va_list args;
    va_start(args, format);
    int size = std::vsnprintf(nullptr, 0, format, args);
    va_end(args);

    std::string message;
    bool ok = size >= 0;
    if (size > 0)
    {
        message.resize(size_t(size) + 1);
        va_start(args, format);
        int written = std::vsnprintf(&message[0], message.size(), format, args);
        va_end(args);
        ok = written == size;
        if (ok) message.resize(size_t(size));
    }
    if (!ok) message = std::string("(invalid log format: ") + format + ")";
    logPrintf(category, "e%u:%u %s", unsigned(entity.index), unsigned(entity.generation), message.c_str());
}

}  // namespace Engine
}  // namespace Saiga
