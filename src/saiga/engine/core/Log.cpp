/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The process-wide engine log: SAIGA_LOG formatting, the ring and the stdout echo.

#include "saiga/engine/core/Log.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <mutex>

namespace Saiga
{
namespace Engine
{
namespace
{
std::atomic<bool> echo{true};

// Ring insertion and stdout echo under one lock: stdout shows the lines in sequence order, without interleaving.
// Never destroyed (like the ring): static objects may log from their destructors at exit.
std::mutex& logMutex()
{
    static std::mutex* m = new std::mutex;
    return *m;
}
}  // namespace

LogRing& engineLog()
{
    // Never destroyed: logging stays valid during static destruction
    static LogRing* ring = new LogRing;
    return *ring;
}

void setLogEcho(bool e)
{
    echo = e;
}

bool logEcho()
{
    return echo;
}

void logPrintf(const char* category, const char* format, ...)
{
    std::string message;
    va_list args;
    va_start(args, format);
    va_list copy;
    va_copy(copy, args);
    int size = std::vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    if (size > 0)
    {
        message.resize(size_t(size) + 1);
        std::vsnprintf(&message[0], message.size(), format, args);
        message.resize(size_t(size));
    }
    else if (size < 0)
    {
        message = std::string("(invalid log format: ") + format + ")";
    }
    va_end(args);

    std::lock_guard<std::mutex> lock(logMutex());
    if (echo) std::cout << "[" << category << "] " << message << std::endl;
    engineLog().add(category, std::move(message));
}

}  // namespace Engine
}  // namespace Saiga
