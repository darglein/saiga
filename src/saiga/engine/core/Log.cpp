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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>

namespace Saiga
{
namespace Engine
{
namespace
{
std::atomic<bool> echo{true};
std::atomic<bool> mcpEcho{[]
                          {
                              const char* e = std::getenv("SAIGA_MCP_ECHO");
                              return e && std::strcmp(e, "1") == 0;
                          }()};

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

void setMcpLogEcho(bool e)
{
    mcpEcho = e;
}

bool mcpLogEcho()
{
    return mcpEcho;
}

namespace
{
// Line breaks of the stdout echo escaped: one stdout line per log call (the ring keeps the original text)
std::string oneLine(const std::string& s)
{
    std::string r;
    r.reserve(s.size());
    for (char c : s)
    {
        if (c == '\n')
            r += "\\n";
        else if (c == '\r')
            r += "\\r";
        else
            r += c;
    }
    return r;
}
}  // namespace

void logPrintf(const char* category, const char* format, ...)
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
        // The second pass can fail too (e.g. an allocation inside the formatter)
        ok = written == size;
        if (ok) message.resize(size_t(size));
    }
    if (!ok) message = std::string("(invalid log format: ") + format + ")";

    std::lock_guard<std::mutex> lock(logMutex());
    // Into the ring before anything else can throw (building or writing the echo): the line is kept either way
    if (!echo || (!mcpEcho && std::strcmp(category, "mcp") == 0))
    {
        engineLog().add(category, std::move(message));
        return;
    }
    engineLog().add(category, message);
    std::cout << "[" << oneLine(category) << "] " << oneLine(message) << std::endl;
}

}  // namespace Engine
}  // namespace Saiga
