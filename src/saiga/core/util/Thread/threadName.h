
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <string>
#include <thread>

namespace Saiga
{
/**
 * Sets a thread name for debugging purposes.
 *
 * Basically copy+paste from here: https://stackoverflow.com/questions/10121560/stdthread-naming-your-thread
 * Slight modifications on the interface.
 */
SAIGA_CORE_API extern void setThreadName(const std::string& name);
SAIGA_CORE_API extern void setThreadName(std::thread& thread, const std::string& name);

/**
 * A simple wrapper for std::thread that joins during destruction.
 *
 * This is a simple implementation of joining_thread from the C++ core guidelines:
 * https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rconc-join
 *
 * TODO: Maybe move to different file or rename this file to "Thread.h"
 * Usage:
 *
 * auto thread = ScopedThread([this]() {
 *      doSomething();
 * });
 */
class SAIGA_CORE_API ScopedThread : public std::thread
{
   public:
    template <typename... Args>
    ScopedThread(Args&&... args) : std::thread(std::forward<Args>(args)...)
    {
    }
    ScopedThread(ScopedThread&& __t) { swap(__t); }
    ScopedThread& operator=(ScopedThread&& __t) noexcept;
    ~ScopedThread()
    {
        if (joinable()) join();
    }

   private:
    // Scoped Threads are not allowed to detach.
    // Note: Upcasting ScopedThread to std::thread and then detaching is UB
    using std::thread::detach;
};

}  // namespace Saiga
