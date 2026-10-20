
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <atomic>
#include <memory>
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
    // template <typename... Args>
    // ScopedThread(Args&&... args) : std::thread(std::forward<Args>(args)...)
    // {
    // }
    ScopedThread() = default;

    // The function receives a pointer to the terminate flag, which is set by Quit().
    // The flag is allocated on the heap, because the thread keeps using it after the ScopedThread has been moved.
    // The thread is started in the body, so the flag is initialized before the thread can read it.
    template <typename T>
    ScopedThread(T function)
    {
        std::thread t(function, terminate_called.get());
        std::thread::swap(t);
    }
    ScopedThread(ScopedThread&& __t)
    {
        std::thread::swap(__t);
        terminate_called.swap(__t.terminate_called);
    }
    ScopedThread& operator=(ScopedThread&& __t) noexcept;
    ~ScopedThread() { Quit(); }

    void Quit()
    {
        *terminate_called = true;
        if (joinable()) join();
    }

   private:
    std::unique_ptr<std::atomic<bool>> terminate_called = std::make_unique<std::atomic<bool>>(false);


    // Scoped Threads are not allowed to detach.
    // Note: Upcasting ScopedThread to std::thread and then detaching is UB
    using std::thread::detach;
};

}  // namespace Saiga
