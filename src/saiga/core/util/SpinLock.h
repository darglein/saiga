/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <atomic>
#include <mutex>

namespace Saiga
{
/**
 * A busy waiting spin lock for very small critical sections.
 * Can (and should) be used in std scoped lock wrappers.
 * Usage Example:
 *
 * SpinLock sl;
 * {
 *     std::unique_lock l(sl);
 *     // Critical Section
 * }
 *
 * @brief The SpinLock class
 */
class SAIGA_CORE_API SpinLock
{
   public:
    void lock()
    {
        while (locked.test_and_set(std::memory_order_acquire))
        {
            ;
        }
    }
    void unlock() { locked.clear(std::memory_order_release); }

   private:
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
};

}  // namespace Saiga
