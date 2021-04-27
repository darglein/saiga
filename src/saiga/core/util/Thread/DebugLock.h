/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"

#include <atomic>
#include <mutex>
#include <set>
#include <thread>
namespace Saiga
{
/**
 * A debug lock, which derives from an actual lock. This provides the additional member function IsLockedByCaller().
 */
template <typename MutexType>
class DebugLock : private MutexType
{
   public:
    void lock()
    {
        MutexType::lock();
        Acquire();
    }
    bool try_lock()
    {
        if (MutexType::try_lock())
        {
            Acquire();
            return true;
        }
        return false;
    }
    void unlock()
    {
        Release();
        MutexType::unlock();
    }

    void lock_shared()
    {
        MutexType::lock_shared();
        AcquireShared();
    }
    void unlock_shared()
    {
        ReleaseShared();
        MutexType::unlock_shared();
    }

    bool IsLockedByCaller() { return owner == std::this_thread::get_id(); }

    bool IsLockedByCallerShared()
    {
        std::unique_lock l(debug_lock);
        // either full lock or shared lock is fine
        return owner == std::this_thread::get_id() || owners.find(std::this_thread::get_id()) != owners.end();
    }

   private:
    void Acquire()
    {
        std::unique_lock l(debug_lock);

        SAIGA_ASSERT(owner != std::this_thread::get_id(), "double lock");
        SAIGA_ASSERT(owners.find(std::this_thread::get_id()) == owners.end(), "double shared lock");
        owner = std::this_thread::get_id();
    }
    void Release()
    {
        std::unique_lock l(debug_lock);
        SAIGA_ASSERT(std::this_thread::get_id() != std::thread::id());
        SAIGA_ASSERT(owner == std::this_thread::get_id(), "double release");
        owner = std::thread::id();
    }

    void AcquireShared()
    {
        std::unique_lock l(debug_lock);

        SAIGA_ASSERT(owner != std::this_thread::get_id(), "double lock");
        SAIGA_ASSERT(owners.find(std::this_thread::get_id()) == owners.end(), "double shared lock");
        owners.insert(std::this_thread::get_id());
    }
    void ReleaseShared()
    {
        std::unique_lock l(debug_lock);
        SAIGA_ASSERT(owners.find(std::this_thread::get_id()) != owners.end(), "double shared lock");
        owners.erase(std::this_thread::get_id());
    }
    std::atomic<std::thread::id> owner = std::thread::id();
    std::set<std::thread::id> owners;
    std::mutex debug_lock;
};

}  // namespace Saiga
