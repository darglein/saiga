/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "threadName.h"

#include "saiga/core/util/assert.h"
#ifdef __APPLE__
#    include <pthread.h>
#endif

#ifdef __APPLE__
static void SetThreadName(uint32_t dwThreadID, const char* threadName) {}
#elif _WIN32
#    include <windows.h>

const DWORD MS_VC_EXCEPTION = 0x406D1388;

#    pragma pack(push, 8)
typedef struct tagTHREADNAME_INFO
{
    DWORD dwType;      // Must be 0x1000.
    LPCSTR szName;     // Pointer to name (in user addr space).
    DWORD dwThreadID;  // Thread ID (-1=caller thread).
    DWORD dwFlags;     // Reserved for future use, must be zero.
} THREADNAME_INFO;
#    pragma pack(pop)


static void SetThreadName(uint32_t dwThreadID, const char* threadName)
{
    // DWORD dwThreadID = ::GetThreadId( static_cast<HANDLE>( t.native_handle() ) );

    THREADNAME_INFO info;
    info.dwType     = 0x1000;
    info.szName     = threadName;
    info.dwThreadID = dwThreadID;
    info.dwFlags    = 0;

    __try
    {
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
    }
}


#else

#    include <sys/prctl.h>
#endif


namespace Saiga
{
void setThreadName(const std::string& name)
{
#ifdef __APPLE__
    pthread_setname_np(name.c_str());
#elif _WIN32

    SetThreadName(GetCurrentThreadId(), name.c_str());
#else
    prctl(PR_SET_NAME, name.c_str(), 0, 0, 0);
#endif
}

void setThreadName(std::thread& thread, const std::string& name)
{
#ifdef __APPLE__
    // macOS can only name the calling thread. Calling pthread_setname_np here would rename the caller instead of
    // 'thread', so this is a no-op. Call setThreadName(name) from inside the thread instead.
    (void)thread;
    (void)name;
#elif _WIN32
    DWORD threadId = ::GetThreadId(static_cast<HANDLE>(thread.native_handle()));
    SetThreadName(threadId, name.c_str());
#else
    // pthread_setname_np fails for names longer than 15 characters (prctl above truncates them).
    auto handle = thread.native_handle();
    pthread_setname_np(handle, name.substr(0, 15).c_str());
#endif
}

ScopedThread& ScopedThread::operator=(ScopedThread&& __t) noexcept
{
    if (joinable())
    {
        SAIGA_EXIT_ERROR("Cannot assign to a running thread.");
    }
    std::thread::swap(__t);
    terminate_called.swap(__t.terminate_called);
    return *this;
}


}  // namespace Saiga
