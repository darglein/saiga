/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/crash.h"

#include "saiga/core/util/assert.h"

#include <iostream>


#if defined(_WIN32)
#    include <signal.h>
#    include <stdio.h>
#    include <stdlib.h>
#    include <tchar.h>
#    include <windows.h>
// dbghelp must be included after windows.h
#    include <DbgHelp.h>
#endif


#if defined(__unix__)
#    include <execinfo.h>
#    include <signal.h>
#    include <stdio.h>
#    include <stdlib.h>
#    include <string.h>
#    include <ucontext.h>
#    include <unistd.h>
#endif
#include "internal/noGraphicsAPI.h"
namespace Saiga
{
std::function<void()> customCrashHandler;

void addCustomSegfaultHandler(std::function<void()> fnc)
{
    customCrashHandler = fnc;
}

#if defined(__unix__)
// Source: http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes

/* This structure mirrors the one found in /usr/include/asm/ucontext.h */
typedef struct _sig_ucontext
{
    unsigned long uc_flags;
    struct ucontext* uc_link;
    stack_t uc_stack;
    struct sigcontext uc_mcontext;
    sigset_t uc_sigmask;
} sig_ucontext_t;


void printCurrentStack()
{
    void* array[50];
    char** messages;
    int size, i;

    size = backtrace(array, 50);

    /* overwrite sigaction with caller's address */
    //    array[1] = caller_address;

    messages = backtrace_symbols(array, size);

    /* skip first stack frame (points here) */
    for (i = 0; i < size && messages != NULL; ++i)
    {
        std::cout << "[bt]: (" << i << ") " << messages[i] << std::endl;
    }

    free(messages);
}

void crit_err_hdlr(int sig_num, siginfo_t* info, void* ucontext)
{
    void* caller_address;
    sig_ucontext_t* uc;

    uc = (sig_ucontext_t*)ucontext;

    /* Get the address at the time the signal was raised */
#    if defined(__i386__)                         // gcc specific
    caller_address = (void*)uc->uc_mcontext.eip;  // EIP: x86 specific
#    elif defined(__x86_64__)                     // gcc specific
    caller_address = (void*)uc->uc_mcontext.rip;  // RIP: x86_64 specific
#    elif defined(__arm__)
    caller_address = (void*)uc->uc_mcontext.arm_pc;
#    else
    caller_address = nullptr;
    (void)uc;
#    endif

    std::cout << "signal " << sig_num << " (" << strsignal(sig_num) << ")"
              << ", address is " << info->si_addr << " from " << (void*)caller_address << std::endl;


    printCurrentStack();

    if (customCrashHandler) customCrashHandler();


    // make sure the program exits here, because otherwise the programm will continue after the segfault
    SAIGA_ASSERT(0);
    exit(EXIT_FAILURE);
}


void catchSegFaults()
{
    struct sigaction sigact = {0};

    sigact.sa_sigaction = crit_err_hdlr;
    sigact.sa_flags     = SA_RESTART | SA_SIGINFO;

    if (sigaction(SIGSEGV, &sigact, (struct sigaction*)NULL) != 0)
    {
        std::cerr << "error setting signal handlern" << std::endl;
        SAIGA_ASSERT(0);
    }
}

#endif

#if defined(_WIN32)
// The code requires you to link against the DbgHelp.lib library

void printCurrentStack()
{
    std::string outWalk;
    // Set up the symbol options so that we can gather information from the current
    // executable's PDB files, as well as the Microsoft symbol servers.  We also want
    // to undecorate the symbol names we're returned.  If you want, you can add other
    // symbol servers or paths via a semi-colon separated list in SymInitialized.
    ::SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_INCLUDE_32BIT_MODULES | SYMOPT_UNDNAME);
    if (!::SymInitialize(::GetCurrentProcess(), "http://msdl.microsoft.com/download/symbols", TRUE)) return;

    // Capture up to 25 stack frames from the current call stack.  We're going to
    // skip the first stack frame returned because that's the GetStackWalk function
    // itself, which we don't care about.
    const int numAddrs    = 50;
    PVOID addrs[numAddrs] = {0};
    USHORT frames         = CaptureStackBackTrace(0, numAddrs - 1, addrs, NULL);

    for (USHORT i = 0; i < frames; i++)
    {
        // Allocate a buffer large enough to hold the symbol information on the stack and get
        // a pointer to the buffer.  We also have to set the size of the symbol structure itself
        // and the number of bytes reserved for the name.
        ULONG64 buffer[(sizeof(SYMBOL_INFO) + 1024 + sizeof(ULONG64) - 1) / sizeof(ULONG64)] = {0};
        SYMBOL_INFO* info                                                                    = (SYMBOL_INFO*)buffer;
        info->SizeOfStruct                                                                   = sizeof(SYMBOL_INFO);
        info->MaxNameLen                                                                     = 1024;

        // Attempt to get information about the symbol and add it to our output parameter.
        DWORD64 displacement = 0;
        if (::SymFromAddr(::GetCurrentProcess(), (DWORD64)addrs[i], &displacement, info))
        {
            // outWalk.append(info->Name, info->NameLen);
            // outWalk.append("\n");
            std::cout << "[bt]: (" << i << ") " << info->Name << std::endl;
        }
    }

    ::SymCleanup(::GetCurrentProcess());
}

void SignalHandler(int signal)
{
    printCurrentStack();

    if (customCrashHandler) customCrashHandler();


    // make sure the program exits here, because otherwise the programm will continue after the segfault
    SAIGA_ASSERT(0);
    exit(EXIT_FAILURE);
}

void catchSegFaults()
{
    typedef void (*SignalHandlerPointer)(int);

    SignalHandlerPointer previousHandler;
    previousHandler = signal(SIGSEGV, SignalHandler);
}
#endif


#if defined(__APPLE__)
void catchSegFaults() {}
#endif

}  // namespace Saiga
