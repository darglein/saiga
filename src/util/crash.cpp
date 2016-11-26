#include "saiga/util/crash.h"
#include "saiga/util/assert.h"

#include <iostream>

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

std::function<void()> customCrashHandler;

//Source: http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes

/* This structure mirrors the one found in /usr/include/asm/ucontext.h */
typedef struct _sig_ucontext {
    unsigned long     uc_flags;
    struct ucontext   *uc_link;
    stack_t           uc_stack;
    struct sigcontext uc_mcontext;
    sigset_t          uc_sigmask;
} sig_ucontext_t;

void crit_err_hdlr(int sig_num, siginfo_t * info, void * ucontext)
{
    void *             array[50];
    void *             caller_address;
    char **            messages;
    int                size, i;
    sig_ucontext_t *   uc;

    uc = (sig_ucontext_t *)ucontext;

    /* Get the address at the time the signal was raised */
#if defined(__i386__) // gcc specific
    caller_address = (void *) uc->uc_mcontext.eip; // EIP: x86 specific
#elif defined(__x86_64__) // gcc specific
    caller_address = (void *) uc->uc_mcontext.rip; // RIP: x86_64 specific
#else
#error Unsupported architecture. // TODO: Add support for other arch.
#endif

    std:: cout << "signal " << sig_num << " (" << strsignal(sig_num) << ")"
               << ", address is " << info->si_addr << " from " << (void *)caller_address << std::endl;


    size = backtrace(array, 50);

    /* overwrite sigaction with caller's address */
    array[1] = caller_address;

    messages = backtrace_symbols(array, size);

    /* skip first stack frame (points here) */
    for (i = 1; i < size && messages != NULL; ++i){
        std::cout << "[bt]: (" << i << ") " << messages[i] << std::endl;
    }

    free(messages);

    if(customCrashHandler)
        customCrashHandler();


    //make sure the program exits here, because otherwise the programm will continue after the segfault
    SAIGA_ASSERT(0);
    exit(EXIT_FAILURE);
}


void catchSegFaults()
{
    struct sigaction sigact;

    sigact.sa_sigaction = crit_err_hdlr;
    sigact.sa_flags = SA_RESTART | SA_SIGINFO;

    if (sigaction(SIGSEGV, &sigact, (struct sigaction *)NULL) != 0){
        std::cerr << "error setting signal handlern" << std::endl;
        SAIGA_ASSERT(0);
    }
}

void addCustomSegfaultHandler(std::function<void ()> fnc)
{
    customCrashHandler = fnc;
}
