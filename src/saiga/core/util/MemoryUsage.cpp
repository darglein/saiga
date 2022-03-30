/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#    include <windows.h>
//
#    include <psapi.h>



#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#    include <sys/resource.h>
#    include <unistd.h>

#    if defined(__APPLE__) && defined(__MACH__)
#        include <mach/mach.h>

#    elif (defined(_AIX) || defined(__TOS__AIX__)) || \
        (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#        include <fcntl.h>
#        include <procfs.h>

#    elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#        include <stdio.h>

#    endif

#else

#endif


#include "MemoryUsage.h"


/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
static size_t getPeakRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || \
    (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1) return (size_t)0L; /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L; /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#    if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#    else
    return (size_t)(rusage.ru_maxrss * 1024L);
#    endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L; /* Unsupported. */
#endif
}



/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
static size_t getCurrentRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L; /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE* fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL) return (size_t)0L; /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1)
    {
        fclose(fp);
        return (size_t)0L; /* Can't read? */
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L; /* Unsupported. */
#endif
}

namespace Saiga
{
MemoryInfo GetMemoryInfo()
{
    MemoryInfo result;
    result.current_memory_used = getCurrentRSS();
    result.max_memory_used     = getPeakRSS();
    result.valid               = result.current_memory_used > 0 && result.max_memory_used > 0;
    return result;
}

std::ostream& operator<<(std::ostream& strm, const MemoryInfo& mem_info)
{
    strm << "[Memory Info]\n";
    strm << "Current Usage (MB): " << mem_info.current_memory_used / (1000.0 * 1000.0) << "\n";
    strm << "Max Usage (MB):     " << mem_info.max_memory_used / (1000.0 * 1000.0);
    return strm;
}

}  // namespace Saiga