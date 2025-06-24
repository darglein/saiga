/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "logging.h"


NullStream glog_nstrm;



// A default no-op function if no custom one is installed
void default_glog_fail_func() {
   abort();
}


void (*g_custom_glog_fail_func)() = &default_glog_fail_func;

namespace google
{
// --- Implementation of InstallFailureFunction ---
void InstallFailureFunction(void (*custom_glog_fail_func)()) {
    g_custom_glog_fail_func = custom_glog_fail_func;
}
}

// std::ostream& LogMessage::stream()
// {
//     auto& strm = result ? glog_nstrm : std::cout;
//     return strm;
// }


LogMessage::~LogMessage()
{
    if (!result)
    {
        // If check failed, flush and abort
        std::cout << std::endl;
        g_custom_glog_fail_func();
    }
}
// Implementation of LogMessage::stream()
LogStreamVoidifier LogMessage::stream()
{
    if (result) {
        // If check passed, return an inactive LogStreamVoidifier.
        // Its 'operator bool' will be false, and 'operator<<' will do nothing.
        return LogStreamVoidifier();
    } else {
        // If check failed, return an active LogStreamVoidifier, pointing to std::cout.
        // Its 'operator bool' will be true, and 'operator<<' will write to std::cout.
        return LogStreamVoidifier(std::cout);
    }
}
