/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "logging.h"



// A default no-op function if no custom one is installed
void default_glog_fail_func()
{
    abort();
}


void (*g_custom_glog_fail_func)() = &default_glog_fail_func;

namespace google
{
// --- Implementation of InstallFailureFunction ---
void InstallFailureFunction(void (*custom_glog_fail_func)())
{
    g_custom_glog_fail_func = custom_glog_fail_func;
}
}  // namespace google


LogStreamVoidifier LogMessage::stream()
{
    if (result)
    {
        return LogStreamVoidifier();
    }
    else
    {
        return LogStreamVoidifier(std::cout);
    }
}
LogStreamVoidifier::~LogStreamVoidifier()
{
    if (m_stream)
    {
        (*m_stream) << std::flush;
        g_custom_glog_fail_func();
    }
}
