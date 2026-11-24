/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Linked into every test executable (tests/CMakeLists.txt): the process settings of the tests, which keep the test
// executables fast (AGENTS.md, "Test speed").
//
// No core dumps: every EXPECT_DEATH child ends with SIGABRT, and a core dump of a large test process (e.g. to
// systemd-coredump or apport, which a pipe in /proc/sys/kernel/core_pattern starts for every dump) took longer than the
// death test itself (~0.1 s per case). A soft RLIMIT_CORE of 1 byte disables piped dumps too (core(5): a limit of 0 is
// not enforced for pipes, 1 is). SAIGA_TEST_CORE_DUMPS=1 keeps the limit of the shell, e.g. to debug a crash.
//
// SAIGA_TEST_VULKAN (the executables that link saiga_vulkan): headless Vulkan devices are reused between the tests
// (Vulkan::setHeadlessDeviceReuse; creating an instance and a device per test took longer than most tests). The pool
// is released after the last test, so leaked objects are still reported by the validation layer (which fails the
// run). SAIGA_TEST_NEW_DEVICES=1 creates a new device for every renderer, as an application does.

#include "gtest/gtest.h"

#include <cstdlib>

#if defined(__linux__)
#    include <sys/resource.h>
#endif

#if defined(SAIGA_TEST_VULKAN)
#    include "saiga/vulkan/VulkanDevice.h"
#endif

namespace
{
bool envSet(const char* name)
{
    const char* v = std::getenv(name);
    return v && *v && *v != '0';
}

#if defined(SAIGA_TEST_VULKAN)
// Releases the pooled devices after the last test (inside RUN_ALL_TESTS, so a validation error fails the run) and
// turns the reuse off: a renderer that another environment destroys later (RenderSceneSuiteTest's scene, its TearDown
// may run after this one) destroys its device directly instead of returning it to the pool, which would otherwise be
// destroyed only in the static destruction, after the validation layer (the process aborted at exit)
class ReleasePooledDevices : public ::testing::Environment
{
   public:
    void TearDown() override { Saiga::Vulkan::setHeadlessDeviceReuse(false); }
};
#endif

struct TestProcessSetup
{
    TestProcessSetup()
    {
#if defined(__linux__)
        if (!envSet("SAIGA_TEST_CORE_DUMPS"))
        {
            rlimit limit;
            if (getrlimit(RLIMIT_CORE, &limit) == 0 && (limit.rlim_max == RLIM_INFINITY || limit.rlim_max >= 1))
            {
                limit.rlim_cur = 1;
                setrlimit(RLIMIT_CORE, &limit);
            }
        }
#endif
#if defined(SAIGA_TEST_VULKAN)
        if (!envSet("SAIGA_TEST_NEW_DEVICES"))
        {
            Saiga::Vulkan::setHeadlessDeviceReuse(true);
            ::testing::AddGlobalTestEnvironment(new ReleasePooledDevices());
        }
#endif
    }
};
const TestProcessSetup testProcessSetup;
}  // namespace
