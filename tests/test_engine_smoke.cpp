/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/core/EngineVersion.h"
#include "saiga/saiga_modules.h"

#include "gtest/gtest.h"

#include <cstring>

using namespace Saiga;

// The engine module builds, links (exported symbol of the shared library) and is announced by saiga_modules.h
TEST(EngineSmoke, VersionAndModuleDefine)
{
    const char* version = Engine::engineVersion();
    ASSERT_NE(version, nullptr);
    EXPECT_GT(std::strlen(version), 0u);
#ifndef SAIGA_USE_ENGINE
    FAIL() << "SAIGA_USE_ENGINE is not defined although the engine module is built";
#endif
}
