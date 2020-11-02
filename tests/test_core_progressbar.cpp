/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/ProgressBar.h"

#include "gtest/gtest.h"

using namespace Saiga;

TEST(ProgressBar, Basic)
{
    int n = 1000;
    ProgressBar bar(std::cout, "", n);

    for (int i = 0; i < n; ++i)
    {
        bar.addProgress(1);
        bar.SetPostfix("bla  " + std::to_string(i * 1231));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
