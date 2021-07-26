/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/ProgressBar.h"

#include "gtest/gtest.h"

using namespace Saiga;

TEST(ProgressBar, Basic)
{
    int n = 100;
    ProgressBar bar(std::cout, "", n);

    for (int i = 0; i < n; ++i)
    {
        bar.addProgress(1);
        bar.SetPostfix("bla  " + std::to_string(i * 1231));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

TEST(ProgressBar, SmallTiming)
{
    for (int i = 0; i < 10; ++i)
    {
        SAIGA_BLOCK_TIMER();
        int n = 10;
        ProgressBar bar(std::cout, "", n);
        for (int i = 0; i < n; ++i)
        {
            bar.addProgress(1);
            bar.SetPostfix("bla  " + std::to_string(i * 1231));
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }
}
