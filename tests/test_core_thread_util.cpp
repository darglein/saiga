/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/time/time.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/MemoryUsage.h"
#include "saiga/core/util/Thread/SynchronizedBuffer.h"
#include "saiga/core/util/Thread/SynchronizedSlot.h"
#include "saiga/core/util/Thread/threadPool.h"
#include "saiga/core/util/table.h"

#include "gtest/gtest.h"

#include <sstream>
#include <thread>

using namespace Saiga;
using namespace std::chrono_literals;

TEST(SynchronizedBuffer, ProducerIsWokenWithWaitUntilEmpty)
{
    // A producer blocked in add() and a thread in waitUntilEmpty() both wait on not_full. get() used notify_one(),
    // which could wake only the waitUntilEmpty() thread and leave the producer blocked on an empty buffer.
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        SynchronizedBuffer<int> buffer(1);
        buffer.add(1);

        std::thread waiter([&]() { buffer.waitUntilEmpty(); });
        std::thread producer([&]() { buffer.add(2); });
        std::this_thread::sleep_for(5ms);

        EXPECT_EQ(buffer.get(), 1);
        EXPECT_EQ(buffer.getTimeout(2s), 2) << iteration;
        waiter.join();
        producer.join();
    }
}

TEST(SynchronizedBuffer, ClearWakesProducer)
{
    SynchronizedBuffer<int> buffer(1);
    buffer.add(1);
    std::thread producer([&]() { buffer.add(2); });
    std::this_thread::sleep_for(5ms);
    buffer.clear();
    EXPECT_EQ(buffer.getTimeout(2s), 2);
    producer.join();
}

TEST(ThreadPool, GlobalThreadPoolHasWorkers)
{
    createGlobalThreadPool(-1);
    auto main_id = std::this_thread::get_id();
    auto id      = globalThreadPool->enqueue([]() { return std::this_thread::get_id(); }).get();
    // With 0 workers the task runs inline on the calling thread.
    EXPECT_NE(id, main_id);
    globalThreadPool.reset();
}

TEST(MemoryUsage, Plausible)
{
    auto info = GetMemoryInfo();
    EXPECT_TRUE(info.valid);
    EXPECT_GT(info.max_memory_available, 0);
    EXPECT_GT(info.total_memory_used, 0);
    EXPECT_LE(info.total_memory_used, info.max_memory_available);
    EXPECT_LE(info.current_memory_used, info.max_memory_used);
}

TEST(Time, ScopedTimerMove)
{
    double t = -1;
    {
        ScopedTimer<double> a(t);
        {
            ScopedTimer<double> b(std::move(a));
            std::this_thread::sleep_for(2ms);
        }
        EXPECT_GE(t, 1);
        t = -1;
    }
    // The moved-from timer used to write the time as well.
    EXPECT_EQ(t, -1);
}

TEST(Time, CurrentTimeString)
{
    auto str = CurrentTimeString("%Y");
    EXPECT_EQ(str.size(), 4);
}

TEST(Table, WithoutColumns)
{
    std::stringstream strm;
    Table table;
    table.setStream(strm);
    table << "a" << 1;
    EXPECT_NE(strm.str().find('a'), std::string::npos);
}

TEST(Time, CurrentTimeStringLong)
{
    // The result was read from an uninitialized buffer if it did not fit into 200 characters.
    std::string format(300, 'x');
    EXPECT_EQ(CurrentTimeString(format + "%Y").size(), 304);
    EXPECT_EQ(CurrentTimeString(""), "");
}

TEST(SynchronizedSlot, SetDoesNotMoveFromLvalue)
{
    // set() and setOverride() used std::move on their forwarding reference and emptied the caller's lvalue.
    SynchronizedSlot<std::string> slot;
    std::string value = "a long string that does not fit into the small string buffer";
    slot.set(value);
    EXPECT_EQ(value, "a long string that does not fit into the small string buffer");
    EXPECT_EQ(slot.get(), value);

    slot.setOverride(value);
    slot.setOverride(value);
    EXPECT_EQ(value, "a long string that does not fit into the small string buffer");
    EXPECT_EQ(slot.peek(), value);
    EXPECT_EQ(slot.get(), value);
    EXPECT_TRUE(slot.empty());

    std::thread producer([&]() { slot.set(std::string("x")); });
    EXPECT_EQ(slot.get(), "x");
    producer.join();
}
