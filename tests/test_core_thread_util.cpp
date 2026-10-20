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
#include "saiga/core/util/Thread/threadName.h"
#include "saiga/core/util/Thread/threadPool.h"
#include "saiga/core/util/pipeline.h"
#include "saiga/core/util/table.h"

#include "gtest/gtest.h"

#include <future>
#include <sstream>
#include <thread>

#if defined(__linux__)
#    include <pthread.h>
#endif

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

TEST(ScopedThread, QuitAfterMove)
{
    // The thread received a pointer to the terminate flag of the object that started it. After a move it pointed to
    // the flag of the moved-from (possibly destroyed) object, so Quit() of the new owner was never seen by the thread.
    std::atomic<bool> finished = false;
    ScopedThread st;
    st = ScopedThread(
        [&finished](std::atomic<bool>* terminate)
        {
            while (!*terminate) std::this_thread::sleep_for(1ms);
            finished = true;
        });
    ScopedThread st2(std::move(st));
    EXPECT_FALSE(st.joinable());
    std::this_thread::sleep_for(20ms);
    EXPECT_FALSE(finished);

    st2.Quit();
    EXPECT_TRUE(finished);
    EXPECT_FALSE(st2.joinable());
}

#if defined(__linux__)
TEST(ThreadName, LongNameOfOtherThread)
{
    // Linux limits thread names to 15 characters. pthread_setname_np fails for longer names, so the name of another
    // thread was not set at all (setThreadName() of the calling thread uses prctl, which truncates).
    std::atomic<bool> stop = false;
    std::thread t(
        [&]()
        {
            while (!stop) std::this_thread::sleep_for(1ms);
        });
    setThreadName(t, "a_very_long_thread_name");
    char name[16] = {};
    EXPECT_EQ(pthread_getname_np(t.native_handle(), name, sizeof(name)), 0);
    EXPECT_STREQ(name, "a_very_long_thr");

    setThreadName(t, "short");
    EXPECT_EQ(pthread_getname_np(t.native_handle(), name, sizeof(name)), 0);
    EXPECT_STREQ(name, "short");
    stop = true;
    t.join();
}
#endif

TEST(PipelineStage, StopWithFullQueue)
{
    // Without override, the worker blocks in add() once the queue is full. stop() joined it without making room in
    // the queue and waited forever.
    auto stage                = new PipelineStage<int, 1, false>("test");
    std::atomic<int> produced = 0;
    stage->run([&produced]() { return ++produced; });
    // The first element fills the queue, the worker blocks in add() with the second one.
    while (produced < 2) std::this_thread::sleep_for(1ms);
    std::this_thread::sleep_for(10ms);

    auto stopped = std::async(std::launch::async, [stage]() { stage->stop(); });
    if (stopped.wait_for(5s) != std::future_status::ready)
    {
        // Leak the stage and the future, their destructors would block forever.
        new auto(std::move(stopped));
        FAIL() << "PipelineStage::stop() did not return";
    }
    delete stage;

    PipelineStage<int, 2, true> overriding;
    std::atomic<int> counter = 0;
    overriding.run([&counter]() { return ++counter; });
    int v = 0;
    overriding.get(v);
    EXPECT_GT(v, 0);
    overriding.stop();
}
