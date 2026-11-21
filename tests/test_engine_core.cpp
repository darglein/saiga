/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine's core utilities (docs/engine_design.md §8.5, §7.10, §17.1; stage S06): JobPool, Rng, Hash, log ring.

#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/JobPool.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/core/Rng.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace Saiga;
using namespace Saiga::Engine;

// ---------------------------------------------------------------------------------------------------------------------
// JobPool

TEST(JobPool, EveryIndexExactlyOnce)
{
    for (int workers : {0, 1, 3, 8})
    {
        JobPool pool(workers);
        EXPECT_EQ(pool.workerThreads(), workers);
        for (int64_t chunk : {1, 7, 1000})
        {
            for (auto range : {std::pair<int64_t, int64_t>{0, 500}, {-37, 263}, {5, 6}, {10, 10}, {10, 3}})
            {
                int64_t begin = range.first, end = range.second;
                int64_t n = std::max<int64_t>(0, end - begin);
                std::vector<std::atomic<int>> hits(size_t(n + 1));
                std::atomic<int> calls{0};
                pool.parallelFor(begin, end, chunk,
                                 [&](int64_t b, int64_t e)
                                 {
                                     calls++;
                                     // Inside the range, checked before any arithmetic on the bounds (a chunk past
                                     // end would hit the padding slot of hits)
                                     ASSERT_GE(b, begin);
                                     ASSERT_LT(b, e);
                                     ASSERT_LE(e, end);
                                     // Fixed chunk boundaries
                                     EXPECT_EQ((b - begin) % chunk, 0);
                                     EXPECT_TRUE(e == end || e - b == chunk);
                                     for (int64_t i = b; i < e; ++i) hits[size_t(i - begin)]++;
                                 });
                for (int64_t i = 0; i < n; ++i)
                    EXPECT_EQ(hits[size_t(i)].load(), 1) << workers << " " << chunk << " " << i;
                EXPECT_EQ(calls.load(), n == 0 ? 0 : int((n + chunk - 1) / chunk));
            }
        }
    }
}

TEST(JobPool, SerialInOrderWithoutWorkers)
{
    JobPool pool(0);
    std::vector<int64_t> order;
    const auto caller = std::this_thread::get_id();
    pool.parallelFor(0, 10, 3,
                     [&](int64_t b, int64_t)
                     {
                         EXPECT_EQ(std::this_thread::get_id(), caller);
                         order.push_back(b);
                     });
    EXPECT_EQ(order, (std::vector<int64_t>{0, 3, 6, 9}));
}

// N workers, N + 1 chunks: every chunk that does not run on the calling thread waits until a chunk ran there. Without
// the caller taking chunks, the workers would all wait (and time out).
TEST(JobPool, CallerParticipates)
{
    for (int workers : {1, 3})
    {
        JobPool pool(workers);
        const auto caller = std::this_thread::get_id();
        std::atomic<bool> callerRan{false};
        std::atomic<bool> timedOut{false};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        pool.parallelFor(0, workers + 1, 1,
                         [&](int64_t, int64_t)
                         {
                             if (std::this_thread::get_id() == caller)
                             {
                                 callerRan = true;
                                 return;
                             }
                             while (!callerRan)
                             {
                                 if (std::chrono::steady_clock::now() > deadline)
                                 {
                                     timedOut = true;
                                     return;
                                 }
                                 std::this_thread::yield();
                             }
                         });
        EXPECT_TRUE(callerRan.load());
        EXPECT_FALSE(timedOut.load());
    }
}

TEST(JobPool, UsesTheWorkers)
{
    // Every chunk waits until two different threads have entered a chunk: needs at least one worker besides the caller
    JobPool pool(2);
    std::mutex m;
    std::set<std::thread::id> threads;
    std::atomic<bool> timedOut{false};
    // One deadline for all chunks: a failure takes 10 s, not 10 s per chunk
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    pool.parallelFor(0, 8, 1,
                     [&](int64_t, int64_t)
                     {
                         {
                             std::lock_guard<std::mutex> lock(m);
                             threads.insert(std::this_thread::get_id());
                         }
                         while (true)
                         {
                             {
                                 std::lock_guard<std::mutex> lock(m);
                                 if (threads.size() >= 2) return;
                             }
                             if (std::chrono::steady_clock::now() > deadline)
                             {
                                 timedOut = true;
                                 return;
                             }
                             std::this_thread::yield();
                         }
                     });
    EXPECT_FALSE(timedOut.load());
    EXPECT_GE(threads.size(), 2u);
}

TEST(JobPool, NestedRunsSerially)
{
    JobPool pool(3);
    std::atomic<int64_t> sum{0};
    pool.parallelFor(0, 8, 1,
                     [&](int64_t, int64_t)
                     {
                         const auto self = std::this_thread::get_id();
                         pool.parallelFor(0, 100, 10,
                                          [&](int64_t b, int64_t e)
                                          {
                                              EXPECT_EQ(std::this_thread::get_id(), self);
                                              for (int64_t i = b; i < e; ++i) sum += i;
                                          });
                     });
    EXPECT_EQ(sum.load(), 8 * 4950);
}

TEST(JobPool, ExceptionReachesTheCaller)
{
    for (int workers : {0, 3})
    {
        JobPool pool(workers);
        std::atomic<int> calls{0};
        EXPECT_THROW(pool.parallelFor(0, 1000, 1,
                                      [&](int64_t b, int64_t)
                                      {
                                          calls++;
                                          if (b == 10) throw std::runtime_error("chunk 10");
                                      }),
                     std::runtime_error);
        // Serially the chunks after the failure are skipped. With workers skipping is best effort (chunks claimed
        // before the failure is seen still run), so it is not checked there.
        if (workers == 0)
        {
            EXPECT_EQ(calls.load(), 11);
        }
        // The pool still works
        std::atomic<int64_t> n{0};
        pool.parallelFor(0, 100, 7, [&](int64_t b, int64_t e) { n += e - b; });
        EXPECT_EQ(n.load(), 100);
    }
}

TEST(JobPool, ManySmallJobsAndChangingWorkerCount)
{
    JobPool pool(4);
    int64_t total = 0;
    for (int round = 0; round < 2000; ++round)
    {
        std::atomic<int64_t> n{0};
        pool.parallelFor(0, 5, 1, [&](int64_t b, int64_t e) { n += e - b; });
        total += n;
        if (round == 700) pool.setWorkerThreads(1);
        if (round == 1400) pool.setWorkerThreads(6);
    }
    EXPECT_EQ(total, 2000 * 5);
    EXPECT_EQ(pool.workerThreads(), 6);
}

TEST(JobPool, CallsFromTwoThreads)
{
    JobPool pool(3);
    std::atomic<int64_t> a{0}, b{0};
    // Chunks of the two callers' jobs never run at the same time (serialized calls)
    std::atomic<int> activeA{0}, activeB{0};
    std::atomic<bool> overlap{false};
    std::thread other(
        [&]()
        {
            for (int i = 0; i < 300; ++i)
                pool.parallelFor(0, 50, 3,
                                 [&](int64_t s, int64_t e)
                                 {
                                     activeA++;
                                     if (activeB > 0) overlap = true;
                                     a += e - s;
                                     activeA--;
                                 });
        });
    for (int i = 0; i < 300; ++i)
        pool.parallelFor(0, 40, 3,
                         [&](int64_t s, int64_t e)
                         {
                             activeB++;
                             if (activeA > 0) overlap = true;
                             b += e - s;
                             activeB--;
                         });
    other.join();
    EXPECT_EQ(a.load(), 300 * 50);
    EXPECT_EQ(b.load(), 300 * 40);
    EXPECT_FALSE(overlap.load());
}

// Regression: the nested check was one flag for all pools, so a chunk of pool B calling pool A ran A's chunks inline
// without A's call lock, concurrently with another caller of A
TEST(JobPool, ChunkOfAnotherPoolCallsThisPoolSerialized)
{
    JobPool a(2), b(2);
    std::atomic<int> activeOuter{0}, activeInner{0};
    std::atomic<bool> overlap{false};
    std::atomic<int64_t> n{0};
    // Thread 1 uses pool A directly; thread 2 calls pool A from chunks of pool B
    std::thread direct(
        [&]()
        {
            for (int i = 0; i < 200; ++i)
                a.parallelFor(0, 8, 1,
                              [&](int64_t, int64_t)
                              {
                                  activeOuter++;
                                  if (activeInner > 0) overlap = true;
                                  n++;
                                  activeOuter--;
                              });
        });
    for (int i = 0; i < 50; ++i)
        b.parallelFor(0, 4, 1,
                      [&](int64_t, int64_t)
                      {
                          a.parallelFor(0, 8, 1,
                                        [&](int64_t, int64_t)
                                        {
                                            activeInner++;
                                            if (activeOuter > 0) overlap = true;
                                            n++;
                                            activeInner--;
                                        });
                      });
    direct.join();
    EXPECT_FALSE(overlap.load());
    EXPECT_EQ(n.load(), 200 * 8 + 50 * 4 * 8);
}

TEST(JobPool, TooManyChunksThrows)
{
    // 2^63 chunks do not fit int64: an error in every build, not a silently skipped loop
    JobPool pool(0);
    EXPECT_THROW(pool.parallelFor(std::numeric_limits<int64_t>::min(), 0, 1, [](int64_t, int64_t) {}),
                 std::length_error);
    // The same range with larger chunks works
    int64_t calls = 0;
    pool.parallelFor(std::numeric_limits<int64_t>::min(), 0, int64_t(1) << 62, [&](int64_t, int64_t) { calls++; });
    EXPECT_EQ(calls, 2);
}

TEST(JobPool, HugeRangeWithoutOverflow)
{
    // A range wider than INT64_MAX in few chunks
    JobPool pool(2);
    const int64_t lo = std::numeric_limits<int64_t>::min() + 5, hi = std::numeric_limits<int64_t>::max() - 5;
    const int64_t chunk = std::numeric_limits<int64_t>::max() / 4;
    std::mutex m;
    std::vector<std::pair<int64_t, int64_t>> chunks;
    pool.parallelFor(lo, hi, chunk,
                     [&](int64_t b, int64_t e)
                     {
                         std::lock_guard<std::mutex> lock(m);
                         chunks.push_back({b, e});
                     });
    std::sort(chunks.begin(), chunks.end());
    ASSERT_FALSE(chunks.empty());
    EXPECT_EQ(chunks.front().first, lo);
    EXPECT_EQ(chunks.back().second, hi);
    for (size_t i = 1; i < chunks.size(); ++i) EXPECT_EQ(chunks[i].first, chunks[i - 1].second);
}

// ---------------------------------------------------------------------------------------------------------------------
// Rng

TEST(Rng, Pcg32ReferenceValues)
{
    // pcg32-demo of the PCG reference implementation (pcg32_srandom(42, 54))
    Rng rng(42, 54);
    const uint32_t expected[] = {0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e};
    for (uint32_t e : expected) EXPECT_EQ(rng.nextU32(), e);
}

TEST(Rng, SeedsAndStreams)
{
    Rng a(7), b(7), c(7, 1), d(8);
    bool differsStream = false, differsSeed = false;
    for (int i = 0; i < 100; ++i)
    {
        uint32_t x = a.nextU32();
        EXPECT_EQ(x, b.nextU32());
        differsStream |= x != c.nextU32();
        differsSeed |= x != d.nextU32();
    }
    EXPECT_TRUE(differsStream);
    EXPECT_TRUE(differsSeed);
}

TEST(Rng, Uniform)
{
    Rng rng(1);
    double sum  = 0;
    const int n = 200000;
    for (int i = 0; i < n; ++i)
    {
        double u = rng.uniform();
        ASSERT_GE(u, 0.0);
        ASSERT_LT(u, 1.0);
        sum += u;
    }
    EXPECT_NEAR(sum / n, 0.5, 0.005);
    for (int i = 0; i < 1000; ++i)
    {
        double u = rng.uniform(-3.0, 5.0);
        EXPECT_GE(u, -3.0);
        EXPECT_LT(u, 5.0);
    }
}

TEST(Rng, RangeInclusiveAndUniform)
{
    Rng rng(3);
    // Single value, negative bounds
    for (int i = 0; i < 10; ++i) EXPECT_EQ(rng.range(4, 4), 4);
    std::vector<int> counts(7, 0);
    const int n = 70000;
    for (int i = 0; i < n; ++i)
    {
        int64_t v = rng.range(-3, 3);
        ASSERT_GE(v, -3);
        ASSERT_LE(v, 3);
        counts[size_t(v + 3)]++;
    }
    // Both ends are reached, each value about n / 7
    for (int c : counts) EXPECT_NEAR(c, n / 7, n / 7 * 0.05);

    // Spans above 2^32 and the full int64 range
    const int64_t lo = -(int64_t(1) << 40), hi = int64_t(1) << 40;
    bool negative = false, positive = false;
    for (int i = 0; i < 1000; ++i)
    {
        int64_t v = rng.range(lo, hi);
        ASSERT_GE(v, lo);
        ASSERT_LE(v, hi);
        negative |= v < 0;
        positive |= v > 0;
    }
    EXPECT_TRUE(negative && positive);
    const int64_t mn = std::numeric_limits<int64_t>::min(), mx = std::numeric_limits<int64_t>::max();
    negative = positive = false;
    for (int i = 0; i < 100; ++i)
    {
        int64_t v = rng.range(mn, mx);
        negative |= v < 0;
        positive |= v > 0;
    }
    EXPECT_TRUE(negative && positive);
    // A span of exactly 2^32 (one output, no rejection)
    for (int i = 0; i < 1000; ++i)
    {
        int64_t v = rng.range(0, 0xFFFFFFFFll);
        ASSERT_GE(v, 0);
        ASSERT_LE(v, 0xFFFFFFFFll);
    }
}

TEST(Rng, FixedValuesOfRangeAndUniform)
{
    // Fixed values (equal with GCC and Clang): range() and uniform() must not change their use of the stream, saved
    // worlds and scenarios depend on it
    Rng rng(42, 54);
    for (int64_t e : {63, 48, 72, 51, 74}) EXPECT_EQ(rng.range(0, 99), e);
    Rng wide(42, 54);
    for (int64_t e : {-311469586936ll, 208363971578ll, -773974028750ll})
        EXPECT_EQ(wide.range(-(int64_t(1) << 40), int64_t(1) << 40), e);
    Rng u(42, 54);
    EXPECT_EQ(u.uniform(), 0.6303102186438938);
    // uniform(a, b) uses std::fma: the same values with and without FMA contraction (checked with -march=haswell)
    Rng ab(42, 54);
    for (double e : {5.7546532796584069, 7.2051208401029054, 7.5290504719977251}) EXPECT_EQ(ab.uniform(-3.7, 11.3), e);

    // The rejection paths (about half of the draws are rejected for these spans): a wrong rejection bound changes the
    // values. 64 bits: a span of 2^63 + 21; 32 bits (Lemire): a span of 2^31 + 1.
    Rng r64(7, 3);
    const int64_t lo = std::numeric_limits<int64_t>::min() / 2 - 10, hi = std::numeric_limits<int64_t>::max() / 2 + 10;
    for (int64_t e : {2497150458469127928ll, 3064514463451230757ll, -3083869450825681280ll, -1377254491934152336ll,
                      3186856330796894013ll})
        EXPECT_EQ(r64.range(lo, hi), e);
    Rng r32(7, 3);
    for (int64_t e : {1592787854, 1172360065, 893627349, 25647383, 1544824218})
        EXPECT_EQ(r32.range(0, 0x80000000ll), e);
}

// Regression: a + (b - a) * u rounds up to b for u = 1 - 2^-53 (e.g. a = 1, b = 2), outside of [a, b)
TEST(Rng, UniformIntervalNeverReturnsTheUpperBound)
{
    const double u = 1.0 - 1.0 / 9007199254740992.0;
    EXPECT_EQ(std::fma(2.0 - 1.0, u, 1.0), 2.0);  // the unguarded formula rounds up to b
    // With an interval of one ulp (2 at 1e16) about half of the raw results round up to b; the only value in [a, b) is
    // a
    Rng rng(5);
    for (int i = 0; i < 1000; ++i) EXPECT_EQ(rng.uniform(1e16, 1e16 + 2), 1e16);
    EXPECT_EQ(rng.uniform(2.0, 2.0), 2.0);
}

// Regression: b - a overflows to infinity for [-DBL_MAX, DBL_MAX], and every draw returned the same value
TEST(Rng, UniformWideInterval)
{
    Rng rng(9);
    const double mx = std::numeric_limits<double>::max();
    bool negative = false, positive = false;
    for (int i = 0; i < 100; ++i)
    {
        double v = rng.uniform(-mx, mx);
        ASSERT_TRUE(std::isfinite(v));
        ASSERT_LT(v, mx);
        negative |= v < 0;
        positive |= v > 0;
    }
    EXPECT_TRUE(negative && positive);
}

TEST(Rng, StateSaveAndRestore)
{
    Rng rng(99, 5);
    for (int i = 0; i < 10; ++i) rng.nextU32();
    Rng::State saved = rng.state();
    std::vector<uint32_t> next;
    for (int i = 0; i < 10; ++i) next.push_back(rng.nextU32());

    Rng restored;
    restored.setState(saved);
    EXPECT_EQ(restored.state(), saved);
    for (uint32_t x : next) EXPECT_EQ(restored.nextU32(), x);
}

// ---------------------------------------------------------------------------------------------------------------------
// Hash

TEST(Hash, Fnv1a64ReferenceValues)
{
    EXPECT_EQ(fnv1a64(""), 0xcbf29ce484222325ULL);
    EXPECT_EQ(fnv1a64("a"), 0xaf63dc4c8601ec8cULL);
    EXPECT_EQ(fnv1a64("foobar"), 0x85944171f73967e8ULL);
    const char bytes[] = {'f', 'o', 'o', 'b', 'a', 'r'};
    EXPECT_EQ(fnv1a64Bytes(bytes, sizeof(bytes)), 0x85944171f73967e8ULL);
    // Usable at compile time (IDs from names)
    static_assert(fnv1a64("foobar") == 0x85944171f73967e8ULL, "constexpr FNV-1a");
}

TEST(Hash, HasherIsByteOrderIndependent)
{
    // Integers are hashed as little-endian bytes on every host
    const unsigned char le32[] = {0x78, 0x56, 0x34, 0x12};
    EXPECT_EQ(Hasher().addU32(0x12345678u).value(), fnv1a64Bytes(le32, 4));
    const unsigned char le64[] = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01};
    EXPECT_EQ(Hasher().addU64(0x0102030405060708ull).value(), fnv1a64Bytes(le64, 8));
    EXPECT_EQ(Hasher().addI32(-1).value(), Hasher().addU32(0xFFFFFFFFu).value());
    EXPECT_EQ(Hasher().addI64(-2).value(), Hasher().addU64(0xFFFFFFFFFFFFFFFEull).value());
    // A fixed value for a mixed sequence (equal with GCC and Clang; a change here changes every content hash and
    // world.hash())
    Hasher h;
    h.addU8(1).addU32(2).addI64(-3).addDouble(0.5).addString("chunk");
    EXPECT_EQ(h.value(), 0x3de363562a7defd1ULL);
}

TEST(Hash, DoublesStringsAndCombine)
{
    EXPECT_NE(Hasher().addDouble(0.0).value(), Hasher().addDouble(-0.0).value());
    EXPECT_NE(Hasher().addString("ab").addString("c").value(), Hasher().addString("a").addString("bc").value());
    EXPECT_NE(hashCombine(1, 2), hashCombine(2, 1));
    EXPECT_EQ(hashCombine(1, 2), Hasher().addU64(1).addU64(2).value());
    // A seed hashed with different values gives different derived seeds
    std::set<uint64_t> seeds;
    for (uint64_t i = 0; i < 1000; ++i) seeds.insert(hashCombine(12345, i));
    EXPECT_EQ(seeds.size(), 1000u);
}

// ---------------------------------------------------------------------------------------------------------------------
// Log

TEST(Log, RingSequenceNumbersAndSince)
{
    LogRing ring(4);
    EXPECT_EQ(ring.lastSequence(), 0u);
    EXPECT_TRUE(ring.since(0).empty());
    for (int i = 1; i <= 6; ++i) EXPECT_EQ(ring.add("test", "line " + std::to_string(i)), uint64_t(i));
    // The newest 4 are kept
    auto all = ring.since(0);
    ASSERT_EQ(all.size(), 4u);
    EXPECT_EQ(all.front().sequence, 3u);
    EXPECT_EQ(all.front().message, "line 3");
    EXPECT_EQ(all.back().sequence, 6u);
    auto newer = ring.since(4);
    ASSERT_EQ(newer.size(), 2u);
    EXPECT_EQ(newer[0].sequence, 5u);
    EXPECT_EQ(newer[1].category, "test");
    EXPECT_TRUE(ring.since(6).empty());
    EXPECT_TRUE(ring.since(100).empty());
    // A ring of capacity 0 is rejected in every build
    EXPECT_THROW(LogRing(0), std::invalid_argument);
    // clear keeps the sequence counter
    ring.clear();
    EXPECT_TRUE(ring.since(0).empty());
    EXPECT_EQ(ring.add("test", "after clear"), 7u);
}

// S19c: each line carries the time since the start (the debug UI's console shows it)
TEST(Log, LinesCarryTheTime)
{
    LogRing ring;
    ring.add("test", "a");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    ring.add("test", "b");
    const auto lines = ring.since(0);
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_GE(lines[0].time, 0.0);
    EXPECT_GT(lines[1].time, lines[0].time);
    EXPECT_LE(lines[1].time, LogRing::clockSeconds());
}

TEST(Log, SaigaLogFormatsIntoTheRing)
{
    setLogEcho(false);
    uint64_t before = engineLog().lastSequence();
    SAIGA_LOG("test", "value %d, name %s", 42, "x");
    std::string longText(5000, 'a');
    SAIGA_LOG("test", "%s|%d", longText.c_str(), 7);
    auto lines = engineLog().since(before);
    ASSERT_EQ(lines.size(), 2u);
    EXPECT_EQ(lines[0].category, "test");
    EXPECT_EQ(lines[0].message, "value 42, name x");
    EXPECT_EQ(lines[1].message, longText + "|7");
    EXPECT_EQ(lines[1].sequence, lines[0].sequence + 1);
    // The ring keeps line breaks (only the stdout echo escapes them)
    SAIGA_LOG("test", "first\nsecond");
    auto multiline = engineLog().since(lines[1].sequence);
    setLogEcho(true);
    ASSERT_EQ(multiline.size(), 1u);
    EXPECT_EQ(multiline[0].message, "first\nsecond");
}

TEST(Log, ConcurrentLoggingLosesNothing)
{
    setLogEcho(false);
    LogRing& ring                = engineLog();
    uint64_t start               = ring.lastSequence();
    static constexpr int threads = 4, perThread = 200;  // static: usable in the lambda without a capture (MSVC)
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t)
        ts.emplace_back(
            [t]()
            {
                for (int i = 0; i < perThread; ++i) SAIGA_LOG("thread", "%d %d", t, i);
            });
    for (auto& t : ts) t.join();
    setLogEcho(true);
    auto lines = ring.since(start);
    ASSERT_EQ(lines.size(), size_t(threads * perThread));
    std::set<std::string> messages;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        EXPECT_EQ(lines[i].sequence, start + 1 + i);
        messages.insert(lines[i].message);
    }
    EXPECT_EQ(messages.size(), size_t(threads * perThread));
}
