/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/Thread/omp.h"
using namespace Saiga;

int sizeMB               = 1000 * 1;
size_t blockSize         = sizeMB * 1000UL * 1000UL;
double blockSizeGB       = (blockSize / (1000.0 * 1000.0 * 1000.0));
std::atomic_int debugVar = 0;
void memcpyTest(int threads)
{
    double readWriteGB = blockSizeGB * 2 * threads;  // read + write
    std::vector<char> src(blockSize * threads);
    std::vector<char> dst(blockSize * threads);



    OMP::setNumThreads(threads);

    auto singleThread = measureObject(5, [&]() {
#pragma omp parallel for
        for (int i = 0; i < threads; ++i)
        {
            size_t offset = blockSize * i;
            memcpy(dst.data() + offset, src.data() + offset, blockSize);
        }
    });
    auto t    = singleThread.median / 1000.0;
    double bw = readWriteGB / t;
    std::cout << "Threads " << threads << " Bandwidth: " << bw << " GB/s" << std::endl;
}


void latencyTest()
{
    std::vector<char> src(blockSize * 8);
    int N = 10000000;

    int sum    = 0;
    auto stats = measureObject(5, [&]() {
        size_t offset = 0;
        for (int i = 0; i < N; ++i)
        {
            int* ptr = (int*)(src.data() + offset);
            sum += *ptr;
            // just go in 10kb steps through the memory
            offset += 1000 * 10;
            if (offset >= blockSize * 8) offset = 0;
        }
    });
    debugVar += sum;
    auto t        = stats.median / 1000.0;
    double teleNS = t / N * (1000.0 * 1000.0 * 1000.0);
    std::cout << "Latency: " << teleNS << "ns" << std::endl;
}

int main(int, char**)
{
    catchSegFaults();

    latencyTest();
    std::cout << "Starting RAM Bandwidth test with " << blockSizeGB << " GB blocks." << std::endl;
    memcpyTest(1);
    memcpyTest(2);
    memcpyTest(3);
    memcpyTest(4);
    memcpyTest(8);

    std::cout << "Done." << std::endl;

    return 0;
}
