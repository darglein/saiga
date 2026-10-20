/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/util/BinaryFile.h"
#include "saiga/core/util/zlib.h"

#include "gtest/gtest.h"

#include <cstring>

namespace Saiga
{
TEST(zlib, SimpleCompressUncompress)
{
    for (int i = 0; i < 10; ++i)
    {
        std::vector<int> data;
        for (int i = 0; i < 10000; ++i)
        {
            data.push_back(rand() % 10);
        }

        auto compressed   = compress(data.data(), data.size() * sizeof(int));
        auto decompressed = uncompress(compressed.data());
        std::cout << "Compress Uncompress (bytes): " << data.size() * sizeof(int) << " -> " << compressed.size()
                  << " -> " << decompressed.size() << std::endl;

        std::vector<int> data2(data.size(), -1);
        memcpy(data2.data(), decompressed.data(), decompressed.size());

        EXPECT_EQ(data, data2);
    }
}

TEST(zlib, BinaryVector)
{
    std::vector<int> data;
    for (int i = 0; i < 10000; ++i)
    {
        data.push_back(rand() % 10);
    }


    BinaryOutputVector ov;
    ov << data;
    std::vector<char> binary_data = ov.data;
    EXPECT_EQ(binary_data.size(), data.size() * sizeof(int) + sizeof(size_t));

    std::vector<int> data2;
    BinaryInputVector iv(binary_data.data(), binary_data.size());
    iv >> data2;


    EXPECT_EQ(data, data2);
}

TEST(zlib, BinaryVectorCompress)
{
    std::vector<int> data;
    for (int i = 0; i < 10000; ++i)
    {
        data.push_back(rand() % 10);
    }

    BinaryOutputVector ov;
    ov << data;
    std::vector<char> binary_data = ov.data;


    auto compressed   = compress(binary_data.data(), binary_data.size());
    auto decompressed = uncompress(compressed.data());
    std::cout << "Compress Uncompress (bytes): " << binary_data.size() << " -> " << compressed.size() << " -> "
              << decompressed.size() << std::endl;

    std::vector<int> data2;
    BinaryInputVector iv(decompressed.data(), decompressed.size());
    iv >> data2;
    EXPECT_EQ(data, data2);
}

TEST(zlib, DeclaredSizeZero)
{
    // With a declared size of 0, the data is decoded into a one byte scratch buffer to detect a stream that is not
    // empty. That check read stream.total_out, which the loop resets, instead of the accumulated output size.
    std::vector<unsigned char> data(100, 7);
    auto compressed = compress(data.data(), data.size());
    size_t header[3];
    memcpy(header, compressed.data(), sizeof(header));
    header[2] = 0;
    memcpy(compressed.data(), header, sizeof(header));
    EXPECT_FALSE(uncompress(compressed.data(), compressed.size(), 1000).has_value());

    auto empty  = compress(data.data(), 0);
    auto result = uncompress(empty.data(), empty.size(), 1000);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

}  // namespace Saiga
