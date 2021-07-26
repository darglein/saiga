/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/math/all.h"
#include "saiga/core/util/Align.h"

#include "gtest/gtest.h"

using namespace Saiga;
/**
 * This test checks if aligned_malloc and the alignedVector actually work.
 * -> Make sure the data is aligned to the given size
 */

struct SAIGA_ALIGN(128) Align128
{
    int a;
};


struct SAIGA_ALIGN(2048) AlignLarge
{
    int a;
};



TEST(Align, aligned_malloc)
{
    std::cout << alignof(PerspectiveCamera) << std::endl;
    std::cout << sizeof(PerspectiveCamera) << std::endl;
    std::cout << alignof(mat4) << std::endl;
    std::cout << sizeof(mat4) << std::endl;

    size_t alocSize = 12345;
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(aligned_malloc<4>(alocSize)) % 4);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(aligned_malloc<8>(alocSize)) % 8);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(aligned_malloc<16>(alocSize)) % 16);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(aligned_malloc<32>(alocSize)) % 32);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(aligned_malloc<64>(alocSize)) % 64);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(aligned_malloc<128>(alocSize)) % 128);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(aligned_malloc<256>(alocSize)) % 256);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(aligned_malloc<1024>(alocSize)) % 1024);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(aligned_malloc<1024 * 1024>(alocSize)) % (1024 * 1024));
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(aligned_malloc<1024 * 4096>(alocSize)) % (1024 * 4096));
}


TEST(Align, AlignedVector)
{
    size_t alocSize = 12345;
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(AlignedVector<uint8_t>(alocSize).data()) % 1);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(AlignedVector<uint16_t>(alocSize).data()) % 2);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(AlignedVector<uint32_t>(alocSize).data()) % 4);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(AlignedVector<int, 16>(alocSize).data()) % 16);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(AlignedVector<int, 128>(alocSize).data()) % 128);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(AlignedVector<int, 1024>(alocSize).data()) % 1024);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(AlignedVector<int, 1024 * 4096>(alocSize).data()) % (1024 * 4096));
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(AlignedVector<AlignLarge>(alocSize).data()) % 2048);
}

TEST(Align, AlignedVector_Resize)
{
    AlignedVector<int, 1024> v(123);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(v.data()) % 1024);
    v.resize(1234);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(v.data()) % 1024);
    v.push_back(0);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(v.data()) % 1024);
}



TEST(Align, AligendSmartPtr)
{
    auto ptr = make_aligned_shared<int, 1024>(123);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(ptr.get()) % 1024);

    auto ptr2 = make_aligned_shared<AlignLarge>();
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(ptr2.get()) % 2048);

    auto ptr3 = make_aligned_unique<AlignLarge>();
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(ptr3.get()) % 2048);
}


TEST(Align, AlignStruct)
{
    Align128 test;
    AlignLarge test2;
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(&test) % 128);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(&test2) % 2048);


    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(new Align128()) % 128);
    EXPECT_EQ(0, reinterpret_cast<uintptr_t>(new AlignLarge()) % (2048));
}
