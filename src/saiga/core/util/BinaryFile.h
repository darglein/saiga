/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>
namespace Saiga
{
/**
 * Usage Reading:
 *
 * BinaryFile bf(file, std::ios_base::in);
 * int id;
 * bf >> id;
 *
 * Usage Writing:
 *
 *  BinaryFile bf(file, std::ios_base::out);
 *  bf << id;
 */
struct BinaryFile
{
    BinaryFile(const std::string& file, std::ios_base::openmode __mode = std::ios_base::in)
        : strm(file, std::ios::binary | __mode)
    {
    }

    template <typename T>
    void write(const T& v)
    {
        strm.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }

    template <typename T>
    void write(const std::vector<T>& vec)
    {
        write((size_t)vec.size());
        for (auto& v : vec) write(v);
    }

    template <typename T>
    void read(std::vector<T>& vec)
    {
        size_t s;
        read(s);
        vec.resize(s);
        for (auto& v : vec) read(v);
    }

    template <typename T>
    void read(T& v)
    {
        strm.read(reinterpret_cast<char*>(&v), sizeof(T));
    }

    template <typename T>
    BinaryFile& operator<<(const T& v)
    {
        write(v);
        return *this;
    }

    template <typename T>
    BinaryFile& operator>>(T& v)
    {
        read(v);
        return *this;
    }

    std::fstream strm;
};


struct BinaryOutputVector
{
    BinaryOutputVector(size_t reserve_bytes = 1000) { data.reserve(reserve_bytes); }

    void write(const char* d, size_t size)
    {
        size_t old_size = data.size();
        data.resize(data.size() + size);
        std::memcpy(&data[old_size], d, size);
    }


    template <typename T>
    void write(const T& v)
    {
        write(reinterpret_cast<const char*>(&v), sizeof(T));
    }

    template <typename T>
    void write(const std::vector<T>& vec)
    {
        write((size_t)vec.size());
        for (auto& v : vec) write(v);
    }


    template <typename T>
    BinaryOutputVector& operator<<(const T& v)
    {
        write(v);
        return *this;
    }
    std::vector<char> data;
};

struct BinaryInputVector
{
    BinaryInputVector(const void* d, size_t size) : data((const char*)d), size(size) {}

    void read(char* dst, size_t size)
    {
        SAIGA_ASSERT(current + size <= this->size);
        auto src = data + current;
        memcpy(dst, src, size);
        current += size;
    }



    template <typename T>
    void read(std::vector<T>& vec)
    {
        size_t s;
        read(s);
        vec.resize(s);
        for (auto& v : vec) read(v);
    }

    template <typename T>
    void read(T& v)
    {
        read(reinterpret_cast<char*>(&v), sizeof(T));
    }


    template <typename T>
    BinaryInputVector& operator>>(T& v)
    {
        read(v);
        return *this;
    }
    const char* data;
    size_t size;
    size_t current = 0;
};
}  // namespace Saiga
