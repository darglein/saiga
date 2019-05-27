/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include <fstream>
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
}  // namespace Saiga
