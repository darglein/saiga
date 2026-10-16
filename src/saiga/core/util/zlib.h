/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <cstddef>
#include <optional>
#include <vector>

#ifdef SAIGA_USE_ZLIB

namespace Saiga
{
// Compress and uncompress an array of bytes.
// In the compressed data, we store the size in a header struct.
// Therefore we do not need the size for uncompessing.
// Example usage:
//
//    std::vector<int> data;
//    ...
//    auto compressed   = compress(data.data(), data.size() * sizeof(int));
//    auto decompressed = uncompress(compressed.data());
//
SAIGA_CORE_API std::vector<unsigned char> compress(const void* data, std::size_t size);
SAIGA_CORE_API std::vector<unsigned char> uncompress(const void* data);

// Like above for untrusted data: never reads more than 'size' bytes and returns nothing if the data is invalid or
// would decompress to more than max_decompressed_size bytes.
SAIGA_CORE_API std::optional<std::vector<unsigned char>> uncompress(const void* data, std::size_t size,
                                                                    std::size_t max_decompressed_size);
}  // namespace Saiga

#endif
