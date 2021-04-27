/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/zlib.h"

#include "saiga/core/util/assert.h"

#ifdef SAIGA_USE_ZLIB
#    include <zlib.h>
namespace Saiga
{
constexpr size_t header_size = 3 * sizeof(size_t);
constexpr size_t magic_value = 0x6712956A9725DEUL;

std::vector<unsigned char> compress(const void* data, size_t decompressed_data_size)
{
    size_t c_bounds = compressBound(decompressed_data_size);

    std::vector<Byte> result(c_bounds + header_size);
    size_t* out_header = (size_t*)result.data();
    Byte* out_data     = result.data() + header_size;

    uLongf compressed_data_size = c_bounds;
    ::compress(out_data, &compressed_data_size, (const Byte*)data, decompressed_data_size);

    // Write header
    out_header[0] = magic_value;
    out_header[1] = compressed_data_size;
    out_header[2] = decompressed_data_size;

    result.resize(compressed_data_size + header_size);
    return result;
}

std::vector<unsigned char> uncompress(const void* data)
{
    const Byte* bdata           = (const Byte*)data;
    const size_t* header        = (const size_t*)bdata;
    const Byte* compressed_data = bdata + header_size;

    // Read Header
    SAIGA_ASSERT(header[0] == magic_value);
    size_t compressed_data_size   = header[1];
    size_t decompressed_data_size = header[2];


    std::vector<unsigned char> result(decompressed_data_size);
    uLongf actual_out_size = decompressed_data_size;
    ::uncompress(result.data(), &actual_out_size, compressed_data, compressed_data_size);

    SAIGA_ASSERT(actual_out_size == decompressed_data_size);

    return result;
}

}  // namespace Saiga

#endif
