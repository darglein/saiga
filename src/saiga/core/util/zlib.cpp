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

int compress3(Bytef* dest, size_t* destLen, const Bytef* source, size_t sourceLen, int level = Z_DEFAULT_COMPRESSION)
{
    z_stream stream;
    int err;
    const uInt max = (uInt)-1;
    size_t left;

    left     = *destLen;
    *destLen = 0;

    stream.zalloc = (alloc_func)0;
    stream.zfree  = (free_func)0;
    stream.opaque = (voidpf)0;

    err = deflateInit(&stream, level);
    if (err != Z_OK) return err;

    stream.next_out  = dest;
    stream.avail_out = 0;
    stream.next_in   = (z_const Bytef*)source;
    stream.avail_in  = 0;

    size_t total_out_uint64 = 0;

    do
    {
        if (stream.avail_out == 0)
        {
            stream.avail_out = left > (size_t)max ? max : (uInt)left;
            left -= (size_t)stream.avail_out;
        }
        if (stream.avail_in == 0)
        {
            stream.avail_in = sourceLen > (size_t)max ? max : (uInt)sourceLen;
            sourceLen -= (size_t)stream.avail_in;
        }
        err = deflate(&stream, sourceLen ? Z_NO_FLUSH : Z_FINISH);
        total_out_uint64 += stream.total_out;

    } while (err == Z_OK);

    *destLen = total_out_uint64;
    deflateEnd(&stream);
    return err == Z_STREAM_END ? Z_OK : err;
}

size_t compressBound3(size_t sourceLen)
{
    return sourceLen + (sourceLen >> 12) + (sourceLen >> 14) + (sourceLen >> 25) + 13;
}

std::vector<unsigned char> compress(const void* data, size_t decompressed_data_size)
{
    size_t c_bounds = compressBound3(decompressed_data_size);

    std::vector<Byte> result(c_bounds + header_size);
    size_t* out_header = (size_t*)result.data();
    Byte* out_data     = result.data() + header_size;

    size_t compressed_data_size = c_bounds;
    compress3(out_data, &compressed_data_size, (const Byte*)data, decompressed_data_size);

    // Write header
    out_header[0] = magic_value;
    out_header[1] = compressed_data_size;
    out_header[2] = decompressed_data_size;

    result.resize(compressed_data_size + header_size);
    return result;
}

int uncompress3(Bytef* dest, size_t* destLen, const Bytef* source, size_t* sourceLen)
{
    z_stream stream;
    int err;
    const uInt max = (uInt)-1;
    size_t len, left;
    Byte buf[1]; /* for detection of incomplete stream when *destLen == 0 */

    len = *sourceLen;
    if (*destLen)
    {
        left     = *destLen;
        *destLen = 0;
    }
    else
    {
        left = 1;
        dest = buf;
    }

    stream.next_in  = (z_const Bytef*)source;
    stream.avail_in = 0;
    stream.zalloc   = (alloc_func)0;
    stream.zfree    = (free_func)0;
    stream.opaque   = (voidpf)0;

    err = inflateInit(&stream);
    if (err != Z_OK) return err;

    stream.next_out  = dest;
    stream.avail_out = 0;

    size_t total_out_uint64 = 0;
    do
    {
        if (stream.avail_out == 0)
        {
            stream.avail_out = left > (uLong)max ? max : (uInt)left;
            left -= (size_t)stream.avail_out;
        }
        if (stream.avail_in == 0)
        {
            stream.avail_in = len > (uLong)max ? max : (uInt)len;
            len -= (size_t)stream.avail_in;
        }
        err = inflate(&stream, Z_NO_FLUSH);
        total_out_uint64 += stream.total_out;
        stream.total_out = 0;
    } while (err == Z_OK);

    *sourceLen -= len + stream.avail_in;
    if (dest != buf)
        *destLen = total_out_uint64;
    else if (stream.total_out && err == Z_BUF_ERROR)
        left = 1;

    inflateEnd(&stream);
    return err == Z_STREAM_END                             ? Z_OK
           : err == Z_NEED_DICT                            ? Z_DATA_ERROR
           : err == Z_BUF_ERROR && left + stream.avail_out ? Z_DATA_ERROR
                                                           : err;
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
    size_t actual_out_size = decompressed_data_size;
    uncompress3(result.data(), &actual_out_size, compressed_data, &compressed_data_size);

    SAIGA_ASSERT(actual_out_size == decompressed_data_size);

    return result;
}

}  // namespace Saiga

#endif


///**
// * Copyright (c) 2021 Darius Rückert
// * Licensed under the MIT License.
// * See LICENSE file for more information.
// */
//
//#include "saiga/core/util/zlib.h"
//
//#include "saiga/core/util/assert.h"
//
//#ifdef SAIGA_USE_ZLIB
//#    include <zlib.h>
//namespace Saiga
//{
//constexpr size_t header_size = 3 * sizeof(size_t);
//constexpr size_t magic_value = 0x6712956A9725DEUL;
//
//std::vector<unsigned char> compress(const void* data, size_t decompressed_data_size)
//{
//    size_t c_bounds = compressBound(decompressed_data_size);
//
//    std::vector<Byte> result(c_bounds + header_size);
//    size_t* out_header = (size_t*)result.data();
//    Byte* out_data     = result.data() + header_size;
//
//    uLongf compressed_data_size = c_bounds;
//    ::compress(out_data, &compressed_data_size, (const Byte*)data, decompressed_data_size);
//
//    // Write header
//    out_header[0] = magic_value;
//    out_header[1] = compressed_data_size;
//    out_header[2] = decompressed_data_size;
//
//    result.resize(compressed_data_size + header_size);
//    return result;
//}
//
//std::vector<unsigned char> uncompress(const void* data)
//{
//    const Byte* bdata           = (const Byte*)data;
//    const size_t* header        = (const size_t*)bdata;
//    const Byte* compressed_data = bdata + header_size;
//
//    // Read Header
//    SAIGA_ASSERT(header[0] == magic_value);
//    size_t compressed_data_size   = header[1];
//    size_t decompressed_data_size = header[2];
//
//
//    std::vector<unsigned char> result(decompressed_data_size);
//    uLongf actual_out_size = decompressed_data_size;
//    ::uncompress(result.data(), &actual_out_size, compressed_data, compressed_data_size);
//
//    SAIGA_ASSERT(actual_out_size == decompressed_data_size);
//
//    return result;
//}
//
//}  // namespace Saiga
//
//#endif
