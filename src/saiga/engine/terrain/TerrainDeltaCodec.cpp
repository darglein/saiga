/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/terrain/TerrainDeltaCodec.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/net/Varint.h"

#include <zstd.h>

namespace Saiga
{
namespace Engine
{
namespace TerrainDeltaCodec
{
namespace
{
void encodeRaw(const int32_t* values, size_t count, std::vector<uint8_t>& out)
{
    int64_t previous = 0;
    for (size_t i = 0; i < count; ++i)
    {
        SAIGA_ASSERT(values[i] >= -kMaxTerrainDelta && values[i] <= kMaxTerrainDelta,
                     "TerrainDeltaCodec: a delta beyond 2^24");
        writeVarint(out, zigzag(int64_t(values[i]) - previous));
        previous = values[i];
    }
}

bool fail(std::string* error, const std::string& message)
{
    if (error) *error = "terrain deltas: " + message;
    return false;
}

bool decodeRaw(const uint8_t* data, size_t size, size_t count, int32_t* out, std::string* error)
{
    size_t pos       = 0;
    int64_t previous = 0;
    for (size_t i = 0; i < count; ++i)
    {
        uint64_t v = 0;
        if (!readVarint(data, size, pos, v)) return fail(error, "truncated or malformed varint");
        // Bounded before the difference is added (no overflow on hostile input)
        if (v > (uint64_t(1) << 27)) return fail(error, "a difference beyond 2^26");
        const int64_t value = previous + unzigzag(v);
        if (value < -kMaxTerrainDelta || value > kMaxTerrainDelta) return fail(error, "a delta beyond 2^24");
        out[i]   = int32_t(value);
        previous = value;
    }
    if (pos != size) return fail(error, "bytes left over");
    return true;
}
}  // namespace

void encode(const int32_t* values, size_t count, bool zstd, std::vector<uint8_t>& out)
{
    if (!zstd)
    {
        encodeRaw(values, count, out);
        return;
    }
    std::vector<uint8_t> raw;
    raw.reserve(count * 2);
    encodeRaw(values, count, raw);
    const size_t at = out.size();
    out.resize(at + ZSTD_compressBound(raw.size()));
    const size_t n = ZSTD_compress(out.data() + at, out.size() - at, raw.data(), raw.size(), kTerrainDeltaZstdLevel);
    SAIGA_ASSERT(!ZSTD_isError(n), "TerrainDeltaCodec: zstd failed");
    out.resize(at + n);
}

bool decode(const uint8_t* data, size_t size, bool zstd, size_t count, int32_t* out, std::string* error)
{
    if (!zstd) return decodeRaw(data, size, count, out, error);
    // At most 4 varint bytes per value
    const size_t bound           = count * 4;
    const unsigned long long len = ZSTD_getFrameContentSize(data, size);
    if (len == ZSTD_CONTENTSIZE_ERROR || len == ZSTD_CONTENTSIZE_UNKNOWN) return fail(error, "bad zstd frame");
    if (len > bound) return fail(error, "the decompressed size exceeds the bound");
    // The frame must span all bytes (nothing left over)
    const size_t frame = ZSTD_findFrameCompressedSize(data, size);
    if (ZSTD_isError(frame) || frame != size) return fail(error, "bad zstd frame or bytes left over");
    std::vector<uint8_t> raw(static_cast<size_t>(len));
    const size_t got = ZSTD_decompress(raw.data(), raw.size(), data, size);
    if (ZSTD_isError(got) || got != raw.size()) return fail(error, "bad zstd stream");
    return decodeRaw(raw.data(), raw.size(), count, out, error);
}
}  // namespace TerrainDeltaCodec

}  // namespace Engine
}  // namespace Saiga
