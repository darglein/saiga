/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/Fragmenter.h"

#include <algorithm>

namespace Saiga
{
namespace Engine
{
namespace
{
void putU32(std::vector<uint8_t>& out, uint32_t v)
{
    for (int i = 0; i < 4; ++i) out.push_back(uint8_t(v >> (8 * i)));
}
uint32_t getU32(const uint8_t* p)
{
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
// At most this many packets in incomplete reassemblies per stream
constexpr size_t kMaxPartialPackets = 64;
}  // namespace

std::vector<std::vector<uint8_t>> fragmentPacket(uint32_t sequence, const std::vector<uint8_t>& packet)
{
    const size_t count = std::max<size_t>(1, (packet.size() + kFragmentPayload - 1) / kFragmentPayload);
    std::vector<std::vector<uint8_t>> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        std::vector<uint8_t> f;
        const size_t begin = i * kFragmentPayload;
        const size_t end   = std::min(packet.size(), begin + kFragmentPayload);
        f.reserve(kFragmentHeader + (end - begin));
        putU32(f, sequence);
        putU32(f, uint32_t(i));
        putU32(f, uint32_t(count));
        f.insert(f.end(), packet.begin() + std::ptrdiff_t(begin), packet.begin() + std::ptrdiff_t(end));
        out.push_back(std::move(f));
    }
    return out;
}

Reassembler::Result Reassembler::add(const std::vector<uint8_t>& fragment, std::vector<uint8_t>& packet,
                                     std::string& error)
{
    if (fragment.size() < kFragmentHeader || fragment.size() > kMaxPacket)
    {
        error = "a fragment of " + std::to_string(fragment.size()) + " bytes";
        return Result::Error;
    }
    const uint32_t sequence = getU32(fragment.data());
    const uint32_t index    = getU32(fragment.data() + 4);
    const uint32_t count    = getU32(fragment.data() + 8);
    const size_t payload    = fragment.size() - kFragmentHeader;
    if (count == 0 || count > kMaxFragments || index >= count)
    {
        error = "fragment " + std::to_string(index) + " of " + std::to_string(count);
        return Result::Error;
    }
    // Every fragment but the last is full
    if ((index + 1 < count && payload != kFragmentPayload) || (payload == 0 && count > 1))
    {
        error = "fragment " + std::to_string(index) + " has " + std::to_string(payload) + " bytes";
        return Result::Error;
    }
    if (count == 1)
    {
        if (partial.count(sequence))
        {
            error = "packet " + std::to_string(sequence) + " changed its fragment count";
            return Result::Error;
        }
        packet.assign(fragment.begin() + kFragmentHeader, fragment.end());
        return Result::Complete;
    }
    auto it = partial.find(sequence);
    if (it == partial.end())
    {
        if (partial.size() >= kMaxPartialPackets)
        {
            error = "too many incomplete packets";
            return Result::Error;
        }
        it               = partial.emplace(sequence, Partial{}).first;
        it->second.count = count;
        it->second.parts.resize(count);
    }
    Partial& p = it->second;
    if (p.count != count)
    {
        error = "packet " + std::to_string(sequence) + " changed its fragment count";
        return Result::Error;
    }
    if (!p.parts[index].empty())
    {
        error = "fragment " + std::to_string(index) + " of packet " + std::to_string(sequence) + " twice";
        return Result::Error;
    }
    if (bytes + payload > kMaxReliablePacket)
    {
        error = "incomplete packets over " + std::to_string(kMaxReliablePacket) + " bytes";
        return Result::Error;
    }
    p.parts[index].assign(fragment.begin() + kFragmentHeader, fragment.end());
    bytes += payload;
    p.received++;
    if (p.received < p.count) return Result::Incomplete;
    packet.clear();
    for (auto& part : p.parts)
    {
        bytes -= part.size();
        packet.insert(packet.end(), part.begin(), part.end());
    }
    partial.erase(it);
    return Result::Complete;
}

}  // namespace Engine
}  // namespace Saiga
