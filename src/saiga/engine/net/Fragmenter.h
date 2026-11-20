/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Splits large reliable packets into fragments of at most kMaxPacket bytes (a 12-byte header: packet sequence, index,
// count) and reassembles them (S15; the loopback's fragmenter, reusable by the S33 transports for messages above a
// backend's maximum). The reassembler treats fragments as untrusted input: it returns errors, never asserts.

#pragma once

#include "saiga/engine/net/Transport.h"
#include "saiga/export.h"

#include <map>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
constexpr size_t kFragmentHeader  = 12;
constexpr size_t kFragmentPayload = kMaxPacket - kFragmentHeader;
constexpr uint32_t kMaxFragments  = uint32_t((kMaxReliablePacket + kFragmentPayload - 1) / kFragmentPayload);

// The fragments of one packet (each at most kMaxPacket bytes)
SAIGA_ENGINE_API std::vector<std::vector<uint8_t>> fragmentPacket(uint32_t sequence,
                                                                  const std::vector<uint8_t>& packet);

class SAIGA_ENGINE_API Reassembler
{
   public:
    enum class Result
    {
        Incomplete,  // stored, more fragments to come
        Complete,    // packet holds the reassembled packet
        Error,       // malformed or over the limits (error says why); the stream is broken
    };
    // One fragment of the stream (per sender and channel)
    Result add(const std::vector<uint8_t>& fragment, std::vector<uint8_t>& packet, std::string& error);
    size_t pendingBytes() const { return bytes; }

   private:
    struct Partial
    {
        uint32_t count    = 0;
        uint32_t received = 0;
        std::vector<std::vector<uint8_t>> parts;
    };
    std::map<uint32_t, Partial> partial;
    size_t bytes = 0;
};

}  // namespace Engine
}  // namespace Saiga
