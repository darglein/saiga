/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The handshake messages (docs/engine_design.md §13.2, §13.6; S26): raw engine messages, little-endian fixed-width
// integers and strings as varint length + bytes (WireFormat.h). The IDs 0..2 and the prefixes of JoinRequest (the
// protocol version) and JoinReject (reason, text) never change (D3), so a peer of another protocol version always
// reads the rejection.
//
//   JoinRequest  client -> host, Control: protocol version (u32); schema hash (u64); the section hashes (count u8, u64
//                each: types, messages, classes, actions, templates, flags); render table hash (u64); seed (u64);
//                flags (u8: bit 0 environment enabled, bit 1 non-legacy generation); generator count (u16); per
//                generator in name order: name (string), version() (u16), slot count (u16), per slot its name (string)
//   JoinReject   host -> client, Control: reason (u8, JoinRejectReason); text (string)
//   JoinAccept   host -> client, State: epoch (u32); flags (u8: bit 0 generation info present); generation info
//   Resync (v2)  host -> clients, State: the layout of JoinAccept (a host load)
//
// Bounds: names <= 256 bytes, <= 1024 generators, <= 255 slots, a JoinRequest <= 256 KB, a text <= 4096 bytes; unknown
// flag bits and bytes left over are errors.
//
// kProtocolVersion changes with the engine constants of the framing (S15's packet and fragment layout), the handshake
// fields after the frozen prefixes and the semantics of the chunk sync (ChunkMessages.h). Message IDs, names, versions
// and channels are covered by the network schema hash (NetSchema.h), not by the protocol version.

#pragma once

#include "saiga/engine/net/GenerationInfo.h"
#include "saiga/engine/net/NetSchema.h"
#include "saiga/export.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
constexpr uint32_t kProtocolVersion        = 1;
constexpr size_t kMaxJoinRequestBytes      = size_t(256) << 10;
constexpr size_t kMaxJoinRejectTextBytes   = 4096;
constexpr uint8_t kJoinFlagEnvironment     = 1;
constexpr uint8_t kJoinFlagNonLegacy       = 2;
constexpr uint8_t kJoinAcceptHasGeneration = 1;

enum class JoinRejectReason : uint8_t
{
    ProtocolVersion,
    Environment,
    Schema,
    RenderTables,
    Seed,
    Generators,
    Malformed,
    Count
};
SAIGA_ENGINE_API const char* joinRejectReasonName(JoinRejectReason reason);

struct JoinRequest
{
    uint32_t protocolVersion = kProtocolVersion;
    uint64_t schemaHash      = 0;
    std::array<uint64_t, kNetSchemaSections> sectionHashes{};
    uint64_t renderTableHash = 0;
    uint64_t seed            = 0;
    bool environment         = false;
    bool nonLegacy           = false;
    struct Generator
    {
        std::string name;
        uint16_t version = 0;
        std::vector<std::string> slots;
    };
    std::vector<Generator> generators;  // name order
};

struct JoinReject
{
    JoinRejectReason reason = JoinRejectReason::Malformed;
    std::string text;
};

// JoinAccept and Resync v2
struct JoinAccept
{
    uint32_t epoch     = 0;
    bool hasGeneration = false;
    GenerationInfo generation;
};

SAIGA_ENGINE_API void encodeJoinRequest(const JoinRequest& m, std::vector<uint8_t>& out);
// The frozen prefix: the protocol version of a JoinRequest of any version (false: fewer than 4 bytes)
SAIGA_ENGINE_API bool joinRequestVersion(const uint8_t* data, size_t size, uint32_t& version);
// Untrusted input of this protocol version: false with the error
SAIGA_ENGINE_API bool decodeJoinRequest(const uint8_t* data, size_t size, JoinRequest& out, std::string& error);
SAIGA_ENGINE_API void encodeJoinReject(const JoinReject& m, std::vector<uint8_t>& out);
SAIGA_ENGINE_API bool decodeJoinReject(const uint8_t* data, size_t size, JoinReject& out, std::string& error);
SAIGA_ENGINE_API void encodeJoinAccept(const JoinAccept& m, std::vector<uint8_t>& out);
// what: "JoinAccept" or "Resync" (the error text)
SAIGA_ENGINE_API bool decodeJoinAccept(const uint8_t* data, size_t size, JoinAccept& out, std::string& error,
                                       const char* what = "JoinAccept");

}  // namespace Engine
}  // namespace Saiga
