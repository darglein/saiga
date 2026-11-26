/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/JoinMessages.h"

namespace Saiga
{
namespace Engine
{
namespace
{
constexpr size_t kMaxNameBytes  = 256;
constexpr size_t kMaxGenerators = 1024;
constexpr size_t kMaxSlots      = 255;
}  // namespace

const char* joinRejectReasonName(JoinRejectReason reason)
{
    switch (reason)
    {
        case JoinRejectReason::ProtocolVersion:
            return "protocol_version";
        case JoinRejectReason::Environment:
            return "environment";
        case JoinRejectReason::Schema:
            return "schema";
        case JoinRejectReason::RenderTables:
            return "render_tables";
        case JoinRejectReason::Seed:
            return "seed";
        case JoinRejectReason::Generators:
            return "generators";
        case JoinRejectReason::Malformed:
            return "malformed";
        case JoinRejectReason::Count:
            break;
    }
    return "unknown";
}

void encodeJoinRequest(const JoinRequest& m, std::vector<uint8_t>& out)
{
    putU32(out, m.protocolVersion);
    putU64(out, m.schemaHash);
    putU8(out, uint8_t(m.sectionHashes.size()));
    for (uint64_t h : m.sectionHashes) putU64(out, h);
    putU64(out, m.renderTableHash);
    putU64(out, m.seed);
    putU8(out, uint8_t((m.environment ? kJoinFlagEnvironment : 0) | (m.nonLegacy ? kJoinFlagNonLegacy : 0)));
    putU16(out, uint16_t(m.generators.size()));
    for (const JoinRequest::Generator& g : m.generators)
    {
        putString(out, g.name);
        putU16(out, g.version);
        putU16(out, uint16_t(g.slots.size()));
        for (const std::string& s : g.slots) putString(out, s);
    }
}

bool joinRequestVersion(const uint8_t* data, size_t size, uint32_t& version)
{
    WireReader r(data, size);
    version = r.u32();
    return r.ok;
}

bool decodeJoinRequest(const uint8_t* data, size_t size, JoinRequest& out, std::string& error)
{
    out = JoinRequest();
    if (size > kMaxJoinRequestBytes) return (error = "JoinRequest: above 256 KB", false);
    WireReader r(data, size);
    out.protocolVersion   = r.u32();
    out.schemaHash        = r.u64();
    const uint8_t section = r.u8();
    if (!r.ok) return (error = "JoinRequest: truncated", false);
    if (section != kNetSchemaSections) return (error = "JoinRequest: another number of schema sections", false);
    for (uint64_t& h : out.sectionHashes) h = r.u64();
    out.renderTableHash = r.u64();
    out.seed            = r.u64();
    const uint8_t flags = r.u8();
    const uint16_t gens = r.u16();
    if (!r.ok) return (error = "JoinRequest: truncated", false);
    if (flags & ~uint8_t(kJoinFlagEnvironment | kJoinFlagNonLegacy))
        return (error = "JoinRequest: unknown flags", false);
    out.environment = (flags & kJoinFlagEnvironment) != 0;
    out.nonLegacy   = (flags & kJoinFlagNonLegacy) != 0;
    // A generator takes at least 5 bytes, a slot 1
    if (gens > kMaxGenerators || uint64_t(gens) * 5 > r.remaining())
        return (error = "JoinRequest: the generator count does not fit", false);
    out.generators.resize(gens);
    for (JoinRequest::Generator& g : out.generators)
    {
        g.name               = r.string(kMaxNameBytes);
        g.version            = r.u16();
        const uint16_t slots = r.u16();
        if (!r.ok) return (error = "JoinRequest: truncated or a name above 256 bytes", false);
        if (slots > kMaxSlots || slots > r.remaining())
            return (error = "JoinRequest: the slot count does not fit", false);
        g.slots.resize(slots);
        for (std::string& s : g.slots) s = r.string(kMaxNameBytes);
        if (!r.ok) return (error = "JoinRequest: truncated or a name above 256 bytes", false);
    }
    if (!r.atEnd()) return (error = "JoinRequest: bytes left over", false);
    return true;
}

void encodeJoinReject(const JoinReject& m, std::vector<uint8_t>& out)
{
    putU8(out, uint8_t(m.reason));
    putString(out, m.text.size() > kMaxJoinRejectTextBytes ? m.text.substr(0, kMaxJoinRejectTextBytes) : m.text);
}

bool decodeJoinReject(const uint8_t* data, size_t size, JoinReject& out, std::string& error)
{
    out = JoinReject();
    WireReader r(data, size);
    const uint8_t reason = r.u8();
    out.text             = r.string(kMaxJoinRejectTextBytes);
    if (!r.ok) return (error = "JoinReject: truncated or a text above 4096 bytes", false);
    if (reason >= uint8_t(JoinRejectReason::Count)) return (error = "JoinReject: an unknown reason", false);
    out.reason = JoinRejectReason(reason);
    if (!r.atEnd()) return (error = "JoinReject: bytes left over", false);
    return true;
}

void encodeJoinAccept(const JoinAccept& m, std::vector<uint8_t>& out)
{
    putU32(out, m.epoch);
    putU8(out, m.hasGeneration ? kJoinAcceptHasGeneration : 0);
    if (m.hasGeneration) encodeGenerationInfo(m.generation, out);
}

bool decodeJoinAccept(const uint8_t* data, size_t size, JoinAccept& out, std::string& error, const char* what)
{
    out = JoinAccept();
    WireReader r(data, size);
    out.epoch           = r.u32();
    const uint8_t flags = r.u8();
    if (!r.ok) return (error = std::string(what) + ": truncated", false);
    if (flags & ~kJoinAcceptHasGeneration) return (error = std::string(what) + ": unknown flags", false);
    out.hasGeneration = (flags & kJoinAcceptHasGeneration) != 0;
    if (out.hasGeneration && !decodeGenerationInfo(r, out.generation, error))
        return (error = std::string(what) + ": " + error, false);
    if (!r.atEnd()) return (error = std::string(what) + ": bytes left over", false);
    return true;
}

}  // namespace Engine
}  // namespace Saiga
