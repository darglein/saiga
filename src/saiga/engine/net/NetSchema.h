/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The network schema (docs/engine_design.md §13.2, §13.6; S26, D4): the registration a peer's wire format depends on.
// Unlike the save SchemaTable it lists every registered component (also Saved::No) with its replicate mode and its
// kind (plain, global, space state: EnvironmentSetup::enabled shows here), the fields with their quantization, the
// events with their replicate mode, the game object classes, the messages with version, reliability, channel and raw
// flag, the player actions and held buttons, the space templates' layout and the world flags that change the protocol
// (WorldConfig::streaming). Not included: data files, prefabs (the plan hash), meshes and materials (the render table
// hash), the generators and the seed (own handshake fields), the streaming radii and Saved (they do not change the
// wire). Each section has a canonical encoding hashed with FNV-1a 64; the schema hash is the hash of the section
// hashes, so a JoinReject names the differing sections without a schema transfer.

#pragma once

#include "saiga/engine/ecs/TypeSchema.h"
#include "saiga/engine/net/NetIds.h"
#include "saiga/export.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

// The sections in the order of JoinRequest's section hashes
enum class NetSchemaSection : uint8_t
{
    Types,
    Messages,
    Classes,
    Actions,
    Templates,
    Flags,
};
constexpr size_t kNetSchemaSections = 6;
// "types", "messages", ... (the JoinReject text)
SAIGA_ENGINE_API const char* netSchemaSectionName(NetSchemaSection section);

// Built once per World at the end of its construction (constant afterwards: World::netSchema)
struct SAIGA_ENGINE_API NetSchema
{
    struct Type  // a component, an event or a nested struct
    {
        SchemaKind kind = SchemaKind::Struct;
        std::string name;
        uint32_t id = 0, version = 0;
        uint8_t replicate = 0;  // Replicate (components and events)
        uint8_t stateKind = 0;  // 0 plain, 1 global, 2 space state (components)
        std::vector<FieldInfo> fields;
    };
    struct Message
    {
        MessageId id = 0;
        std::string name;
        uint32_t version    = 0;
        uint8_t reliability = 0, channel = 0;
        bool raw = false;
    };
    struct Template
    {
        std::string name;
        uint8_t kind        = 0;
        int32_t baseSpacing = 0, sizeChunks = 0;
        uint16_t generator = 0;
    };
    std::vector<Type> types;                                // components by ID, events by ID, nested structs by name
    std::vector<std::pair<uint32_t, std::string>> classes;  // game object classes (ID, name)
    std::vector<Message> messages;                          // by ID
    std::vector<std::pair<uint32_t, std::string>> actions;  // ActionList actions (ID, name)
    std::vector<std::pair<int, std::string>> held;          // ActionList held buttons (bit, name)
    std::vector<Template> templates;                        // WorldConfig::spaceTemplates (the layout fields)
    uint32_t worldFlags = 0;                                // bit 0: WorldConfig::streaming (D19)
    // The host rejects a ChunkInterest above it, the client never asks for more: both must agree (review round 2)
    uint32_t maxChunksPerPeer = 0;

    static NetSchema fromWorld(const World& world);
    // The canonical encoding of one section (appended to out)
    void encodeSection(NetSchemaSection section, std::vector<uint8_t>& out) const;
    // The canonical encoding of every section in order
    void encode(std::vector<uint8_t>& out) const;
    // FNV-1a 64 of each section's encoding
    std::array<uint64_t, kNetSchemaSections> sectionHashes() const;
    // FNV-1a 64 over the section hashes (little-endian)
    uint64_t hash() const;
};

// The hash of section hashes (NetSchema::hash)
SAIGA_ENGINE_API uint64_t netSchemaHashOf(const std::array<uint64_t, kNetSchemaSections>& sections);

}  // namespace Engine
}  // namespace Saiga
