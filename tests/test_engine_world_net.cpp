/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The world over the network (docs/engine_design.md §13.2, §13.6; S26): the network schema, the handshake (JoinRequest,
// JoinReject, JoinAccept, Resync v2, admission), chunk interest, snapshots and indexes, the client's per-chunk queue,
// re-requests, unloads and the host's subscriptions, over LoopbackTransport.

#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/net/NetSchema.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <string>

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// ---- NetSchema ------------------------------------------------------------------------------------------------------

// The registration variants of HashCoversRegistration (selected by g_variant before a world is built)
int g_variant = 0;
enum Variant
{
    Base,
    FieldAdded,
    ComponentVersion,
    ReplicateMode,
    Environment,
    EventAdded,
    ClassAdded,
    MessageVersion,
    MessageChannel,
    ActionAdded,
    HeldAdded,
    TemplateSpacing,
    Streaming,
    NotSaved,
    StreamingRadius,
    VariantCount
};

struct SchemaComp
{
    static constexpr const char* name = "SchemaComp";
    static constexpr ComponentId id   = kFirstGameComponent;
    static constexpr uint32_t version = 1;
    int32_t a                         = 0;
    template <class V>
    void visit(V& v)
    {
        v("a", a);
    }
};
struct SchemaCompField
{
    static constexpr const char* name = "SchemaComp";
    static constexpr ComponentId id   = kFirstGameComponent;
    static constexpr uint32_t version = 1;
    int32_t a = 0, b = 0;
    template <class V>
    void visit(V& v)
    {
        v("a", a);
        v("b", b);
    }
};
struct SchemaCompV2
{
    static constexpr const char* name = "SchemaComp";
    static constexpr ComponentId id   = kFirstGameComponent;
    static constexpr uint32_t version = 2;
    int32_t a                         = 0;
    template <class V>
    void visit(V& v)
    {
        v("a", a);
    }
};
struct SchemaEvent
{
    static constexpr const char* name = "SchemaEvent";
    static constexpr EventId id       = 64;
    static constexpr uint32_t version = 1;
    int32_t value                     = 0;
    template <class V>
    void visit(V& v)
    {
        v("value", value);
    }
};
class SchemaObject final : public GameObject
{
   public:
    static constexpr ClassId id       = kFirstGameClass;
    static constexpr const char* name = "SchemaObject";
};
struct SchemaMessage
{
    static constexpr MessageId id     = 64;
    static constexpr const char* name = "SchemaMessage";
    static constexpr uint32_t version = 1;
    int32_t v                         = 0;
    template <class V>
    void visit(V& vis)
    {
        vis("v", v);
    }
};
struct SchemaMessageV2
{
    static constexpr MessageId id     = 64;
    static constexpr const char* name = "SchemaMessage";
    static constexpr uint32_t version = 2;
    int32_t v                         = 0;
    template <class V>
    void visit(V& vis)
    {
        vis("v", v);
    }
};
struct SchemaMessageState
{
    static constexpr MessageId id     = 64;
    static constexpr const char* name = "SchemaMessage";
    static constexpr uint32_t version = 1;
    static constexpr Channel channel  = Channel::State;
    int32_t v                         = 0;
    template <class V>
    void visit(V& vis)
    {
        vis("v", v);
    }
};

void schemaComponents(Registry& r)
{
    if (g_variant == FieldAdded)
        r.registerComponent<SchemaCompField>(Replicate::Host);
    else if (g_variant == ComponentVersion)
        r.registerComponent<SchemaCompV2>(Replicate::Host);
    else if (g_variant == ReplicateMode)
        r.registerComponent<SchemaComp>(Replicate::HostReliable);
    else if (g_variant == NotSaved)
        r.registerComponent<SchemaComp>(Replicate::Host, Saved::No);
    else
        r.registerComponent<SchemaComp>(Replicate::Host);
}
void schemaEvents(EventRegistry& e)
{
    if (g_variant == EventAdded) e.add<SchemaEvent>();
}
void schemaClasses(ClassList& c)
{
    if (g_variant == ClassAdded) c.add<SchemaObject>();
}
void onSchemaMessage(World&, uint8_t, const SchemaMessage&) {}
void onSchemaMessageV2(World&, uint8_t, const SchemaMessageV2&) {}
void onSchemaMessageState(World&, uint8_t, const SchemaMessageState&) {}
void schemaMessages(MessageList& l)
{
    if (g_variant == MessageVersion)
        l.add<SchemaMessageV2>(Reliability::Reliable, onSchemaMessageV2);
    else if (g_variant == MessageChannel)
        l.add<SchemaMessageState>(Reliability::Reliable, onSchemaMessageState);
    else
        l.add<SchemaMessage>(Reliability::Reliable, onSchemaMessage);
}
void schemaActions(ActionList& a)
{
    a.add(1, "jump");
    a.addHeld(0, "fire");
    if (g_variant == ActionAdded) a.add(2, "crouch");
    if (g_variant == HeldAdded) a.addHeld(1, "aim");
}

WorldSetup schemaSetup(int variant)
{
    g_variant = variant;
    WorldSetup s;
    s.profiling  = false;
    s.components = schemaComponents;
    s.events     = schemaEvents;
    s.classes    = schemaClasses;
    s.messages   = schemaMessages;
    s.actions    = schemaActions;
    SpaceTemplate t;
    t.name       = "flat";
    t.sizeChunks = 16;
    if (variant == TemplateSpacing) t.baseSpacing = 1;
    if (variant == StreamingRadius) t.fullRadius = 100;
    s.config.spaceTemplates.push_back(t);
    s.config.streaming    = variant == Streaming;
    s.environment.enabled = variant == Environment;
    return s;
}

TEST(NetSchema, EqualSetupsHashEqual)
{
    World a(schemaSetup(Base)), b(schemaSetup(Base));
    EXPECT_EQ(a.netSchemaHash(), b.netSchemaHash());
    EXPECT_EQ(a.netSchema().sectionHashes(), b.netSchema().sectionHashes());
    EXPECT_EQ(a.netSchemaHash(), a.netSchema().hash());
    std::vector<uint8_t> ea, eb;
    a.netSchema().encode(ea);
    b.netSchema().encode(eb);
    EXPECT_EQ(ea, eb);
    // Every registered component is in it (also Saved::No ones), the messages with their versions
    bool comp = false;
    for (const NetSchema::Type& t : a.netSchema().types) comp = comp || t.name == "SchemaComp";
    EXPECT_TRUE(comp);
    EXPECT_FALSE(a.netSchema().messages.empty());
}

TEST(NetSchema, HashCoversRegistration)
{
    World base(schemaSetup(Base));
    const auto baseSections = base.netSchema().sectionHashes();
    // The section that must differ per variant
    const std::vector<std::pair<int, NetSchemaSection>> changes = {
        {FieldAdded, NetSchemaSection::Types},          {ComponentVersion, NetSchemaSection::Types},
        {ReplicateMode, NetSchemaSection::Types},       {Environment, NetSchemaSection::Types},
        {EventAdded, NetSchemaSection::Types},          {ClassAdded, NetSchemaSection::Classes},
        {MessageVersion, NetSchemaSection::Messages},   {MessageChannel, NetSchemaSection::Messages},
        {ActionAdded, NetSchemaSection::Actions},       {HeldAdded, NetSchemaSection::Actions},
        {TemplateSpacing, NetSchemaSection::Templates}, {Streaming, NetSchemaSection::Flags}};
    for (const auto& [variant, section] : changes)
    {
        World w(schemaSetup(variant));
        EXPECT_NE(w.netSchemaHash(), base.netSchemaHash()) << "variant " << variant;
        const auto sections = w.netSchema().sectionHashes();
        // Streaming also registers its engine components (the types change too)
        for (size_t i = 0; i < kNetSchemaSections; ++i)
        {
            if (variant == Streaming && i != size_t(section)) continue;
            EXPECT_EQ(sections[i] != baseSections[i], i == size_t(section))
                << "variant " << variant << ", section " << netSchemaSectionName(NetSchemaSection(i));
        }
    }
    // Not in the schema: Saved, the streaming radii (data files are not read by it either)
    for (int variant : {int(NotSaved), int(StreamingRadius)})
    {
        World w(schemaSetup(variant));
        EXPECT_EQ(w.netSchemaHash(), base.netSchemaHash()) << "variant " << variant;
    }
}

}  // namespace
