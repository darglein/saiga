/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The streaming test setup (docs/engine_design.md §12.3; S24): shared by test_engine_streaming,
// test_engine_streaming_scenarios and test_engine_streaming_benchmark (S25 and S26 extend it).
// streamingTestSetup() = contentTestSetup(2) (overworld 128 chunks at 2 m with test_tree / test_rock slots, sites 0 and
// 1, test_valley, test_dungeon) with WorldConfig::streaming, an appended flat open-world template test_flat (no
// terrain, no slots: chunks without parts), an appended test_small (test_terrain, 16 chunks at 2 m, no sites: the
// visual-path tests) and small radii on every template (simulationRadius 64, fullRadius 128, prefetchMargin 64,
// unloadMargin 64), plus the test game objects Ticker (counts its updates in the saved component TickCount) and Drifter
// (writes a seeded CharacterIntent in update) and the phase-4 test system count (TickCount::systemSteps over
// view<TickCount>). Not a generator file (not linted).

#pragma once

#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/StreamingAnchor.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/streaming/SimulationSet.h"
#include "saiga/engine/world/World.h"

#include <memory>
#include <string>

#include "content_test_setup.h"

namespace StreamingTest
{
using namespace Saiga;
using namespace Saiga::Engine;

enum class StreamingTestClass : uint16_t
{
    Ticker = uint16_t(ContentTest::ContentTestClass::Count),
    Drifter,
    Count
};

enum class StreamingTestComponent : uint8_t
{
    TickCount = kFirstGameComponent,
    Count
};

// The update calls of a Ticker and the phase-4 count system's visits (saved, replicated)
struct TickCount
{
    static constexpr const char* name = "TickCount";
    static constexpr ComponentId id   = ComponentId(StreamingTestComponent::TickCount);
    static constexpr uint32_t version = 1;
    uint32_t updates                  = 0;
    uint32_t systemSteps              = 0;
    template <class V>
    void visit(V& v)
    {
        v("updates", updates);
        v("systemSteps", systemSteps);
    }
};

// Counts its update calls
class Ticker final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(StreamingTestClass::Ticker);
    static constexpr const char* name = "Ticker";
    void onSpawn() override
    {
        if (!has<TickCount>()) add<TickCount>();
    }
    void update(double) override
    {
        if (has<TickCount>()) write<TickCount>().updates++;
    }
};

// Walks: a seeded direction per entity that changes every 64 steps, 2 m/s
class Drifter final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(StreamingTestClass::Drifter);
    static constexpr const char* name = "Drifter";
    void onSpawn() override
    {
        if (!has<CharacterIntent>()) add<CharacterIntent>();
    }
    void update(double) override
    {
        if (!has<CharacterIntent>()) return;
        const uint64_t h                         = hashCombine(uint64_t(entity().index), world().stepIndex() / 64);
        const double a                           = double(h % 6283) / 1000.0;
        write<CharacterIntent>().desiredVelocity = Vec3(std::cos(a) * 2, 0, std::sin(a) * 2);
    }
};

inline void countSystem(World& w, double)
{
    w.view<TickCount>().each([](Entity, TickCount& c) { c.systemSteps++; });
}

inline void registerStreamingTestClasses(ClassList& c)
{
    ContentTest::registerContentTestClasses(c);
    c.add<Ticker>();
    c.add<Drifter>();
}

inline void registerStreamingTestComponents(Registry& r)
{
    r.registerComponent<TickCount>(Replicate::Host);
}

inline void registerStreamingTestSystems(SystemList& s)
{
    s.add(Phase::Gameplay, "count", countSystem, RunOn::Host);
}

// The setup (cache: null: a new one per setup; maxVersion of test_terrain: 1 or 2, as contentTestSetup)
inline WorldSetup streamingTestSetup(std::shared_ptr<ContentCache> cache = nullptr, uint16_t maxVersion = 2)
{
    WorldSetup s       = ContentTest::contentTestSetup(maxVersion, cache);
    s.config.streaming = true;
    SpaceTemplate flat;
    flat.name       = "test_flat";
    flat.generator  = 0;
    flat.sizeChunks = 16;
    s.config.spaceTemplates.push_back(flat);
    // A small terrain space for the tests of the visual path with pinned tiles (its pyramid is 129^2 samples; the
    // overworld's 1025^2 cost 0.4 s per process)
    SpaceTemplate small = s.config.spaceTemplates[0];
    small.name          = "test_small";
    small.sizeChunks    = 16;
    s.config.spaceTemplates.push_back(small);
    for (SpaceTemplate& t : s.config.spaceTemplates)
    {
        t.simulationRadius = 64;
        t.fullRadius       = 128;
        t.prefetchMargin   = 64;
        t.unloadMargin     = 64;
    }
    s.classes    = registerStreamingTestClasses;
    s.components = registerStreamingTestComponents;
    s.systems    = registerStreamingTestSystems;
    return s;
}

// One content cache per process for the tests that count no generations, cache hits or jobs (AGENTS.md, test speed)
inline std::shared_ptr<ContentCache> sharedStreamingCache()
{
    static std::shared_ptr<ContentCache> c = std::make_shared<ContentCache>();
    return c;
}

// The space of a template by name
inline SpaceId spaceNamed(const World& w, const std::string& name)
{
    for (size_t i = 0; i < w.spaceCount(); ++i)
        if (w.spaceTemplate(SpaceId(i)).name == name) return SpaceId(i);
    return SpaceId(~0u);
}

// A Player root at a position (space 0 unless given)
inline Entity addPlayer(World& w, const Vec3& p, SpaceId space = 0, double moveSpeed = 5)
{
    const Entity e = w.createIn(space);
    Player pl;
    pl.moveSpeed = moveSpeed;
    w.add<Player>(e, pl);
    w.add<CharacterIntent>(e);
    w.get<Transform>(e).teleport(p, Quat::Identity());
    return e;
}

inline ChunkId chunkOf(SpaceId space, double x, double z)
{
    ChunkId c;
    c.space = space;
    c.x     = int32_t(std::floor(x / 256));
    c.z     = int32_t(std::floor(z / 256));
    return c;
}

}  // namespace StreamingTest
