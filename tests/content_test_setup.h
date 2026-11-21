/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The content test setup (docs/engine_design.md §12.1; S23a, E12): shared by test_engine_content, the content scenarios
// of test_engine_scenarios and S23b's tests. contentTestSetup(maxVersion) gives the WorldSetup with the templates
// (overworld: test_terrain, 128 chunks, 2 m; test_valley: test_terrain, 16 chunks, 1 m; test_dungeon: enclosed, 16
// chunks, AuthoredSpaceGenerator), the generators, the prefabs of data/tests/engine/prefabs/, the test classes
// TestChest (adds Health and its RenderInstance in onSpawn), TestMarker (adds nothing) and TestEntrance, and the test
// meshes (renderAssets). Space IDs: 0 overworld, 1 test_valley (listed), 2 test_dungeon (site 0). Not a generator file
// (not linted).

#pragma once

#include "saiga/engine/components/Health.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/content/ContentCache.h"
#include "saiga/engine/content/GeneratorList.h"
#include "saiga/engine/content/gen/AuthoredSpaceGenerator.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "content_test_generator.h"

namespace ContentTest
{
enum class ContentTestClass : uint16_t
{
    TestChest = Saiga::Engine::kFirstGameClass,
    TestMarker,
    TestEntrance,
    Count
};

// A chest: Health and its mesh, added in onSpawn (a prefab's Health value is merged after onSpawn)
class TestChest final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ContentTestClass::TestChest);
    static constexpr const char* name = "TestChest";
    void onSpawn() override
    {
        if (!has<Health>()) add<Health>();
        if (!has<RenderInstance>()) add<RenderInstance>(RenderInstance("test/chest"));
    }
    void onLoad() override { world().setRenderMesh(entity(), "test/chest"); }
};

// Adds nothing itself (a prefab component is added after onSpawn)
class TestMarker final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ContentTestClass::TestMarker);
    static constexpr const char* name = "TestMarker";
};

// An entrance: its SpaceLink comes from the prefab's space_link at stamping
class TestEntrance final : public GameObject
{
   public:
    static constexpr ClassId id       = ClassId(ContentTestClass::TestEntrance);
    static constexpr const char* name = "TestEntrance";
};

inline void registerContentTestClasses(ClassList& c)
{
    c.add<TestChest>();
    c.add<TestMarker>();
    c.add<TestEntrance>();
}

// Small inline meshes, built once per process; LOD end distances of 4 km
inline std::shared_ptr<const Vulkan::MeshLods> contentTestBox(float sx, float sy, float sz, uint32_t color)
{
    auto lods = std::make_shared<Vulkan::MeshLods>();
    Vulkan::MeshLod lod;
    lod.endDistance = 4000;
    for (int k = 0; k < 8; ++k)
    {
        Vulkan::MeshVertex v;
        v.position = vec3((k & 1) ? sx : -sx, (k & 2) ? 2 * sy : 0, (k & 4) ? sz : -sz);
        v.color    = color;
        lod.vertices.push_back(v);
    }
    lod.indices = {0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4,
                   2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5};
    lods->lods.push_back(lod);
    lods->sphereCenter = vec3(0, sy, 0);
    lods->sphereRadius = std::sqrt(sx * sx + sy * sy + sz * sz) * 1.5f;
    return lods;
}

inline std::shared_ptr<const Vulkan::MeshLods> contentTestCone(float r, float h, uint32_t color)
{
    auto lods = std::make_shared<Vulkan::MeshLods>();
    Vulkan::MeshLod lod;
    lod.endDistance = 4000;
    Vulkan::MeshVertex top;
    top.position = vec3(0, h, 0);
    top.color    = color;
    lod.vertices.push_back(top);
    for (int k = 0; k < 8; ++k)
    {
        const float a = float(k) * 0.785398163f;
        Vulkan::MeshVertex v;
        v.position = vec3(r * std::cos(a), 0, r * std::sin(a));
        v.color    = color;
        lod.vertices.push_back(v);
    }
    for (uint32_t k = 0; k < 8; ++k) lod.indices.insert(lod.indices.end(), {0, 1 + (k + 1) % 8, 1 + k});
    lods->lods.push_back(lod);
    lods->sphereCenter = vec3(0, h / 2, 0);
    lods->sphereRadius = std::max(r, h);
    return lods;
}

inline void contentTestAssets(World& w)
{
    static const auto treeA    = contentTestCone(1.5f, 6, 0xFF2E7D32u);
    static const auto treeB    = contentTestCone(2.0f, 9, 0xFF1B5E20u);
    static const auto rock     = contentTestBox(0.8f, 0.4f, 0.6f, 0xFF757575u);
    static const auto pillar   = contentTestBox(0.5f, 2.0f, 0.5f, 0xFFBCAAA4u);
    static const auto chest    = contentTestBox(0.6f, 0.4f, 0.4f, 0xFF8D6E63u);
    const Vulkan::MaterialId m = w.registerMaterial("test/content_flat", Vulkan::MaterialData());
    w.registerMesh("test/tree_a", treeA, m);
    w.registerMesh("test/tree_b", treeB, m);
    w.registerMesh("test/rock", rock, m);
    w.registerMesh("test/pillar", pillar, m);
    w.registerMesh("test/chest", chest, m);
}

constexpr GeneratorId kTestTerrainGenerator = 1, kTestDungeonGenerator = 2;

template <uint16_t MaxVersion, Variant V>
void registerContentTestGenerators(GeneratorList& l)
{
    l.add(kTestTerrainGenerator, std::make_shared<TestTerrainGenerator>(MaxVersion, V));
    l.add(kTestDungeonGenerator, std::make_shared<AuthoredSpaceGenerator>("test_dungeon", "test_dungeon_rooms", 1));
}

inline std::vector<SpaceTemplate> contentTestTemplates()
{
    SpaceTemplate overworld;
    overworld.name        = "overworld";
    overworld.generator   = kTestTerrainGenerator;
    overworld.sizeChunks  = 128;
    overworld.baseSpacing = 2;
    overworld.seaLevel    = 0;
    SpaceTemplate valley  = overworld;
    valley.name           = "test_valley";
    valley.sizeChunks     = 16;
    valley.baseSpacing    = 1;
    SpaceTemplate dungeon = SpaceTemplate::enclosed("test_dungeon", 16);
    dungeon.generator     = kTestDungeonGenerator;
    return {overworld, valley, dungeon};
}

// The setup of the content tests: maxVersion 1 or 2 of test_terrain, a variant (tests only), the cache (null: a new
// one per setup)
inline WorldSetup contentTestSetup(uint16_t maxVersion, std::shared_ptr<ContentCache> cache = nullptr,
                                   Variant variant = Variant::Committed)
{
    WorldSetup s;
    s.seed                  = 12345;
    s.profiling             = false;
    s.dataDirectory         = std::string(SAIGA_PROJECT_SOURCE_DIR) + "/data/tests/engine";
    s.config.spaceTemplates = contentTestTemplates();
    s.classes               = registerContentTestClasses;
    s.renderAssets          = contentTestAssets;
    s.sharedContent         = cache ? cache : std::make_shared<ContentCache>();
    const bool v2           = maxVersion >= 2;
    switch (variant)
    {
        case Variant::Committed:
            s.generators = v2 ? registerContentTestGenerators<2, Variant::Committed>
                              : registerContentTestGenerators<1, Variant::Committed>;
            break;
        case Variant::PlanChangeV2:
            s.generators = v2 ? registerContentTestGenerators<2, Variant::PlanChangeV2>
                              : registerContentTestGenerators<1, Variant::PlanChangeV2>;
            break;
        case Variant::TableChangeV2:
            s.generators = v2 ? registerContentTestGenerators<2, Variant::TableChangeV2>
                              : registerContentTestGenerators<1, Variant::TableChangeV2>;
            break;
        case Variant::ReleasedChangeV1:
            s.generators = v2 ? registerContentTestGenerators<2, Variant::ReleasedChangeV1>
                              : registerContentTestGenerators<1, Variant::ReleasedChangeV1>;
            break;
    }
    return s;
}

}  // namespace ContentTest
