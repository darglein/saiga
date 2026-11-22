/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The flight simulator demo in a World: components, the flight model system, Aircraft and Autopilot, scenarios.

#include "games/flight/FlightWorld.h"

#include "saiga/engine/components/Health.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/content/ContentHashFile.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/content/GeneratorList.h"
#include "saiga/engine/content/Prefab.h"
#include "saiga/engine/content/gen/FixedMath.h"
#include "saiga/engine/sync/RecordingRenderSink.h"
#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioWorld.h"
#include "saiga/saiga_buildconfig.h"
#include "saiga/vulkan/world/FoliageGenerator.h"
#include "saiga/vulkan/world/PrototypeForest.h"
#include "saiga/vulkan/world/PrototypeTerrain.h"
#include "saiga/vulkan/world/WorldSettings.h"

#include "games/common/RenderAssets.h"
#include "games/flight/content/OverworldGenerator.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

namespace Saiga
{
namespace Game
{
using namespace Engine;

// The generator's constants mirror S20's prototype water level (the water plane of PrototypeScene)
static_assert(kFlightSeaLevel == Vulkan::kPrototypeSeaLevel, "the flight sea level is the prototype's water level");

std::shared_ptr<const FlightContext> FlightContext::create(uint64_t seed)
{
    auto c  = std::make_shared<FlightContext>();
    c->seed = seed;
    // The generator's circuit (the land under it): the centre of the space, S20's radius 0.3 * 8192 m
    c->circuitCenter = Vec3(double(kFlightCircuitCentre), 0, double(kFlightCircuitCentre));
    c->circuitRadius = 0.3 * Vulkan::kPrototypeArea;
    return c;
}

double flightGround(const World& world, double x, double z)
{
    return std::max(world.terrainHeight(0, x, z), double(kFlightSeaLevel) * kHeightUnit);
}

FlightModel flightModelOf(const World& w, Entity e)
{
    const AircraftState& s = w.read<AircraftState>(e);
    const Transform& t     = w.read<Transform>(e);
    FlightModel m;
    m.params            = w.data(s.model);
    const World* world  = &w;
    m.ground            = [world](double x, double z) { return flightGround(*world, x, z); };
    m.state.position    = t.position();
    m.state.orientation = t.rotation();
    m.state.velocity    = s.velocity;
    m.state.bodyRates   = s.bodyRates;
    m.state.onGround    = s.onGround;
    m.state.crashed     = s.crashed;
    return m;
}

namespace
{
// Phase Gameplay (host): the flight model of every airplane (after the autopilot wrote its controls in phase
// Behaviour), with the controls of its seat's driver (a client's driver: the latest received; no live driver: neutral);
// the driver's throttleChange moves the lever
void flightModelSystem(World& w, double dt)
{
    w.view<Transform, AircraftState>().each(
        [&](Entity e, Transform& t, AircraftState& s)
        {
            if (s.crashed)
            {
                s.crashTimer += dt;
                return;
            }
            const AircraftControls* driven = seatControls<AircraftControls>(w, e);
            const AircraftControls c       = driven ? *driven : AircraftControls();
            s.throttle                     = clampd(s.throttle + c.throttleChange * 0.5 * dt, 0, 1);
            FlightModel m                  = flightModelOf(w, e);
            m.step(c, s.throttle, dt);
            t.setPosition(m.state.position);
            t.setRotation(m.state.orientation);
            s.velocity  = m.state.velocity;
            s.bodyRates = m.state.bodyRates;
            s.onGround  = m.state.onGround;
            s.crashed   = m.state.crashed;
        });
}

// Phase Gameplay on clients (--coop --demo): the autopilot flies the owned drivers with DemoPilot from the replicated
// plane (AircraftState with the lever, the newest snapshot pose); the lever follows through throttleChange
void flightDemoPilotSystem(World& w, double)
{
    const FlightContext& ctx = w.context<FlightContext>();
    FlightAutopilot pilot;
    pilot.center = ctx.circuitCenter;
    pilot.radius = ctx.circuitRadius;
    w.ownedBy<AircraftControls, const DemoPilot, const Driver>(w.localPeer())
        .each(
            [&](Entity, AircraftControls& c, const DemoPilot& demo, const Driver& d)
            {
                if (demo.suspended || !d.vehicle.valid() || !w.alive(d.vehicle) || !w.has<AircraftState>(d.vehicle))
                    return;
                const AircraftState& s = w.read<AircraftState>(d.vehicle);
                if (s.crashed)
                {
                    c = AircraftControls();
                    return;
                }
                const FlightModel m = flightModelOf(w, d.vehicle);
                c                   = pilot.controls(m);
                c.throttleChange    = clampd((pilot.throttle(m) - s.throttle) * 4.0, -1, 1);
            });
}
}  // namespace

void AircraftBase::onSpawn()
{
    if (!has<AircraftState>()) add<AircraftState>();
    // Drawn by the engine (S19b)
    if (!has<RenderInstance>()) add<RenderInstance>(RenderInstance(kAirplaneMesh));
    if (!has<VehicleSeat>())
    {
        // Its own driver: the autopilot, or a plane spawned as the player
        add<VehicleSeat>(VehicleSeat{entity(), 0});
        add<Driver>(Driver{entity()});
        if (!has<AircraftControls>()) add<AircraftControls>();
    }
    respawn();
}

void AircraftBase::update(double)
{
    // The action's player is the driver (the autopilot and a plane spawned as the player drive themselves)
    bool requested = false;
    for (const ActionEvent& a : world().events.read<ActionEvent>())
        if (drives(world(), a.player, entity()) && a.action == uint32_t(FlightAction::Respawn)) requested = true;
    if (requested || (read<AircraftState>().crashed && read<AircraftState>().crashTimer > 2.0)) respawn();
}

void AircraftBase::respawn()
{
    const FlightContext& ctx = world().context<FlightContext>();
    // The slot's place: 60 m behind and 20 m inside per slot (heading +z on the circuit's +x side)
    const int slot = has<VehicleSeat>() ? std::max(0, read<VehicleSeat>().slot) : 0;
    Vec3 p         = ctx.circuitCenter + Vec3(ctx.circuitRadius - 20.0 * slot, 0, -60.0 * slot);
    p(1)           = flightGround(world(), p(0), p(2)) + 300;
    FlightModel m;
    m.reset(p, headingOf(Vec3(0, 0, 1)), 65);
    AircraftState& s = write<AircraftState>();
    s.velocity       = m.state.velocity;
    s.bodyRates      = m.state.bodyRates;
    s.onGround       = m.state.onGround;
    s.crashed        = false;
    s.crashTimer     = 0;
    transformWrite().teleport(m.state.position, m.state.orientation);
}

void Autopilot::configure()
{
    const FlightContext& ctx = world().context<FlightContext>();
    pilot.center             = ctx.circuitCenter;
    pilot.radius             = ctx.circuitRadius;
}

void Autopilot::update(double dt)
{
    AircraftBase::update(dt);
    if (read<AircraftState>().crashed) return;
    // Its own driver: the sticks on itself, the lever directly
    const FlightModel m             = flightModelOf(world(), entity());
    write<AircraftControls>()       = pilot.controls(m);
    write<AircraftState>().throttle = pilot.throttle(m);
}

void FlightChest::onSpawn()
{
    if (!has<Health>()) add<Health>();
    if (!has<RenderInstance>()) add<RenderInstance>(RenderInstance(kChestMesh));
}

namespace
{
// The tree meshes of the generator's spruce and broadleaf slots (species 0 and 1 of treeSpecies(seed)) for the
// PrototypeFoliage class: built once per process and seed and shared by every world (renderAssets runs per world, the
// scenarios build several worlds)
std::array<std::shared_ptr<const Vulkan::MeshLods>, 2> treeMeshes(uint32_t seed)
{
    static std::mutex mutex;
    static std::map<uint32_t, std::array<std::shared_ptr<const Vulkan::MeshLods>, 2>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto& entry = cache[seed];
    if (!entry[0])
    {
        const std::vector<Vulkan::TreeSpecies> species = Vulkan::FoliageGenerator::treeSpecies(seed);
        const Vulkan::WorldSettings settings;
        for (size_t i = 0; i < entry.size(); ++i)
            entry[i] = std::make_shared<const Vulkan::MeshLods>(Vulkan::prototypeTreeMesh(
                species[i], Vulkan::prototypeTreeLodEnds(settings), Vulkan::prototypeTreePadding(settings)));
    }
    return entry;
}

// The meshes of the flight demo (WorldSetup::renderAssets): the airplane, the generator's slot meshes and the ruin's
void flightRenderAssets(World& w)
{
    const Vulkan::MaterialId flat = w.registerMaterial("flight/flat", demoMaterial());
    registerDemoMesh(w, kAirplaneMesh, []() { return MeshBuilder::airplane(); }, flat);
    // S23c: the slot meshes (OverworldGenerator::slots) and the ruin prefab's (data/flight/prefabs/ruin.json)
    Vulkan::MaterialData foliage;
    foliage.pipelineClass              = uint8_t(Vulkan::PipelineClass::PrototypeFoliage);
    foliage.twoSided                   = true;
    const Vulkan::MaterialId foliageId = w.registerMaterial("flight/foliage", foliage);
    const auto trees                   = treeMeshes(kFlightFoliageSeed);
    w.registerMesh("flight/spruce", trees[0], foliageId);
    w.registerMesh("flight/broadleaf", trees[1], foliageId);
    const Vec3 rock(0.38, 0.36, 0.33), stone(0.55, 0.5, 0.42), wood(0.4, 0.25, 0.12);
    registerDemoMesh(w, "flight/rock", [rock]() { return MeshBuilder::rock(1, rock); }, flat, 1200);
    registerDemoMesh(w, "flight/ruin_pillar", [stone]() { return MeshBuilder::cylinder(1.5, 6, 12, stone); }, flat);
    registerDemoMesh(w, "flight/ruin_wall", [stone]() { return MeshBuilder::box(Vec3(8, 3, 1), stone); }, flat);
    registerDemoMesh(
        w, kChestMesh, [wood]() { return MeshBuilder::box(Vec3(0.6, 0.4, 0.4), wood, Vec3(0, 0.4, 0)); }, flat);
}
}  // namespace

void flightPeerJoined(World& host, uint8_t peer, bool demo)
{
    const Driven d = spawnDriven<Aircraft>(host, peer, peer);
    if (demo) host.add<DemoPilot>(d.driver);
}

SpaceTemplate flightSpaceTemplate()
{
    SpaceTemplate t;
    t.name        = "flight_overworld";
    t.kind        = SpaceKind::OpenWorld;
    t.baseSpacing = 1;
    t.seaLevel    = kFlightSeaLevel;
    t.sizeChunks  = 128;
    t.generator   = kFlightOverworldGenerator;
    // The trees, rocks and ruins up to 3 km (the content sets; the 32 km world needs more than the 1 km default)
    t.fullRadius = 3000;
    return t;
}

namespace
{
void registerFlightGenerators(GeneratorList& list)
{
    list.add(kFlightOverworldGenerator, std::make_shared<OverworldGenerator>());
}
}  // namespace

WorldSetup flightWorldSetup(std::shared_ptr<const FlightContext> context, const std::string& dataDirectory)
{
    WorldSetup s;
    // S23c: the 32 km overworld of the OverworldGenerator (the prototype's hook is no longer used by flight)
    s.seed                  = context->seed;
    s.config.spaceTemplates = {flightSpaceTemplate()};
    s.generators            = registerFlightGenerators;
    s.components            = [](Registry& r)
    {
        r.registerComponent<AircraftControls>(Replicate::OwnerAuthoritative);
        r.registerComponent<AircraftState>(Replicate::Host);
        registerDrivingComponents(r);
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "flightDemoPilot", flightDemoPilotSystem, RunOn::Client);
        l.add(Phase::Gameplay, "flightModel", flightModelSystem);
        l.add(Phase::Gameplay, "driverPose", driverPoseSystem, RunOn::Both);
    };
    s.classes = [](ClassList& c)
    {
        c.add<Aircraft>();
        c.add<Autopilot>();
        c.add<FlightChest>();
    };
    s.actions       = [](ActionList& a) { a.add(uint32_t(FlightAction::Respawn), "respawn"); };
    s.dataTypes     = [](DataTypeList& d) { d.add<AircraftParameters>(); };
    s.dataDirectory = dataDirectory;
    s.renderAssets  = flightRenderAssets;
    s.setContext(std::move(context));
    return s;
}

// ---- scenarios (--scenarios; docs/engine_design.md §17.2) ----------------------------------------------------------

namespace
{
// The entities that render a mesh
std::vector<Entity> withMesh(World& world, const std::string& mesh)
{
    std::vector<Entity> out;
    world.viewAll<const RenderInstance>().each(
        [&](Entity e, const RenderInstance& r)
        {
            if (r.mesh == mesh) out.push_back(e);
        });
    return out;
}

// S23c: the fixed ruin of the plan (site key 1)
const Site* fixedRuin(const World& world)
{
    for (const Site& s : world.generation().space(0).plan->sites)
        if (s.key == kFlightFixedRuinKey) return &s;
    return nullptr;
}

// S23c: the checks of flight_ruin_is_stamped; the first that does not hold yet (empty: all hold)
std::string ruinStampedOpen(ScenarioWorld& w, const Site& ruin)
{
    const Vec3 expected(double(ruin.x), double(ruin.baseHeight) / 1024.0, double(ruin.z));
    const std::vector<Entity> chests = withMesh(w.world(), kChestMesh);
    if (chests.size() != 1) return "the host has one chest (" + std::to_string(chests.size()) + ")";
    if (w.world().read<Transform>(chests[0]).position() != expected) return "the chest at the prefab position (exact)";
    if (!w.world().has<Health>(chests[0]) || w.world().read<Health>(chests[0]).hp != 40)
        return "the chest with the prefab's Health value";
    for (int p = 1; p <= w.clientCount(); ++p)
        if (withMesh(w.client(p), kChestMesh).size() != 1) return "client " + std::to_string(p) + " has one chest";
    const RegionId region{0, 0, int32_t(floorDiv(ruin.x, 256)), int32_t(floorDiv(ruin.z, 256))};
    for (int p = 0; p <= w.clientCount(); ++p)
    {
        const std::string who = p == 0 ? std::string("the host") : "client " + std::to_string(p);
        if (w.terrainHeight(expected.x(), expected.z(), p == 0 ? std::optional<int>() : std::optional<int>(p)) !=
            expected.y())
            return who + ": terrainHeight at the patch centre is the base height";
        const RecordingRenderSink::InstanceSet* set = w.renderSink(p).instanceSetAt(region);
        if (!set) return who + ": the set of the ruin's chunk";
        // The tier-1 instances, then the 12 static parts (slot 255) in staticList order
        const size_t n = set->instances.size();
        if (n < 12) return who + ": the set has the 12 static parts";
        World& world = p == 0 ? w.world() : w.client(p);
        for (size_t i = 0; i < n; ++i)
        {
            const uint32_t slot = set->instances[i].key >> 24, counter = set->instances[i].key & 0xFFFFFFu;
            if (i < n - 12 && slot == 255) return who + ": a static part before the tier-1 instances";
            if (i >= n - 12)
            {
                const uint32_t part = uint32_t(i - (n - 12));
                if (slot != 255 || counter != part) return who + ": static part " + std::to_string(part) + " in order";
                const char* mesh = part < 8 ? "flight/ruin_pillar" : "flight/ruin_wall";
                if (set->instances[i].mesh != world.findMesh(mesh))
                    return who + ": static part " + std::to_string(part) + " draws " + mesh;
            }
        }
    }
    return std::string();
}
}  // namespace

void flightScenarios(ScenarioList& list)
{
    // The autopilot's positions after 600 and 1800 steps (S13b; re-recorded in S23c over the OverworldGenerator's
    // terrain: before (6506.433, 402.839, 4752.384) and (5858.335, 362.351, 5875.344) over the prototype's; again after
    // the 32 km layout: before (6505.742, 325.853, 4741.635) and (5879.619, 330.511, 5857.057) over the land disc; the
    // circuit moved to the centre of the space: before (6505.847, 351.931, 4741.826) and (5878.704, 352.505, 5858.160))
    list.add("autopilot_flies_the_circuit",
             [](ScenarioWorld& w)
             {
                 const Entity plane = w.spawn<Autopilot>(Vec3::Zero());
                 w.trace<AircraftState>(plane);
                 auto near = [&](const Vec3& expected, const char* when)
                 {
                     const Vec3 p = w.world().read<Transform>(plane).position();
                     w.check((p - expected).norm() < 1.0, std::string(when) + ": at " + std::to_string(p(0)) + " " +
                                                              std::to_string(p(1)) + " " + std::to_string(p(2)));
                 };
                 w.step(600);
                 near(Vec3(18794.478, 369.778, 17027.615), "step 600");
                 w.step(1200);
                 near(Vec3(18163.513, 351.898, 18147.840), "step 1800");
                 const double altitude = flightModelOf(w.world(), plane).altitudeAboveGround();
                 w.check(std::abs(altitude - 300) < 30, "holds 300 m above the ground: " + std::to_string(altitude));
                 w.check(!w.world().read<AircraftState>(plane).crashed, "did not crash");
             });
    // S16b: a driver of the default peer (a client's in the host + client variants): the lever on the host-owned plane,
    // the sticks written on the driver's owner; stepUntil (the sticks arrive up to 8 steps late)
    list.add("climbs_with_full_throttle",
             [](ScenarioWorld& w)
             {
                 const Driven d                               = spawnDriven<Aircraft>(w.world(), w.defaultPeer(), 0);
                 const Entity plane                           = d.vehicle;
                 const double y0                              = w.world().read<Transform>(plane).position().y();
                 w.world().get<AircraftState>(plane).throttle = 1.0;
                 w.edit(d.driver, [](World& owner, Entity e) { owner.get<AircraftControls>(e).pitch = 0.25; });
                 w.trace<AircraftState>(plane);
                 w.check(
                     w.stepUntil([&] { return w.world().read<Transform>(plane).position().y() > y0 + 60; }, 60 * 10),
                     "climbed " + std::to_string(w.world().read<Transform>(plane).position().y() - y0) + " m");
                 w.check(!w.world().read<AircraftState>(plane).crashed, "did not crash");
             });
    list.add(
        "crashes_into_terrain",
        [](ScenarioWorld& w)
        {
            const Driven d                               = spawnDriven<Aircraft>(w.world(), w.defaultPeer(), 0);
            const Entity plane                           = d.vehicle;
            w.world().get<AircraftState>(plane).throttle = 1.0;
            w.edit(d.driver, [](World& owner, Entity e) { owner.get<AircraftControls>(e).pitch = -1.0; });
            w.trace<AircraftState>(plane);
            w.check(w.stepUntil([&] { return w.world().read<AircraftState>(plane).crashed; }, 60 * 30),
                    "crashes when diving");
            // The crash lasts 2 s, then the plane is back on the circuit (host-side)
            w.stepSeconds(2.5);
            w.check(!w.world().read<AircraftState>(plane).crashed, "respawned");
            const double altitude = flightModelOf(w.world(), plane).altitudeAboveGround();
            w.check(altitude > 250, "300 m above the ground after the respawn: " + std::to_string(altitude));
            // The respawn action of the driver reaches its plane (driver -> Driver.vehicle): level sticks, fly
            // away from the respawn point, then the action brings it back
            w.edit(d.driver, [](World& owner, Entity e) { owner.get<AircraftControls>(e) = AircraftControls(); });
            const FlightContext& ctx = w.world().context<FlightContext>();
            Vec3 spawnPoint          = ctx.circuitCenter + Vec3(ctx.circuitRadius, 0, 0);
            spawnPoint(1)            = flightGround(w.world(), spawnPoint(0), spawnPoint(2)) + 300;
            w.check(w.stepUntil([&] { return (w.world().read<Transform>(plane).position() - spawnPoint).norm() > 100; },
                                60 * 5),
                    "flies away from the respawn point");
            w.input(d.driver, uint32_t(FlightAction::Respawn));
            w.check(
                w.stepUntil([&] { return (w.world().read<Transform>(plane).position() - spawnPoint).norm() < 5; }, 60),
                "the driver's respawn action respawns its plane");
        });
    // S23c: a driven plane (its driver is the Player) held 200 m above the fixed ruin: the host stamps the ruin's
    // chest once (exact position, the prefab's Health), clients get it replicated, every peer has the patch's height
    // at the centre and the ruin chunk's set with the tier-1 instances and the 12 static parts. The plane is
    // teleported before every step (level, no velocity), so neither the circuit respawn nor the cruise moves the driver
    // off the ruin's chunk (the scenario worlds stamp within 128 m)
    list.add("flight_ruin_is_stamped",
             [](ScenarioWorld& w)
             {
                 const Site* site = fixedRuin(w.world());
                 if (!w.check(site != nullptr, "the plan has the fixed ruin")) return;
                 const Site ruin    = *site;
                 const Driven d     = spawnDriven<Aircraft>(w.world(), w.defaultPeer(), 0);
                 const Entity plane = d.vehicle;
                 const Vec3 hold(double(ruin.x), double(ruin.baseHeight) / 1024.0 + 200, double(ruin.z));
                 std::string open = "not checked";
                 for (int i = 0; i <= 300; ++i)
                 {
                     World& host = w.world();
                     host.get<Transform>(plane).teleport(hold, Quat::Identity());
                     AircraftState& s = host.get<AircraftState>(plane);
                     s.velocity       = Vec3::Zero();
                     s.bodyRates      = Vec3::Zero();
                     open             = ruinStampedOpen(w, ruin);
                     if (open.empty() || i == 300) break;
                     w.step(1);
                 }
                 w.check(open.empty(), "after 300 steps: " + open);
             });
    // S23c: every slot mesh of the generator and every mesh of the ruin prefab (static parts, RenderInstance
    // components, the chest class's mesh) is registered
    list.add(
        "flight_slot_meshes_resolve",
        [](ScenarioWorld& w)
        {
            const World& world = w.world();
            auto resolves      = [&](const std::string& mesh, const std::string& what)
            { w.check(world.findMesh(mesh) != 0, what + ": the mesh \"" + mesh + "\" is registered"); };
            const GeneratorList::Entry* g = world.generators().find(kFlightOverworldGenerator);
            if (!w.check(g != nullptr, "the overworld generator is registered")) return;
            for (const GeneratorSlot& slot : g->slots)
                for (const std::string& mesh : slot.meshes) resolves(mesh, "slot " + slot.name);
            Prefab ruin;
            std::string error;
            if (!w.check(readPrefab(w.setup().dataDirectory, kFlightRuinPrefab, 1, ruin, error), error)) return;
            w.check(!ruin.statics.empty() && !ruin.entities.empty(), "the ruin has static parts and entities");
            for (const PrefabStatic& part : ruin.statics) resolves(part.mesh, "ruin static part");
            for (const PrefabEntity& e : ruin.entities)
            {
                const json11::Json& r = e.components["RenderInstance"];
                if (r.is_object()) resolves(r["mesh"].string_value(), "ruin entity " + e.className);
                if (e.className == FlightChest::name) resolves(kChestMesh, "the FlightChest class");
            }
        },
        true);
    // S23c: the game's committed content hashes (data/flight/content_hashes.json; phase A of the pipeline on a
    // JobPool(3)). SAIGA_UPDATE_CONTENT_HASHES=1 rewrites the file in the source tree (it needs
    // SAIGA_WRITE_PROJECT_DIR_TO_HEADER) and fails the run
    list.add(
        "flight_content_hashes",
        [](ScenarioWorld& w)
        {
            const char* updateEnv  = std::getenv("SAIGA_UPDATE_CONTENT_HASHES");
            const bool update      = updateEnv && std::string(updateEnv) == "1";
            std::string file       = w.setup().dataDirectory + "/content_hashes.json";
            const std::string root = SAIGA_PROJECT_SOURCE_DIR;
            if (update)
            {
                if (!w.check(!root.empty(),
                             "SAIGA_UPDATE_CONTENT_HASHES needs SAIGA_WRITE_PROJECT_DIR_TO_HEADER=ON "
                             "(the file is rewritten in the source tree only)"))
                    return;
                file = root + "/data/flight/content_hashes.json";
            }
            const ContentHashReport report = checkContentHashes(w.setup(), file, update);
            std::cout << "flight_content_hashes: " << report.summary() << std::endl;
            w.check(report.ok, report.summary());
        },
        true);
}

}  // namespace Game
}  // namespace Saiga
