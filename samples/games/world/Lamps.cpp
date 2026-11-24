/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lamps of game_world (S36b).

#include "games/world/Lamps.h"

#include "saiga/engine/components/LightSource.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/components/WorldTime.h"
#include "saiga/engine/lighting/Celestial.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/tools/ScenarioWorld.h"

#include "games/common/GameMath.h"
#include "games/common/Mesh.h"
#include "games/common/RenderAssets.h"

#include <cmath>

namespace Saiga
{
namespace Game
{
using namespace Engine;
using namespace MeshBuilder;

namespace
{
constexpr const char* kPostMesh   = "world/lamp_post";
constexpr const char* kHeadMesh   = "world/lamp_head";
constexpr const char* kHeadOff    = "world/lamp_head_off";
constexpr const char* kStreetMesh = "world/street_lamp";
constexpr const char* kTorchMesh  = "world/torch";
constexpr const char* kFlameMesh  = "world/torch_flame";
constexpr const char* kFlameOff   = "world/torch_flame_off";
// The sun elevation below which the lamps are on (degrees)
constexpr double kLampsOnBelowDeg = 4;

MeshData moved(const MeshData& m, const Vec3& t)
{
    MeshData out;
    out.append(m, transformMatrix(t, Quat::Identity()));
    return out;
}

// An entity drawn with a mesh at a position
Entity spawnMesh(World& w, const std::string& mesh, const Vec3& p)
{
    const Entity e = w.create();
    w.get<Transform>(e).setPosition(p);
    w.add<RenderInstance>(e, RenderInstance(mesh));
    return e;
}
}  // namespace

void lampRenderAssets(World& w)
{
    const Vulkan::MaterialId flat = w.registerMaterial("world/flat", demoMaterial());
    // The heads and flames glow while their light is on (lampPolicySystem switches the mesh; the lamps' light is the
    // LightSource); the unlit variants with the flat material
    Vulkan::MaterialData glow       = demoMaterial();
    glow.emissive                   = vec3(0.9f, 0.7f, 0.4f);
    const Vulkan::MaterialId glowId = w.registerMaterial("world/glow", glow);
    const Vec3 metal(0.12, 0.12, 0.13), wood(0.35, 0.22, 0.1);
    // A post with an arm and a lantern hanging from it (the light below the lantern: outside every mesh, so the
    // shadowed lamps' cube faces do not start inside the post)
    registerDemoMesh(
        w, kPostMesh,
        [metal]()
        {
            MeshData out = moved(cylinder(0.07, 1.9, 8, metal), Vec3(0, 1.9, 0));
            out.append(box(Vec3(0.3, 0.035, 0.035), metal, Vec3(0.25, 3.78, 0)));
            out.append(box(Vec3(0.02, 0.06, 0.02), metal, Vec3(0.5, 3.7, 0)));
            return out;
        },
        flat, 600);
    registerDemoMesh(w, kHeadMesh, []() { return box(Vec3(0.1, 0.12, 0.1), Vec3(1, 0.9, 0.7)); }, glowId, 600);
    registerDemoMesh(w, kHeadOff, []() { return box(Vec3(0.1, 0.12, 0.1), Vec3(0.55, 0.52, 0.45)); }, flat, 600);
    registerDemoMesh(
        w, kStreetMesh,
        [metal]()
        {
            MeshData out = moved(cylinder(0.09, 3.0, 8, metal), Vec3(0, 3.0, 0));
            // The arm towards +x and the housing at its end (the light below it)
            out.append(box(Vec3(0.8, 0.05, 0.05), metal, Vec3(0.75, 5.95, 0)));
            out.append(box(Vec3(0.3, 0.08, 0.18), metal, Vec3(1.4, 5.92, 0)));
            return out;
        },
        flat, 600);
    registerDemoMesh(
        w, kTorchMesh, [wood]() { return moved(cylinder(0.05, 0.75, 6, wood), Vec3(0, 0.75, 0)); }, flat, 400);
    registerDemoMesh(w, kFlameMesh, []() { return box(Vec3(0.08, 0.12, 0.08), Vec3(1, 0.55, 0.2)); }, glowId, 400);
    registerDemoMesh(w, kFlameOff, []() { return box(Vec3(0.06, 0.05, 0.06), Vec3(0.12, 0.1, 0.09)); }, flat, 400);
}

int spawnLamps(World& w, int count, double centreX, double centreZ, double spacing, double ringRadius,
               const std::function<double(double, double)>& ground, double seaLevel)
{
    // A square grid around the centre, row by row, or a ring (deterministic)
    const int side  = int(std::ceil(std::sqrt(double(std::max(count, 0)))));
    const int slots = ringRadius > 0 ? count : side * side;
    int spawned     = 0;
    for (int k = 0; k < slots && spawned < count; ++k)
    {
        double x = 0, z = 0;
        if (ringRadius > 0)
        {
            const double a = 6.283185307179586 * double(k) / double(count);
            x              = centreX + ringRadius * std::cos(a);
            z              = centreZ + ringRadius * std::sin(a);
        }
        else
        {
            x = centreX + (double(k % side) - 0.5 * double(side - 1)) * spacing;
            z = centreZ + (double(k / side) - 0.5 * double(side - 1)) * spacing;
        }
        const double y = ground(x, z);
        if (!(y > seaLevel + 0.5)) continue;
        const Vec3 p(x, y, z);
        LightSource l;
        if (spawned % 20 == 10)
        {
            // A torch: short, orange, flickering (the flame carries the light)
            spawnMesh(w, kTorchMesh, p);
            const Entity e     = spawnMesh(w, kFlameMesh, p + Vec3(0, 1.6, 0));
            l.color            = Vec3(1.0, 0.55, 0.22);
            l.lumen            = 400;
            l.range            = 9;
            l.offset           = Vec3(0, 0.15, 0);
            l.flickerAmplitude = 0.35;
            l.flickerFrequency = 6;
            w.add<LightSource>(e, l);
            w.add<LampPolicy>(e);
        }
        else if (spawned % 20 == 5)
        {
            // A street lamp: a spot pointing down from the end of its arm
            const Entity e  = spawnMesh(w, kStreetMesh, p);
            l.kind          = LightKind::Spot;
            l.color         = Vec3(1.0, 0.85, 0.65);
            l.lumen         = 3000;
            l.range         = 16;
            l.innerAngleDeg = 25;
            l.outerAngleDeg = 35;
            l.offset        = Vec3(1.4, 5.8, 0);
            // -z rotated by -90 degrees about x points down
            l.rotation    = axisAngle(Vec3(1, 0, 0), -1.5707963267948966);
            l.castShadows = spawned % 40 == 5;
            w.add<LightSource>(e, l);
            w.add<LampPolicy>(e);
        }
        else
        {
            // A lamp post: a warm point light under its head (the head carries the light); every 8th shadowed
            spawnMesh(w, kPostMesh, p);
            const Entity e = spawnMesh(w, kHeadMesh, p + Vec3(0.5, 3.52, 0));
            l.color        = Vec3(1.0, 0.82, 0.6);
            l.lumen        = 1200;
            l.range        = 18;
            l.offset       = Vec3(0, -0.22, 0);
            l.castShadows  = spawned % 8 == 0;
            w.add<LightSource>(e, l);
            w.add<LampPolicy>(e);
        }
        spawned++;
    }
    return spawned;
}

void lampComponents(Registry& r)
{
    r.registerComponent<LampPolicy>(Replicate::Never);
}

void lampPolicySystem(World& w, double)
{
    if (!hasWorldTime(w)) return;
    const WorldTime& t = w.readGlobal<WorldTime>();
    const bool on =
        celestialState(w.environmentSetup().celestial, t.timeOfDay, t.day).sunElevationDeg < kLampsOnBelowDeg;
    // Only the policy's lamps, only on a transition (a hand switch holds until the next one), and only the lights whose
    // state changes are written (the reliable replication sends the changes)
    const int8_t state = on ? 1 : 0;
    std::vector<Entity> change;
    w.view<const LampPolicy>().each(
        [&](Entity e, const LampPolicy& p)
        {
            if (p.applied != state) change.push_back(e);
        });
    for (Entity e : change)
    {
        w.get<LampPolicy>(e).applied = state;
        const LightSource* l         = w.tryRead<LightSource>(e);
        if (l && l->on != on) w.get<LightSource>(e).on = on;
    }
    // The heads and flames glow while their light is on (also a light switched by hand, e.g. in the Lighting tab)
    w.view<const LightSource, const RenderInstance>().each(
        [&](Entity e, const LightSource& l, const RenderInstance& r)
        {
            const bool head = r.mesh == kHeadMesh || r.mesh == kHeadOff;
            const bool fire = r.mesh == kFlameMesh || r.mesh == kFlameOff;
            if (!head && !fire) return;
            const char* want = head ? (l.on ? kHeadMesh : kHeadOff) : (l.on ? kFlameMesh : kFlameOff);
            if (r.mesh != want) w.setRenderMesh(e, want);
        });
}

void lampScenarios(ScenarioList& list)
{
    // Regression (round 4, minor 1): the policy rewrote every LightSource's `on` each step, so a lamp switched by hand
    // and a light added by hand were switched back in the next step. Now a hand switch holds until the next dusk or
    // dawn, and only the policy's lamps are switched.
    list.add(
        "lamp_hand_switch_holds",
        [](ScenarioWorld& s)
        {
            // s.world(), not a reference: the SaveLoad variant replaces the world
            std::string error;
            auto setHours = [&](double hours)
            {
                WorldTimeChange change;
                change.hours = hours;
                change.rate  = 0.0;
                s.check(setWorldTime(s.world(), change, error), "setWorldTime: " + error);
            };
            setHours(0);
            const int spawned = spawnLamps(s.world(), 1, 100, 100, 25, 0, [](double, double) { return 10.0; }, 0);
            s.check(spawned == 1, "spawns a lamp");
            Entity lamp;
            s.world().view<const LampPolicy>().each([&](Entity e, const LampPolicy&) { lamp = e; });
            s.check(s.world().tryRead<LightSource>(lamp) != nullptr, "the lamp carries a LightSource");
            if (!s.world().tryRead<LightSource>(lamp)) return;
            // A light added by hand at night, off: never switched
            const Entity manual = s.world().create();
            LightSource off;
            off.on = false;
            s.world().add<LightSource>(manual, off);
            s.step(2);
            s.check(s.world().read<LightSource>(lamp).on, "the lamp is on at night");
            s.check(!s.world().read<LightSource>(manual).on, "a light added by hand stays off");
            // Switched off by hand: holds through the night
            s.world().get<LightSource>(lamp).on = false;
            s.step(10);
            s.check(!s.world().read<LightSource>(lamp).on, "a lamp switched off by hand stays off");
            // Dawn and dusk switch it again
            setHours(12);
            s.step(2);
            s.check(!s.world().read<LightSource>(lamp).on, "off by day");
            s.world().get<LightSource>(lamp).on = true;
            s.step(10);
            s.check(s.world().read<LightSource>(lamp).on, "a lamp switched on by hand by day stays on");
            setHours(0);
            s.step(2);
            s.check(s.world().read<LightSource>(lamp).on, "on at the next dusk");
            setHours(12);
            s.step(2);
            s.check(!s.world().read<LightSource>(lamp).on, "off at the next dawn");
            s.check(!s.world().read<LightSource>(manual).on, "the light added by hand was never switched");
        },
        true);
}

}  // namespace Game
}  // namespace Saiga
