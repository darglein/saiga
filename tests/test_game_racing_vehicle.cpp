/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "games/racing/LapTimer.h"
#include "games/racing/RaceDriver.h"
#include "games/racing/Track.h"
#include "games/racing/Vehicle.h"
#include "gtest/gtest.h"

using namespace Saiga;
using namespace Saiga::Game;

namespace
{
constexpr double kDt = 1.0 / 60.0;

Vehicle carOnFlatGround(double y = 0)
{
    Vehicle v;
    v.ground = [y](double, double) { return y; };
    v.reset(Vec3(0, NAN, 0), 0);
    return v;
}

void drive(Vehicle& v, const VehicleControls& c, double seconds)
{
    for (int i = 0; i < int(seconds / kDt + 0.5); ++i) v.step(c, kDt);
}
}  // namespace

TEST(RacingVehicle, SettlesOnTheGround)
{
    Vehicle v = carOnFlatGround(10);
    drive(v, VehicleControls(), 5.0);
    EXPECT_EQ(v.wheelsOnGround(), 4);
    // Center of mass about rest length + radius - static compression - mount offset above the ground
    EXPECT_GT(v.state.position(1), 10.4);
    EXPECT_LT(v.state.position(1), 11.0);
    EXPECT_LT(v.state.velocity.norm(), 0.05);
    EXPECT_GT(v.up()(1), 0.999);
}

TEST(RacingVehicle, AcceleratesAndBrakes)
{
    Vehicle v = carOnFlatGround();
    drive(v, VehicleControls(), 1.0);
    VehicleControls gas;
    gas.throttle = 1;
    drive(v, gas, 5.0);
    double speed = v.forwardSpeed();
    EXPECT_GT(speed, 15);
    EXPECT_EQ(v.wheelsOnGround(), 4);
    // Heading 0 drives along -z
    EXPECT_LT(v.state.position(2), -30);
    EXPECT_NEAR(v.state.position(0), 0, 0.5);

    VehicleControls brake;
    brake.throttle = -1;
    drive(v, brake, 3.0);
    EXPECT_LT(v.forwardSpeed(), 1.5);
}

TEST(RacingVehicle, SteersRight)
{
    Vehicle v = carOnFlatGround();
    VehicleControls c;
    c.throttle = 0.6;
    drive(v, c, 3.0);
    c.steer = 1;
    drive(v, c, 2.0);
    // Turning right from -z: the heading decreases (yaw towards +x)
    EXPECT_LT(headingOf(v.forward()), -0.5);
    EXPECT_GT(v.state.position(0), 3);
    EXPECT_GE(v.wheelsOnGround(), 3);
}

TEST(RacingVehicle, StaysOnSlopedGround)
{
    Vehicle v;
    v.ground = [](double x, double z) { return 0.1 * x + 3 * std::sin(z / 20); };
    v.reset(Vec3(0, NAN, 0), 0);
    VehicleControls c;
    c.throttle    = 0.7;
    double maxAir = 0;
    for (int i = 0; i < 600; ++i)
    {
        v.step(c, kDt);
        double h = v.state.position(1) - v.ground(v.state.position(0), v.state.position(2));
        maxAir   = std::max(maxAir, h);
    }
    EXPECT_LT(maxAir, 1.5);
    EXPECT_GT(v.up()(1), 0.9);
}

TEST(RacingTrack, ArcLengthAndCheckpoints)
{
    Track t;
    t.build(Track::loop(Vec3(0, 0, 0), 100, 12, 3), 8);
    EXPECT_GT(t.length(), 400);
    EXPECT_EQ(t.checkpoints().size(), 8u);
    // Sampling is continuous and wraps around
    for (double s = 0; s < t.length(); s += 7)
    {
        EXPECT_LT((t.position(s + 1) - t.position(s)).norm(), 1.2);
        EXPECT_NEAR(t.tangent(s).norm(), 1, 1e-9);
    }
    EXPECT_LT((t.position(t.length()) - t.position(0)).norm(), 1e-6);
    EXPECT_NEAR(t.closest(t.position(123.4)), 123.4, 1.0);
}

TEST(RacingTrack, LapsNeedAllCheckpointsInOrder)
{
    Track t;
    t.build(Track::loop(Vec3(0, 0, 0), 100, 12, 3), 6);
    LapTimer timer;
    timer.reset(t);
    // Drive along the track for two laps
    Vec3 prev = t.position(0.5);
    for (double s = 1; s < 2 * t.length() + 1; s += 0.5)
    {
        Vec3 p = t.position(s);
        timer.update(t, prev, p, 0.05);
        prev = p;
    }
    EXPECT_EQ(timer.laps, 2);
    EXPECT_GT(timer.bestLapTime, 0);

    // A shortcut across the infield from checkpoint 1 back to the start line does not count
    timer.reset(t);
    prev = t.position(0.5);
    for (double s = 1; s < t.checkpoints()[1].s + 5; s += 0.5)
    {
        Vec3 p = t.position(s);
        timer.update(t, prev, p, 0.05);
        prev = p;
    }
    EXPECT_EQ(timer.nextCheckpoint, 2);
    Vec3 a = prev, b = t.position(t.length() - 3);
    for (int i = 1; i <= 50; ++i)
    {
        Vec3 p = a + (b - a) * (i / 50.0);
        timer.update(t, prev, p, 0.05);
        prev = p;
    }
    for (double s = t.length() - 3; s < t.length() + 3; s += 0.5)
    {
        Vec3 p = t.position(s);
        timer.update(t, prev, p, 0.05);
        prev = p;
    }
    EXPECT_EQ(timer.laps, 0);
}

TEST(RacingDriver, AiCompletesALap)
{
    Track t;
    t.build(Track::loop(Vec3(0, 0, 0), 120, 12, 5), 10);
    Vehicle v;
    v.ground   = [](double x, double z) { return 2 * std::sin(x / 50) * std::cos(z / 70); };
    Vec3 start = t.position(0);
    v.reset(Vec3(start(0), NAN, start(2)), headingOf(t.tangent(0)));
    RaceDriver driver;
    LapTimer timer;
    timer.reset(t);
    double maxOff = 0;
    for (int i = 0; i < 60 * 120 && timer.laps < 1; ++i)
    {
        Vec3 before = v.state.position;
        v.step(driver.controls(v, t), kDt);
        timer.update(t, before, v.state.position, kDt);
        Vec3 c = t.position(t.closest(v.state.position, driver.progress));
        maxOff = std::max(maxOff, Vec3(v.state.position(0) - c(0), 0, v.state.position(2) - c(2)).norm());
    }
    EXPECT_EQ(timer.laps, 1);
    EXPECT_LT(maxOff, t.width * 0.5);
}

TEST(RacingVehicle, DoesNotRollOverAtFullSteer)
{
    // Regression: the tire forces acted at the contact points, 0.8 m below the center of mass, so the car rolled over
    // at about 1 g, below the tire friction (18 m/s with half steer flipped it)
    for (double v0 : {12.0, 18.0, 25.0})
    {
        for (double steer : {0.5, 1.0})
        {
            Vehicle v = carOnFlatGround();
            drive(v, VehicleControls(), 1.0);
            v.state.velocity = v.forward() * v0;
            VehicleControls c;
            c.throttle   = 0.3;
            c.steer      = steer;
            double minUp = 1;
            for (int i = 0; i < 300; ++i)
            {
                v.step(c, kDt);
                minUp = std::min(minUp, v.up()(1));
            }
            EXPECT_GT(minUp, 0.8) << "v0 " << v0 << " steer " << steer;
        }
    }
}

TEST(RacingVehicle, SpawnsAtRestWithoutHopping)
{
    // Regression: the car was spawned with the springs 0.2 m compressed and jumped off all four wheels
    Vehicle v       = carOnFlatGround();
    double y0       = v.state.position(1);
    double maxY     = y0;
    int minContacts = 4;
    for (int i = 0; i < 120; ++i)
    {
        v.step(VehicleControls(), kDt);
        maxY        = std::max(maxY, v.state.position(1));
        minContacts = std::min(minContacts, v.wheelsOnGround());
    }
    EXPECT_NEAR(y0, v.restingHeight(), 1e-12);
    EXPECT_LT(maxY - y0, 0.01);
    EXPECT_EQ(minContacts, 4);
    EXPECT_NEAR(v.state.position(1), y0, 0.01);
}

TEST(RacingVehicle, IsNotThrownUpOnItsSide)
{
    // Regression: on its side the low wheel mounts were underground and reported full compression, which pushed the
    // car up (about 0.15 m in this setup) instead of letting it rest on its side. It must also not sink into the
    // ground: it rests at about its half width (0.9 m).
    Vehicle v           = carOnFlatGround();
    v.state.orientation = axisAngle(Vec3(0, 0, 1), 3.14159265358979 / 2);
    v.state.position(1) = 0.9;
    double maxY         = v.state.position(1);
    for (int i = 0; i < 120; ++i)
    {
        v.step(VehicleControls(), kDt);
        maxY = std::max(maxY, v.state.position(1));
    }
    EXPECT_LT(maxY, 1.0);
    EXPECT_NEAR(v.state.position(1), v.params.halfExtents(0), 0.05);
}

TEST(RacingVehicle, RespawnOnTheAsphaltWithAResetHintDoesNotHop)
{
    // Regression: RacingGame::respawn kept the track-search hint of the crash site; the search (around the hint) missed
    // the lifted asphalt and the car spawned 0.15 m too low and hopped. Track::closest now falls back to a global
    // search, so even a stale hint finds the asphalt (the game also resets its hint on respawn).
    Track t;
    t.width = 12;
    t.build(Track::loop(Vec3(0, 0, 0), 220, 12, 7), 10);
    double hint = t.length() * 0.5;  // stale: the far side of the track
    Vehicle v;
    v.ground = [&](double x, double z) { return t.surfaceOffset(Vec3(x, 0, z), hint); };
    Vec3 p   = t.position(3);
    v.reset(Vec3(p(0), NAN, p(2)), headingOf(t.tangent(3)));
    EXPECT_NEAR(v.state.position(1), t.lift + v.restingHeight(), 1e-9);
    double y0 = v.state.position(1), maxY = y0;
    for (int i = 0; i < 120; ++i)
    {
        v.step(VehicleControls(), kDt);
        maxY = std::max(maxY, v.state.position(1));
    }
    EXPECT_LT(maxY - y0, 0.01);
    EXPECT_EQ(v.wheelsOnGround(), 4);
}

TEST(RacingVehicle, ChassisSlidesToAStopOnItsRoof)
{
    // Regression: upside down the chassis slid on without friction
    Vehicle v           = carOnFlatGround();
    v.state.orientation = axisAngle(Vec3(0, 0, 1), 3.14159265358979);
    v.state.position(1) = 0.3;
    v.state.velocity    = Vec3(12, 0, 0);
    drive(v, VehicleControls(), 5.0);
    EXPECT_LT(v.state.velocity.norm(), 0.5);
}

TEST(RacingTrack, SurfaceOffsetOfTheLiftedAsphalt)
{
    Track t;
    t.width = 12;
    t.build(Track::loop(Vec3(0, 0, 0), 100, 12, 3), 8);
    double hint = -1;
    Vec3 c      = t.position(50);
    Vec3 r(-t.tangent(50)(2), 0, t.tangent(50)(0));
    EXPECT_DOUBLE_EQ(t.surfaceOffset(c, hint), t.lift);
    EXPECT_NEAR(hint, 50, 1.0);
    EXPECT_DOUBLE_EQ(t.surfaceOffset(c + r * 5.9, hint), t.lift);
    EXPECT_NEAR(t.surfaceOffset(c + r * (6 + t.curbWidth / 2), hint), t.lift / 2, 0.02);
    EXPECT_DOUBLE_EQ(t.surfaceOffset(c + r * 10, hint), 0);
}

TEST(RacingDriver, AiRacesTheDemoTrackShapeWithoutRollingOver)
{
    // The track shape of the racing demo (radius 220 m, 12 control points, seed 7, 12 m wide) on gentle hills.
    // Regression: the demo car rolled over twice per lap and left the track
    Track t;
    t.width = 12;
    t.build(Track::loop(Vec3(0, 0, 0), 220, 12, 7), 10);
    Vehicle v;
    v.ground   = [](double x, double z) { return 3 * std::sin(x / 60) * std::cos(z / 80); };
    Vec3 start = t.position(0);
    v.reset(Vec3(start(0), NAN, start(2)), headingOf(t.tangent(0)));
    RaceDriver driver;
    LapTimer timer;
    timer.reset(t);
    double maxOff = 0, minUp = 1;
    for (int i = 0; i < 60 * 240 && timer.laps < 2; ++i)
    {
        Vec3 before = v.state.position;
        v.step(driver.controls(v, t), kDt);
        timer.update(t, before, v.state.position, kDt);
        Vec3 c = t.position(t.closest(v.state.position, driver.progress));
        maxOff = std::max(maxOff, Vec3(v.state.position(0) - c(0), 0, v.state.position(2) - c(2)).norm());
        minUp  = std::min(minUp, v.up()(1));
    }
    EXPECT_EQ(timer.laps, 2);
    EXPECT_LT(maxOff, t.width * 0.5);
    EXPECT_GT(minUp, 0.8);
}

// Regression: the width check used the end point of the step, so a crossing inside the gate width whose end point was
// just outside it did not count and the checkpoint stayed pending
TEST(RacingTrack, GateCrossingIsCheckedWhereTheLineIsCrossed)
{
    Track t;
    t.width = 12;
    t.build(Track::loop(Vec3(0, 0, 0), 100, 12, 3), 6);
    LapTimer timer;
    timer.reset(t);
    const auto& g = t.checkpoints()[timer.nextCheckpoint];
    Vec3 side(-g.tangent(2), 0, g.tangent(0));
    double allowed = t.width * 0.5 + 3;
    // Crosses at lateral offset allowed - 0.005, ends at allowed + 0.02
    Vec3 before = g.center + side * (allowed - 0.03) - g.tangent * 0.05;
    Vec3 after  = g.center + side * (allowed + 0.02) + g.tangent * 0.05;
    int next    = timer.nextCheckpoint;
    timer.update(t, before, after, 0.01);
    EXPECT_EQ(timer.nextCheckpoint, (next + 1) % int(t.checkpoints().size()));
}

// Regression: identical control points gave a zero length and NaN positions (undefined int conversion)
TEST(RacingTrack, DegenerateTrackIsEmpty)
{
    Track t;
    t.build(std::vector<Vec3>(4, Vec3(5, 0, 5)), 4);
    EXPECT_EQ(t.length(), 0);
    EXPECT_TRUE(t.checkpoints().empty());
    double hint = -1;
    EXPECT_EQ(t.surfaceOffset(Vec3(5, 0, 5), hint), 0);
}

// Regression: on two parallel sections 10 m apart, a hint on the first section found its curb although the point was
// on the asphalt of the second section
TEST(RacingTrack, SurfaceOffsetNextToAParallelSection)
{
    std::vector<Vec3> control;
    for (int x = 0; x <= 200; x += 25) control.push_back(Vec3(x, 0, 0));
    for (int x = 200; x >= 0; x -= 25) control.push_back(Vec3(x, 0, 10));
    Track t;
    t.width = 10;
    t.build(control, 4);
    double s0   = t.closest(Vec3(100, 0, 0));
    double hint = s0;
    ASSERT_NEAR(t.position(s0)(2), 0, 0.1);
    // 5.5 m from the first centerline (its curb), 4.5 m from the second (its asphalt)
    EXPECT_DOUBLE_EQ(t.surfaceOffset(Vec3(100, 0, 5.5), hint), t.lift);
    EXPECT_NEAR(t.position(hint)(2), 10, 0.1);
}

// Regression: reset gave every wheel the flat-ground resting compression; on a slope the damper saw a jump and kicked
// the stationary car upwards
TEST(RacingVehicle, SpawnOnASlopeHasNoUpwardKick)
{
    Vehicle v;
    v.ground = [](double, double z) { return 0.1 * z; };
    v.reset(Vec3(0, NAN, 0), 0);
    VehicleControls c;

    v.step(c, kDt);
    EXPECT_LT(v.state.velocity(1), 0.2);
}
