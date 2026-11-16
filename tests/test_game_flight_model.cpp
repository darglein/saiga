/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "games/common/FixedStep.h"
#include "games/flight/FlightAutopilot.h"
#include "games/flight/FlightModel.h"
#include "gtest/gtest.h"

using namespace Saiga;
using namespace Saiga::Game;

namespace
{
FlightModel flatWorld(double groundY = 0)
{
    FlightModel m;
    m.ground = [groundY](double, double) { return groundY; };
    return m;
}

void fly(FlightModel& m, const AircraftControls& c, double seconds)
{
    FixedStep clock;
    clock.maxSteps = 1 << 30;
    // Plus half a step: the accumulated frame time is not an exact multiple of dt
    clock.advance(seconds + clock.dt * 0.5, [&](double dt) { m.step(c, dt); });
}
}  // namespace

TEST(FlightModel, TrimmedLevelFlightHoldsAltitude)
{
    FlightModel m = flatWorld();
    double v      = m.trimSpeed();
    EXPECT_GT(v, 40);
    EXPECT_LT(v, 80);
    m.reset(Vec3(0, 500, 0), 0, v);
    AircraftControls c;
    c.throttle = m.trimThrottle(v);
    for (int i = 0; i < 10; ++i) fly(m, c, 1.0);
    EXPECT_NEAR(m.state.position(1), 500, 10);
    EXPECT_NEAR(m.airspeed(), v, 3);
    EXPECT_NEAR(m.bankAngle(), 0, 1e-6);
    EXPECT_FALSE(m.state.crashed);
}

TEST(FlightModel, PitchUpClimbs)
{
    FlightModel level = flatWorld(), climb = flatWorld();
    double v = level.trimSpeed() * 1.15;
    level.reset(Vec3(0, 500, 0), 0, v);
    climb.reset(Vec3(0, 500, 0), 0, v);
    AircraftControls c;
    c.throttle          = 0.8;
    AircraftControls up = c;
    up.pitch            = 0.4;
    fly(level, c, 5.0);
    fly(climb, up, 2.0);
    fly(climb, c, 3.0);
    EXPECT_GT(climb.state.position(1), level.state.position(1) + 20);
    EXPECT_GT(climb.pitchAngle(), level.pitchAngle());
}

TEST(FlightModel, RollRightTurnsRight)
{
    FlightModel m = flatWorld();
    double v      = m.trimSpeed() * 1.1;
    m.reset(Vec3(0, 800, 0), 0, v);
    AircraftControls c;
    c.throttle = 0.6;
    c.roll     = 0.5;
    fly(m, c, 1.0);
    EXPECT_GT(m.bankAngle(), 0.2);
    c.roll  = 0;
    c.pitch = 0.3;
    fly(m, c, 5.0);
    // Heading 0 is -z, a right turn goes towards +x
    EXPECT_GT(m.state.position(0), 50);
}

TEST(FlightModel, StallAtHighAngleOfAttack)
{
    FlightModel m = flatWorld();
    EXPECT_GT(m.liftCoefficient(m.params.stallAngle), m.liftCoefficient(m.params.stallAngle + 0.15));
    EXPECT_GT(m.liftCoefficient(0.05), m.liftCoefficient(0.0));

    // Nose 30 degrees above the flight path
    m.reset(Vec3(0, 800, 0), 0, 30);
    m.state.orientation = m.state.orientation * axisAngle(Vec3(1, 0, 0), 30.0 * kPi / 180.0);
    EXPECT_GT(m.angleOfAttack(), m.params.stallAngle);
    EXPECT_TRUE(m.stalled());

    // Too slow with the stick pulled back: the plane stalls and sinks
    m.reset(Vec3(0, 800, 0), 0, 30);
    AircraftControls c;
    c.throttle   = 0.0;
    c.pitch      = 1.0;
    bool stalled = false;
    FixedStep clock;
    clock.maxSteps = 1000;
    clock.advance(5.0,
                  [&](double dt)
                  {
                      m.step(c, dt);
                      stalled = stalled || m.stalled();
                  });
    EXPECT_TRUE(stalled);
    EXPECT_LT(m.state.position(1), 800);
}

TEST(FlightModel, CrashIntoTheGround)
{
    FlightModel m = flatWorld(100);
    m.reset(Vec3(0, 300, 0), 0, 60);
    m.state.orientation = m.state.orientation * axisAngle(Vec3(1, 0, 0), -40.0 * kPi / 180.0);
    m.state.velocity    = m.forward() * 60.0;
    AircraftControls c;
    c.throttle = 1;
    FixedStep clock;
    clock.maxSteps = 100000;
    clock.advance(20.0, [&](double dt) { m.step(c, dt); });
    EXPECT_TRUE(m.state.crashed);
    EXPECT_NEAR(m.state.position(1), 100 + m.params.gearHeight, 1e-6);
    // Crashed: no more movement
    Vec3 p = m.state.position;
    fly(m, c, 1.0);
    EXPECT_EQ(m.state.position, p);
}

TEST(FlightModel, GentleTouchdownIsNoCrash)
{
    FlightModel m = flatWorld(0);
    m.reset(Vec3(0, 3, 0), 0, m.trimSpeed() * 0.9);
    m.state.velocity(1) = -1.0;
    AircraftControls c;
    c.throttle = 0;
    fly(m, c, 3.0);
    EXPECT_FALSE(m.state.crashed);
    EXPECT_TRUE(m.state.onGround);
}

TEST(FlightAutopilot, FliesTheCircuitWithoutCrashing)
{
    // Rolling hills
    FlightModel m;
    m.ground = [](double x, double z) { return 80 * std::sin(x / 700) * std::cos(z / 900); };
    FlightAutopilot pilot;
    pilot.center = Vec3(0, 0, 0);
    pilot.radius = 1500;
    // On the circle, heading along its counter clockwise tangent (+z at +x)
    m.reset(Vec3(1500, 400, 0), headingOf(Vec3(0, 0, 1)), 65);
    double minHeight = 1e30, maxDist = 0;
    FixedStep clock;
    clock.maxSteps = 1000000;
    clock.advance(180.0,
                  [&](double dt)
                  {
                      m.step(pilot.controls(m), dt);
                      minHeight = std::min(minHeight, m.altitudeAboveGround());
                      maxDist   = std::max(maxDist, Vec3(m.state.position(0), 0, m.state.position(2)).norm());
                  });
    EXPECT_FALSE(m.state.crashed);
    EXPECT_GT(minHeight, 100);
    EXPECT_LT(maxDist, pilot.radius + 600);
    EXPECT_NEAR(m.airspeed(), pilot.speed, 10);
}
