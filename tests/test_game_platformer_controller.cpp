/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "games/platformer/CharacterAnimation.h"
#include "games/platformer/CharacterController.h"
#include "games/platformer/PlatformerDemo.h"
#include "gtest/gtest.h"

using namespace Saiga;
using namespace Saiga::Game;

namespace
{
constexpr double kDt = 1.0 / 60.0;

Level flatLevel()
{
    Level l;
    l.boxes.push_back({Vec3(-50, -1, -50), Vec3(50, 0, 50), Vec3(1, 1, 1)});
    return l;
}

void run(CharacterController& c, const CharacterInput& in, double seconds)
{
    for (int i = 0; i < int(seconds / kDt + 0.5); ++i) c.step(in, kDt);
}
}  // namespace

TEST(PlatformerController, FallsAndLands)
{
    Level l = flatLevel();
    CharacterController c;
    c.level = &l;
    c.reset(Vec3(0, 5, 0));
    run(c, CharacterInput(), 2.0);
    EXPECT_TRUE(c.state.grounded);
    EXPECT_NEAR(c.state.position(1), 0, 1e-9);
    EXPECT_NEAR(c.state.velocity(1), 0, 1e-9);
}

TEST(PlatformerController, JumpHeight)
{
    Level l = flatLevel();
    CharacterController c;
    c.level = &l;
    c.reset(Vec3(0, 0, 0));
    run(c, CharacterInput(), 0.2);
    ASSERT_TRUE(c.state.grounded);

    CharacterInput jump;
    jump.jumpHeld    = true;
    jump.jumpPressed = true;
    c.step(jump, kDt);
    EXPECT_TRUE(c.state.jumped);
    jump.jumpPressed = false;
    double top       = 0;
    for (int i = 0; i < 120; ++i)
    {
        c.step(jump, kDt);
        top = std::max(top, c.state.position(1));
    }
    // v^2 / 2g = 81 / 50 = 1.62 m (minus the discretization)
    EXPECT_NEAR(top, 1.62, 0.1);
    EXPECT_TRUE(c.state.grounded);

    // A short tap jumps lower (variable jump height)
    c.reset(Vec3(0, 0, 0));
    run(c, CharacterInput(), 0.2);
    CharacterInput tap;
    tap.jumpHeld = tap.jumpPressed = true;
    c.step(tap, kDt);
    tap.jumpHeld = tap.jumpPressed = false;
    double tapTop                  = 0;
    for (int i = 0; i < 120; ++i)
    {
        c.step(tap, kDt);
        tapTop = std::max(tapTop, c.state.position(1));
    }
    EXPECT_LT(tapTop, top - 0.3);
}

TEST(PlatformerController, LandsOnAPlatformAndIsBlockedByWalls)
{
    Level l = flatLevel();
    l.addPlatform(Vec3(0, 1.0, -4), 1.5, 1.5, Vec3(1, 1, 1));
    l.boxes.push_back({Vec3(5, 0, -1), Vec3(6, 3, 1), Vec3(1, 1, 1)});
    CharacterController c;
    c.level = &l;
    c.reset(Vec3(0, 0, 0));
    run(c, CharacterInput(), 0.2);

    // Run towards the platform and jump onto it
    CharacterInput in;
    in.move = Vec3(0, 0, -1);
    run(c, in, 0.15);
    in.jumpPressed = in.jumpHeld = true;
    c.step(in, kDt);
    in.jumpPressed = false;
    run(c, in, 0.5);
    in.move = Vec3(0, 0, 0);
    run(c, in, 1.0);
    EXPECT_TRUE(c.state.grounded);
    EXPECT_NEAR(c.state.position(1), 1.0, 1e-9);

    // The wall at x = 5 stops the character
    c.reset(Vec3(0, 0, 0));
    in      = CharacterInput();
    in.move = Vec3(1, 0, 0);
    run(c, in, 3.0);
    EXPECT_NEAR(c.state.position(0), 5 - c.params.radius, 1e-9);
}

TEST(PlatformerController, NoTunnellingThroughThinPlatforms)
{
    // A 5 cm platform, a character only 0.5 m high and a fall of 400 m: about 140 m/s at the platform, 2.4 m per step,
    // much more than the character's height, so without sub steps the box would pass the platform between two steps
    Level l;
    l.boxes.push_back({Vec3(-2, -0.05, -2), Vec3(2, 0, 2), Vec3(1, 1, 1)});
    CharacterController c;
    c.level         = &l;
    c.params.height = 0.5;
    c.params.radius = 0.15;
    c.reset(Vec3(0, 400, 0));
    run(c, CharacterInput(), 10.0);
    EXPECT_TRUE(c.state.grounded);
    EXPECT_NEAR(c.state.position(1), 0, 1e-9);
}

TEST(PlatformerController, HeadDoesNotStickToTheCeiling)
{
    // Regression: a blocked upward move kept its speed, the head stayed pressed against the ceiling for about 0.3 s
    Level l = flatLevel();
    // A ceiling 2.3 m above the floor (the character is 1.8 m high and jumps about 1.6 m)
    l.boxes.push_back({Vec3(-5, 2.3, -5), Vec3(5, 3.3, 5), Vec3(1, 1, 1)});
    CharacterController c;
    c.level = &l;
    c.reset(Vec3(0, 0, 0));
    run(c, CharacterInput(), 0.2);
    ASSERT_TRUE(c.state.grounded);
    CharacterInput jump;
    jump.jumpHeld = jump.jumpPressed = true;
    c.step(jump, kDt);
    jump.jumpPressed   = false;
    int stepsAtCeiling = 0;
    for (int i = 0; i < 120; ++i)
    {
        c.step(jump, kDt);
        if (c.state.position(1) + c.params.height > 2.3 - 1e-3) stepsAtCeiling++;
    }
    EXPECT_GT(stepsAtCeiling, 0) << "the jump does not reach the ceiling";
    EXPECT_LE(stepsAtCeiling, 1);
    EXPECT_TRUE(c.state.grounded);
}

TEST(PlatformerController, RunSpeedDoesNotDependOnTheStepSize)
{
    // Regression: a blocked vertical move (standing on the ground) ended the sub step loop and dropped the remaining
    // horizontal sub steps, so the character ran at half speed with 30 Hz steps
    for (double dt : {1.0 / 30, 1.0 / 60, 1.0 / 120})
    {
        Level l = flatLevel();
        CharacterController c;
        c.level = &l;
        c.reset(Vec3(0, 0, 0));
        CharacterInput in;
        in.move = Vec3(1, 0, 0);
        for (int i = 0; i < int(2.0 / dt + 0.5); ++i) c.step(in, dt);
        double x0 = c.state.position(0);
        for (int i = 0; i < int(1.0 / dt + 0.5); ++i) c.step(in, dt);
        EXPECT_NEAR(c.state.position(0) - x0, c.params.runSpeed, 0.05) << "dt " << dt;
        EXPECT_TRUE(c.state.grounded);
    }
}

TEST(PlatformerController, CoyoteTimeAndJumpBuffer)
{
    Level l;
    l.boxes.push_back({Vec3(-2, -1, -2), Vec3(2, 0, 2), Vec3(1, 1, 1)});
    CharacterController c;
    c.level = &l;
    c.reset(Vec3(0, 0, 0));
    run(c, CharacterInput(), 0.2);
    // Walk off the ledge, jump 3 steps later: still allowed
    CharacterInput in;
    in.move = Vec3(1, 0, 0);
    for (int i = 0; i < 600 && c.state.grounded; ++i) c.step(in, kDt);
    ASSERT_FALSE(c.state.grounded) << "the character did not leave the platform";
    run(c, in, 2 * kDt);
    in.jumpPressed = in.jumpHeld = true;
    c.step(in, kDt);
    EXPECT_TRUE(c.state.jumped);
    EXPECT_GT(c.state.velocity(1), 0);
}

TEST(PlatformerAnimation, WeightsSumToOneAndPosesAreContinuous)
{
    Level l = flatLevel();
    CharacterController c;
    c.level = &l;
    c.reset(Vec3(0, 0, 0));
    CharacterAnimator anim;

    // Idle, run, jump while running, land, stop
    std::vector<CharacterInput> script;
    CharacterInput idle, runIn, jump;
    runIn.move    = Vec3(0, 0, -1);
    jump          = runIn;
    jump.jumpHeld = jump.jumpPressed = true;
    for (int i = 0; i < 60; ++i) script.push_back(idle);
    for (int i = 0; i < 120; ++i) script.push_back(runIn);
    script.push_back(jump);
    for (int i = 0; i < 90; ++i) script.push_back(runIn);
    for (int i = 0; i < 90; ++i) script.push_back(idle);

    Pose previous   = Pose::identity();
    bool first      = true;
    double maxDelta = 0, maxSquashDrop = 0;
    bool sawAir = false, sawRun = false;
    for (auto& in : script)
    {
        c.step(in, kDt);
        if (c.state.landingSpeed > 0) anim.land(c.state.landingSpeed);
        anim.update(c.state, c.horizontalSpeed(), kDt);
        auto& w = anim.weights();
        ASSERT_NEAR(w.idle + w.move + w.air, 1.0, 1e-12);
        EXPECT_GE(w.idle, 0);
        EXPECT_GE(w.move, 0);
        EXPECT_GE(w.air, 0);
        sawAir = sawAir || w.air > 0.5;
        sawRun = sawRun || anim.runBlend() > 0.5;

        const Pose& p = anim.pose();
        if (!first)
        {
            for (int j = 0; j < JointCount; ++j)
            {
                double dot = std::abs(p.rotations[j].dot(previous.rotations[j]));
                maxDelta   = std::max(maxDelta, 2 * std::acos(std::min(dot, 1.0)));
            }
            maxSquashDrop = std::max(maxSquashDrop, previous.rootOffset(1) - p.rootOffset(1));
        }
        previous = p;
        first    = false;
    }
    EXPECT_TRUE(sawAir);
    EXPECT_TRUE(sawRun);
    // At 60 Hz no joint jumps by more than about 20 degrees per frame (the run cycle moves the legs fastest)
    EXPECT_LT(maxDelta, 0.35);
    EXPECT_LT(maxSquashDrop, 0.05);
}

TEST(PlatformerAnimation, ForwardKinematicsOfTheBindPose)
{
    auto t = jointTransforms(Skeleton::humanoid(), Pose::identity(), Vec3(10, 2, 3), 0);
    // The shins end 0.45 m below the knees: the feet are on the ground in the bind pose
    Vec3 foot = transformPoint(t[JointShinL], Vec3(0, -0.45, 0));
    EXPECT_NEAR(foot(1), 2.0, 1e-9);
    EXPECT_NEAR(transformPoint(t[JointHead], Vec3(0, 0, 0))(1), 2 + 0.95 + 0.1 + 0.55, 1e-9);
}

TEST(PlatformerDemo, ReachesTheGoalOfTheDemoLevel)
{
    Level l = Level::demoLevel();
    CharacterController c;
    c.level = &l;
    c.reset(l.start);
    PlatformerDemo demo;
    bool reached = false;
    int respawns = 0;
    for (int i = 0; i < 60 * 60 && !reached; ++i)
    {
        c.step(demo.input(c, l), kDt);
        if (c.state.position(1) < l.killHeight)
        {
            respawns++;
            c.reset(l.start);
            demo = PlatformerDemo();
        }
        reached = demo.reachedGoal(c, l);
    }
    EXPECT_TRUE(reached);
    EXPECT_EQ(respawns, 0);
}
