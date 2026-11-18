/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Procedural skeletal animation of the platformer character (poses, blending, forward kinematics).

#include "games/platformer/CharacterAnimation.h"

namespace Saiga
{
namespace Game
{
namespace
{
Quat rx(double a)
{
    return axisAngle(Vec3(1, 0, 0), a);
}
Quat rz(double a)
{
    return axisAngle(Vec3(0, 0, 1), a);
}
double mix(double a, double b, double t)
{
    return a + (b - a) * t;
}
}  // namespace

const Skeleton& Skeleton::humanoid()
{
    // Facing -z, left is -x. Positive rotations about x swing a hanging limb forward.
    static const Skeleton s = []()
    {
        Skeleton k;
        k.joints[JointPelvis]    = {-1, Vec3(0, 0.95, 0)};
        k.joints[JointSpine]     = {JointPelvis, Vec3(0, 0.1, 0)};
        k.joints[JointHead]      = {JointSpine, Vec3(0, 0.55, 0)};
        k.joints[JointUpperArmL] = {JointSpine, Vec3(-0.29, 0.47, 0)};
        k.joints[JointLowerArmL] = {JointUpperArmL, Vec3(0, -0.3, 0)};
        k.joints[JointUpperArmR] = {JointSpine, Vec3(0.29, 0.47, 0)};
        k.joints[JointLowerArmR] = {JointUpperArmR, Vec3(0, -0.3, 0)};
        k.joints[JointThighL]    = {JointPelvis, Vec3(-0.12, -0.05, 0)};
        k.joints[JointShinL]     = {JointThighL, Vec3(0, -0.45, 0)};
        k.joints[JointThighR]    = {JointPelvis, Vec3(0.12, -0.05, 0)};
        k.joints[JointShinR]     = {JointThighR, Vec3(0, -0.45, 0)};
        return k;
    }();
    return s;
}

Pose Pose::identity()
{
    Pose p;
    p.rotations.fill(Quat::Identity());
    return p;
}

Pose blendPoses(const Pose* poses, const double* weights, int n)
{
    Pose r = Pose::identity();
    std::vector<Quat> q(n);
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += weights[i];
    for (int j = 0; j < JointCount; ++j)
    {
        for (int i = 0; i < n; ++i) q[i] = poses[i].rotations[j];
        r.rotations[j] = blendRotations(q.data(), weights, n);
    }
    for (int i = 0; i < n; ++i) r.rootOffset += poses[i].rootOffset * (weights[i] / std::max(sum, 1e-12));
    return r;
}

std::array<Mat4, JointCount> jointTransforms(const Skeleton& sk, const Pose& pose, const Vec3& feet, double yaw)
{
    std::array<Mat4, JointCount> world;
    Mat4 root = transformMatrix(feet + pose.rootOffset, yawRotation(yaw));
    for (int j = 0; j < JointCount; ++j)
    {
        // Parents come before their children in the joint order
        const Mat4& parent = sk.joints[j].parent < 0 ? root : world[sk.joints[j].parent];
        world[j]           = parent * transformMatrix(sk.joints[j].offset, pose.rotations[j]);
    }
    return world;
}

Pose CharacterAnimator::idlePose(double t)
{
    Pose p                      = Pose::identity();
    double breath               = std::sin(2 * kPi * t / 3.5);
    p.rotations[JointSpine]     = rx(0.03 * breath);
    p.rotations[JointHead]      = rx(-0.02 * breath);
    p.rotations[JointUpperArmL] = rz(-0.08 - 0.03 * breath);
    p.rotations[JointUpperArmR] = rz(0.08 + 0.03 * breath);
    p.rotations[JointLowerArmL] = rx(0.15);
    p.rotations[JointLowerArmR] = rx(0.15);
    p.rootOffset                = Vec3(0, 0.01 * breath, 0);
    return p;
}

Pose CharacterAnimator::locomotionPose(double phase, double run)
{
    Pose p     = Pose::identity();
    double s   = std::sin(phase);
    double leg = mix(0.45, 0.8, run);
    double arm = mix(0.35, 0.9, run);
    // Knees bend while the leg swings forward
    double kneeL                = mix(0.25, 1.2, run) * std::max(0.0, std::cos(phase)) + 0.1;
    double kneeR                = mix(0.25, 1.2, run) * std::max(0.0, -std::cos(phase)) + 0.1;
    p.rotations[JointThighL]    = rx(leg * s);
    p.rotations[JointThighR]    = rx(-leg * s);
    p.rotations[JointShinL]     = rx(-kneeL);
    p.rotations[JointShinR]     = rx(-kneeR);
    p.rotations[JointUpperArmL] = rx(-arm * s) * rz(-0.1);
    p.rotations[JointUpperArmR] = rx(arm * s) * rz(0.1);
    p.rotations[JointLowerArmL] = rx(mix(0.3, 1.3, run));
    p.rotations[JointLowerArmR] = rx(mix(0.3, 1.3, run));
    // Lean forward when running, the spine twists against the hips
    p.rotations[JointSpine]  = rx(-mix(0.05, 0.25, run)) * axisAngle(Vec3(0, 1, 0), 0.15 * run * s);
    p.rotations[JointPelvis] = axisAngle(Vec3(0, 1, 0), -0.1 * s);
    // Two steps per cycle: the body is lowest when the legs pass each other
    p.rootOffset = Vec3(0, -mix(0.02, 0.07, run) * std::abs(std::cos(phase)), 0);
    return p;
}

Pose CharacterAnimator::airPose(double vy)
{
    // t = 1: rising fast (jump pose), t = -1: falling fast (arms out, legs apart)
    double t      = clampd(vy / 9.0, -1, 1);
    double rising = std::max(t, 0.0), falling = std::max(-t, 0.0);
    Pose p                      = Pose::identity();
    p.rotations[JointThighL]    = rx(0.9 * rising + 0.2 * falling) * rz(-0.15 * falling);
    p.rotations[JointThighR]    = rx(0.3 * rising - 0.1 * falling) * rz(0.15 * falling);
    p.rotations[JointShinL]     = rx(-1.3 * rising - 0.3 * falling - 0.2);
    p.rotations[JointShinR]     = rx(-0.6 * rising - 0.3 * falling - 0.2);
    p.rotations[JointUpperArmL] = rx(2.4 * rising) * rz(-0.3 - 1.0 * falling);
    p.rotations[JointUpperArmR] = rx(2.4 * rising) * rz(0.3 + 1.0 * falling);
    p.rotations[JointLowerArmL] = rx(0.3 + 0.3 * falling);
    p.rotations[JointLowerArmR] = rx(0.3 + 0.3 * falling);
    p.rotations[JointSpine]     = rx(0.1 * rising - 0.1 * falling);
    return p;
}

void CharacterAnimator::land(double impactSpeed)
{
    squashTarget = std::max(squashTarget, clampd(impactSpeed / 25.0, 0, 0.6));
}

void CharacterAnimator::update(const CharacterState& s, double speed, double dt)
{
    time += dt;
    // Smoothed state blends
    move += (clampd(speed / 1.5, 0, 1) - move) * smoothFactor(8, dt);
    run += (clampd((speed - 2.5) / 4.0, 0, 1) - run) * smoothFactor(6, dt);
    air += ((s.grounded ? 0.0 : 1.0) - air) * smoothFactor(12, dt);
    verticalSpeed += (s.velocity(1) - verticalSpeed) * smoothFactor(15, dt);
    // The cycle advances with the distance travelled: two steps per cycle
    double stride = mix(1.3, 2.6, run);
    phase         = std::fmod(phase + speed * dt / stride * 2 * kPi, 2 * kPi);
    // The landing squash: fast attack, slower release
    squash += (squashTarget - squash) * smoothFactor(30, dt);
    squashTarget *= 1.0 - smoothFactor(6, dt);

    w.air  = air;
    w.move = (1 - air) * move;
    w.idle = (1 - air) * (1 - move);

    Pose poses[3]     = {idlePose(time), locomotionPose(phase, run), airPose(verticalSpeed)};
    double weights[3] = {w.idle, w.move, w.air};
    current           = blendPoses(poses, weights, 3);

    // Additive squash: crouch on the knees, lean forward
    if (squash > 1e-4)
    {
        current.rootOffset(1) -= 0.25 * squash;
        for (int j : {JointThighL, JointThighR}) current.rotations[j] = rx(0.7 * squash) * current.rotations[j];
        for (int j : {JointShinL, JointShinR}) current.rotations[j] = rx(-1.4 * squash) * current.rotations[j];
        current.rotations[JointSpine] = rx(-0.35 * squash) * current.rotations[JointSpine];
    }
}

}  // namespace Game
}  // namespace Saiga
