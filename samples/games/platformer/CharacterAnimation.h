/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/platformer/CharacterController.h"

#include <array>

namespace Saiga
{
namespace Game
{
/**
 * Procedural skeletal animation of the platformer tech demo (docs/engine_design.md §18.4).
 *
 * - Skeleton: a joint hierarchy (pelvis, spine, head, arms, legs) with offsets from the parent; the body parts are
 *   boxes attached to the joints (rigid skinning, one bone per part).
 * - Pose: the local rotation of every joint plus an offset of the root.
 * - Animator: evaluates the poses of the states (idle with breathing, locomotion: a walk and a run cycle blended by the
 *   speed, air: jump and fall poses by the vertical speed) and blends them with smoothed weights that sum to 1; a
 *   landing adds a squash. Evaluated per rendered frame from the fixed-step controller state.
 *
 * Transitional: skinned meshes, animation clips, the state machine and pose LOD of the engine (S34) replace it.
 */
enum Joint
{
    JointPelvis = 0,
    JointSpine,
    JointHead,
    JointUpperArmL,
    JointLowerArmL,
    JointUpperArmR,
    JointLowerArmR,
    JointThighL,
    JointShinL,
    JointThighR,
    JointShinR,
    JointCount
};

struct Skeleton
{
    struct JointInfo
    {
        int parent;
        // Offset from the parent joint (bind pose, meters)
        Vec3 offset;
    };
    std::array<JointInfo, JointCount> joints;
    static const Skeleton& humanoid();
};

struct Pose
{
    std::array<Quat, JointCount> rotations;
    Vec3 rootOffset = Vec3(0, 0, 0);
    static Pose identity();
};

// Blends poses with weights (normalized weighted quaternion average per joint, weighted root offset)
Pose blendPoses(const Pose* poses, const double* weights, int n);

// World transforms of the joints of a character at feet position with facing yaw
std::array<Mat4, JointCount> jointTransforms(const Skeleton& skeleton, const Pose& pose, const Vec3& feet, double yaw);

class CharacterAnimator
{
   public:
    struct Weights
    {
        double idle = 1, move = 0, air = 0;
    };

    // Advances the animation by the frame time dt from the (interpolated) controller state
    void update(const CharacterState& state, double horizontalSpeed, double dt);
    const Pose& pose() const { return current; }
    const Weights& weights() const { return w; }
    double cyclePhase() const { return phase; }
    double runBlend() const { return run; }

    // The poses of the states
    static Pose idlePose(double time);
    static Pose locomotionPose(double phase, double run);
    static Pose airPose(double verticalSpeed);

    // Landing impacts (m/s) that are fed in by the game: the controller reports them per fixed step
    void land(double impactSpeed);

   private:
    Pose current = Pose::identity();
    Weights w;
    double time = 0, phase = 0;
    double move = 0, run = 0, air = 0;
    double squash = 0, squashTarget = 0;
    double verticalSpeed = 0;
};

}  // namespace Game
}  // namespace Saiga
