/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/vision/VisionTypes.h"

// This file includes base types for frames and keyframes in a SLAM or SfM system.
// Typically an implementation wants to define their own types, but deriving from the classes here.
namespace Saiga
{
/**
 * A frame pose is the transformation from world to camera space.
 * A projection from world->imagespace is therefore:
 *
 * imagePoint = cameraIntrinsics * framePose * worldPoint
 *
 * This class is basically a wrapper for se3 which includes a few more helper functions used in reconstruction
 * enviroments.
 */
class FramePose
{
   public:
    SE3 Pose() const { return se3_; }
    SE3 PoseInv() const { return Pose().inverse(); }
    void SetPose(const SE3& v) { se3_ = v; }
    void SetPose(const Mat4& value) { SetPose(SE3::fitToSE3(value)); }

    Vec3 CameraPosition() const { return Pose().inverse().translation(); }
    Mat4 ViewMatrix() const { return Pose().matrix(); }
    Mat4 ModelMatrix() const { return Pose().inverse().matrix(); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   protected:
    SE3 se3_;
};


// The base class for a single frame (not necessary a keyframe!).
//
class FrameBase
{
};


class KeyframeBase : public FramePose
{
   public:
    KeyframeBase(int id) : id_(id) {}

    explicit operator bool() const { return !bad_; }
    bool isBad() const { return bad_; }
    bool Valid() const { return !bad_; }
    int id() const { return id_; }

   protected:
    int id_;
    bool bad_ = false;
};



}  // namespace Saiga
