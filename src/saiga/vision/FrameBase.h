/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/Thread/SpinLock.h"
#include "saiga/vision/VisionTypes.h"

#include <mutex>
namespace Saiga
{
/**
 * A frame pose is the transformation from world to camera space.
 * A projection from world->imagespace is therefore:
 *
 * imagePoint = cameraIntrinsics * framePose * worldPoint
 */
class FramePose
{
   public:
    SE3 Pose() const { return se3; }
    void setPose(const SE3& v) { se3 = v; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
   protected:
    SE3 se3;
};

class FramePoseSync
{
   public:
    SE3 Pose() const
    {
        std::unique_lock lock(poseMutex);
        return se3;
    }
    void setPose(const SE3& v)
    {
        std::unique_lock lock(poseMutex);
        se3 = v;
    }

    // Expose the mutex so it can be used by the base class to sync additional things.
    auto& getPoseMutex() const { return poseMutex; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   protected:
    SE3 se3;
    // use a fast spinlock here befause the critical sections are extremly small
    mutable SpinLock poseMutex;
};

// Derive from either FramePose or FramePoseSync depending on the SYNC parameter.
template <bool SYNC = false>
class FrameBase : public std::conditional<SYNC, FramePoseSync, FramePose>::type
{
   public:
    using PoseType = typename std::conditional<SYNC, FramePoseSync, FramePose>::type;

    using PoseType::Pose;
    using PoseType::setPose;



    Vec3 CameraPosition() const { return Pose().inverse().translation(); }
    Mat4 ViewMatrix() const { return Pose().matrix(); }
    Mat4 ModelMatrix() const { return Pose().inverse().matrix(); }

    SE3 PoseInv() const { return Pose().inverse(); }
    //    SE3 Pose() { return se3.Pose(); }
    void setPose(const Mat4& value) { setPose(SE3::fitToSE3(value)); }
    //    void setPose(const SE3& value) { se3.setPose(value); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



}  // namespace Saiga
