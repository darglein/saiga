/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui_saiga.h"
#include "saiga/vision/VisionTypes.h"

#include <mutex>
#include <optional>

namespace Saiga
{
/**
 * Simple motion model to predict the next camera position in a SLAM enviroment.
 * Usage:
 *
 *
 * currentFrame->setPose(motionModel.predictNextPose(lastFrame->Pose()));
 *
 * // Compute correct pose
 * // ...
 *
 * SE3 relativeMotion = currentFrame->Pose() * lastFrame->Pose().inverse();
 * motionModel.addRelativeMotion(relativeMotion);
 * lastFrame = currentFrame
 *
 */
class SAIGA_VISION_API MotionModel
{
   public:
    struct SAIGA_VISION_API Settings
    {
        // Number of previous frame that are included
        int valid_range = 1;
        double damping  = 1.0;
        /**
         *  Reads all paramters from the given config file.
         *  Creates the file with the default values if it doesn't exist.
         */
        void fromConfigFile(const std::string& file);
    };
    Settings params;

    MotionModel(const Settings& params) : params(params) { data.reserve(10000); }

    /**
     * Adds a relative transformation between two frames.
     */
    void addRelativeMotion(const SE3& velocity, int frameId);


    /**
     * @brief predictVelocityForFrame
     * @param frameId
     * @return
     */
    std::optional<SE3> predictVelocityForFrame(int frameId);


    // Multiplies the linear velocity by 'scale'.
    // The rotation stays unchanged.
    void ScaleLinearVelocity(double scale);

    void clear() { data.clear(); }

   private:
    struct MotionData
    {
        SE3 velocity;
        bool valid;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    AlignedVector<MotionData> data;


    std::mutex mut;
};

}  // namespace Saiga
