/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <mutex>
#include "saiga/imgui/imgui_saiga.h"
#include "saiga/vision/VisionTypes.h"

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
class SAIGA_GLOBAL MotionModel
{
   public:
    struct SAIGA_GLOBAL Parameters
    {
        // Number of previous frame that are included
        int smoothness = 3;

        // Exponential factor of new frames compared to old frames.
        // Range [0,1]
        double alpha = 0.5;

        // Velocity damping applied at the end
        // Range [0,1]
        double damping = 0.9;

        // Used for converting the frame velocity to real velocity
        double fps = 30;

        /**
         *  Reads all paramters from the given config file.
         *  Creates the file with the default values if it doesn't exist.
         */
        void fromConfigFile(const std::string& file);
    };
    Parameters params;

    MotionModel(const Parameters& params);

    /**
     * Adds a relative transformation between two frames.
     */
    void addRelativeMotion(const SE3& T, size_t frameId);
    void updateRelativeMotion(const SE3& T, size_t frameId);


    /**
     * Computes the current velocity in units/frame. You can add it to the last frame position
     * to get the new extimated location:
     */
    SE3 getFrameVelocity();

    /**
     * Real velocity in units/second.
     *
     * = getFrameVelocity() * fps
     */
    SE3 getRealVelocity();

    SE3 predictNextPose(const SE3& currentPose) { return getFrameVelocity() * currentPose; }

    void renderVelocityGraph();

   private:
    AlignedVector<SE3> data;
    std::vector<size_t> indices;
    ImGui::Graph grapht = {"Velocity"};
    ImGui::Graph grapha = {"Angular Velocity"};
    std::mutex mut;
};

}  // namespace Saiga
