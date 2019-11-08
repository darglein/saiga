/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui_saiga.h"
#include "saiga/vision/VisionTypes.h"

#include <mutex>

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
    struct SAIGA_VISION_API Parameters
    {
        // Number of previous frame that are included
        int smoothness = 3;

        // Exponential factor of new frames compared to old frames.
        // FrameWeight = pow(alpha,currentFrameId - oldFrameId)
        // Range [0,1]
        double alpha = 0.75;

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
    void addRelativeMotion(const SE3& T, size_t frameId, double weight);
    void updateRelativeMotion(const SE3& T, size_t frameId);

    /**
     * Adds an invalid motion. Use this when tracking fails to localize a frame.
     */
    void addInvalidMotion(size_t frameId);

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

    void clear();
   private:
    struct MotionData
    {
        SE3 v;
        double weight = 1;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
    AlignedVector<MotionData> data;
    double averageWeight = 1;
    std::vector<size_t> indices;
    ImGui::Graph grapht = {"Velocity"};
    ImGui::Graph grapha = {"Angular Velocity"};
    std::mutex mut;
    std::vector<double> weights;

    // Cache the current velocity
    void recomputeVelocity();
    SE3 computeVelocity();
    bool validVelocity = false;
    SE3 currentVelocity;
};

}  // namespace Saiga
