/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/geometry/bspline.h"
#include "saiga/core/math/math.h"
#include "saiga/core/model/UnifiedMesh.h"
#include "saiga/core/sophus/Sophus.h"

#include <saiga/core/time/time.h>

namespace Saiga
{
struct SplineKeyframe
{
    // nearest neighbor interpolation
    int user_index;
    std::string name;

    // Data (linear interpolated)
    Eigen::Matrix<double, -1, 1> user_data;

    // Spherical correct interpolation
    Sophus::SE3d pose;
};


inline SplineKeyframe mix(const SplineKeyframe& a, const SplineKeyframe& b, double alpha)
{
    SplineKeyframe result = alpha < 0.5 ? a : b;
    result.pose           = mix(a.pose, b.pose, alpha);
    result.user_data = (1 - alpha) * a.user_data + alpha * b.user_data;
    return result;
}



// A b-spline path for an object, for example a camera.
class SAIGA_CORE_API SplinePath
{
   public:
    std::vector<SplineKeyframe> keyframes;
    Bspline<SplineKeyframe, double> spline;

    void Insert(const SplineKeyframe& p) { keyframes.push_back(p); }


    std::vector<SplineKeyframe> Trajectory();


    int selectedKeyframe = 0;


    SplinePath() {}


    //    void start(Camera& cam, float totalTimeS, float dt);
    //    bool update(Camera& cam);


    void updateCurve()
    {
        if (keyframes.size() < 4)
        {
            return;
        }

        spline = Bspline<SplineKeyframe, double>(keyframes);
    }

    void PrintUserId();


    // return: true if something was changed
    bool imgui();
    //    void render();
    //    void renderGui(Camera& cam);
    //
    // Camera path mesh
    bool visible        = true;
    int subSamples      = 5;
    float keyframeScale = 0.5;

    float time_in_seconds = 15;
    int frame_rate        = 60;

    // A mesh to visualize the trajectory.
    // Should be called each time an update was made
    UnifiedMesh ProxyMesh();
};



}  // namespace Saiga
