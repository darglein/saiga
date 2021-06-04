/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"

namespace Saiga
{
struct SAIGA_VISION_API ScalePyramid
{
    // Maybe this will be changed to a template later.
    using T = double;

    ScalePyramid(int levels = 1, T scale_factor = 1, int total_features = 1000);

    bool IsValidScaleLevel(int level) { return level >= 0 && level < num_levels; }


    // Given the distance to a point and the observed scale level, this function computes the min and max distance this
    // point might be observed. The min distance corresponds to a match at the highest scale level and the max distance
    // to a match at the lowest scale.
    std::pair<T, T> EstimateMinMaxDistance(T distance_to_world_point, int level_of_keypoint)
    {
        // The distance we expect on the scale level 0
        T max_distance = distance_to_world_point * Scale(level_of_keypoint);

        // The distance at the maximum scale level
        T min_distance = max_distance / Scale(num_levels - 1);

        // The keypoint can be right on the edge of a scale level
        // -> Add one more level to min and max
        min_distance /= scale_factor;
        max_distance *= scale_factor;

        return {min_distance, max_distance};
    }


    T PredictScaleLevel(T reference_distance_to_world_point, int reference_level_of_keypoint, T distance_to_world_point)
    {
        // The distance we expect on the scale level 0
        T max_distance = reference_distance_to_world_point * Scale(reference_level_of_keypoint);


        // We assume the keypoint has scale = 1 at maxDistance.
        // This value gives the expected scale of the keypoint at the actual distance.
        T predicted_scale = max_distance / distance_to_world_point;

        // inverse scale formula
        // scale = factor^i    ->    log_factor(scale) = i    ->    log(scale) / log(factor) = i
        return log(predicted_scale) / log_scale_factor;
    }

    static inline bool PredictionConsistent(T predicted_scale, int level_of_keypoint)
    {
        T prediction_error = predicted_scale - level_of_keypoint;

        return prediction_error * prediction_error <
               (scale_prediction_error_threshold * scale_prediction_error_threshold);
    }

    // Predict the scale of a 3D point for a new image. This predicted level can be compared to the observed scale to
    // filter outliers.
    bool CheckScaleConsistencyOfObservation(T reference_distance_to_world_point, int reference_level_of_keypoint,
                                            T distance_to_world_point, int level_of_keypoint)
    {
        T predicted_scale_level =
            PredictScaleLevel(reference_distance_to_world_point, reference_level_of_keypoint, distance_to_world_point);

        return PredictionConsistent(predicted_scale_level, level_of_keypoint);
    }


    inline T ScaleForContiniousLevel(T level) { return pow(scale_factor, level); }
    inline T Scale(int level) { return levels[level].scale; }
    inline T SquaredScale(int level) { return levels[level].squared_scale; }
    inline T InverseScale(int level) { return levels[level].inv_scale; }
    inline T InverseSquaredScale(int level) { return levels[level].inv_squared_scale; }
    inline T Factor() { return scale_factor; }
    inline int Features(int level) { return levels[level].num_features; }


    // Number of scale levels.
    int num_levels;

    int total_num_features;

    // The scale factor between the levels
    T scale_factor;


   private:
    // ====
    // Precomputed values (in the constructor) from the variables above.

    // log(scale_factor).
    T log_scale_factor;

    struct Level
    {
        // The scale factors for each level:
        // factor[i] = scale_factor^i
        T scale;

        // Inverse scale factors.
        // factor[i] = 1.0 / scale_factor^i
        T squared_scale;

        // Squared scale factors
        // factor[i] = (scale_factor^i)^2
        T inv_scale;

        // Inverse squared scale factors
        // factor[i] = 1.0 / (scale_factor^i)^2
        T inv_squared_scale;

        int num_features;
    };

    std::vector<Level> levels;

    static constexpr T scale_prediction_error_threshold = 1.2;
};

std::ostream& operator<<(std::ostream& strm, const ScalePyramid& sp);


}  // namespace Saiga
