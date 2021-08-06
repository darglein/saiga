/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/util/statistics.h"
#include "saiga/core/util/table.h"


namespace Saiga
{
// Computes the image crop as a homography matrix (returned as upper diagonal matrix).
inline IntrinsicsPinholef RandomImageCrop(ivec2 image_size_input, ivec2 image_size_crop, bool translate_to_border,
                                          bool random_translation, vec2 min_max_zoom = vec2(1, 1))
{
    IntrinsicsPinholef K_crop = IntrinsicsPinholef();

    vec2 delta(0, 0);
    float zoom = 1.0f;

    {
        vec2 min_zoom_xy = image_size_crop.array().cast<float>() / image_size_input.array().cast<float>();

        float cmin_zoom = std::max({min_zoom_xy(0), min_zoom_xy(1), min_max_zoom(0)});
        float cmax_zoom = min_max_zoom(1);

        zoom = Random::sampleDouble(cmin_zoom, cmax_zoom);
    }


    vec2 max_translation = image_size_input.cast<float>() * zoom - image_size_crop.cast<float>();


    if (random_translation)
    {
        if (translate_to_border)
        {
            vec2 border = image_size_crop.cast<float>() * 0.5f;
            delta.x()   = Random::sampleDouble(-border.x(), max_translation.x() + border.x());
            delta.y()   = Random::sampleDouble(-border.y(), max_translation.y() + border.y());
        }
        else
        {
            delta.x() = Random::sampleDouble(0, max_translation.x());
            delta.y() = Random::sampleDouble(0, max_translation.y());
        }

        delta = delta.array().max(vec2::Zero().array()).min(max_translation.array());
        // std::cout << "max translation " << max_translation.transpose() << std::endl;
    }
    else
    {
        delta = max_translation * 0.5f;
    }

    K_crop.fx = zoom;
    K_crop.fy = zoom;

    K_crop.cx = -delta(0);
    K_crop.cy = -delta(1);

    return K_crop;
}

inline std::vector<IntrinsicsPinholef> RandomImageCrop(int N, int tries_per_crop, ivec2 image_size_input,
                                                       ivec2 image_size_crop, bool translate_to_border,
                                                       bool random_translation, vec2 min_max_zoom = vec2(1, 1))
{
    std::vector<vec2> centers;
    std::vector<IntrinsicsPinholef> res;
    for (int i = 0; i < N; ++i)
    {
        IntrinsicsPinholef best;
        vec2 best_c;
        float best_dis = -1;

        for (int j = 0; j < tries_per_crop; ++j)
        {
            auto intr = RandomImageCrop(image_size_input, image_size_crop, translate_to_border, random_translation,
                                        min_max_zoom);

            vec2 c = image_size_crop.cast<float>() * 0.5f;
            c      = intr.inverse().normalizedToImage(c);

            float dis = 3573575737;
            for (auto& c2 : centers)
            {
                float d = (c - c2).squaredNorm();
                if (d < dis)
                {
                    dis = d;
                }
            }

            if (centers.empty()) dis = 0;

            if (j == 0 || dis > best_dis)
            {
                best     = intr;
                best_c   = c;
                best_dis = dis;
            }
        }

        centers.push_back(best_c);
        res.push_back(best);
    }
    return res;
}

}  // namespace Saiga
