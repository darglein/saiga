/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "TrainParameters.h"

#include "saiga/core/math/random.h"
#include "saiga/core/util/file.h"

#include <filesystem>
#include <iostream>

namespace Saiga
{
template <typename T>
std::pair<std::vector<T>, std::vector<T>> SplitDataset(std::vector<T> data, float ratio)
{
    int total_n = data.size();
    int first_n = std::round(total_n * ratio);
    std::vector<T> a(data.begin(), data.begin() + first_n);
    std::vector<T> b(data.begin() + first_n, data.end());
    return {a, b};
}

std::pair<std::vector<int>, std::vector<int>> TrainParams::Split(std::vector<int> all_indices) const
{
    if (shuffle_initial_indices)
    {
        std::shuffle(all_indices.begin(), all_indices.end(), Random::generator());
    }
    if (max_images > 0)
    {
        all_indices.resize(std::min<int>(max_images, all_indices.size()));
    }

    std::vector<int> train_indices;
    std::vector<int> test_indices;

    if (split_method == "last")
    {
        // train: [0, ..., n-1]
        // test:  [n,..., N-1]
        int total_n   = all_indices.size();
        int first_n   = std::round(total_n * train_factor);
        train_indices = std::vector<int>(all_indices.begin(), all_indices.begin() + first_n);
        test_indices  = std::vector<int>(all_indices.begin() + first_n, all_indices.end());
    }
    if (split_method == "random")
    {
    }
    else
    {
        // uniform split
        double train_step = 1.0 / (1 - train_factor);
        SAIGA_ASSERT(std::isfinite(train_step));
        SAIGA_ASSERT(train_step >= 1);

        std::cout << "step " << train_step << std::endl;
        int total_n  = all_indices.size();
        double t     = 0;
        int last_idx = -1;
        while (t < total_n)
        {
            int idx = std::round(t);
            test_indices.push_back(all_indices[idx]);

            for (int i = last_idx + 1; i < idx; ++i)
            {
                train_indices.push_back(all_indices[i]);
            }


            last_idx = idx;
            t += train_step;
        }

        for (int i = last_idx + 1; i < total_n; ++i)
        {
            train_indices.push_back(all_indices[i]);
        }
    }

    if (split_remove_neighbors > 0)
    {
        std::vector<int> to_remove;
        for (auto i : test_indices)
        {
            for (int j = -split_remove_neighbors; j <= split_remove_neighbors; ++j)
            {
                to_remove.push_back(i + j);
            }
        }

        for (auto i : to_remove)
        {
            auto it = std::find(train_indices.begin(), train_indices.end(), i);
            if (it != train_indices.end())
            {
                train_indices.erase(it);
            }
        }
    }



    if (train_on_eval)
    {
        std::cout << "Train on eval!" << std::endl;
        train_indices = all_indices;

        int n        = test_indices.size();
        test_indices = all_indices;
        test_indices.resize(n);
    }

    return {train_indices, test_indices};
}
std::vector<int> TrainParams::ReadIndexFile(const std::string& file)
{
    SAIGA_ASSERT(std::filesystem::exists(file));
    auto lines = File::loadFileStringArray(file);

    std::vector<int> res;

    for (auto l : lines)
    {
        if(l.empty()) continue;
        res.push_back(to_int(l));
    }

    return res;
}
}  // namespace Saiga
