/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/math/random.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/ini/Params.h"

#include <iostream>

namespace Saiga
{
struct TrainParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(TrainParams);
    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    {
        SAIGA_PARAM(random_seed);
        SAIGA_PARAM(do_train);
        SAIGA_PARAM(do_eval);
        SAIGA_PARAM(batch_size);
        SAIGA_PARAM(inner_batch_size);
        SAIGA_PARAM(inner_sample_size);
        SAIGA_PARAM(num_epochs);
        SAIGA_PARAM(save_checkpoints_its);
        SAIGA_PARAM(eval_only_on_checkpoint);
        SAIGA_PARAM(name);
        SAIGA_PARAM(debug);
        SAIGA_PARAM(output_file_type);
        SAIGA_PARAM(checkpoint_directory);


        SAIGA_PARAM(split_method);
        SAIGA_PARAM(max_images);
        SAIGA_PARAM(duplicate_train_factor);
        SAIGA_PARAM(shuffle_initial_indices);
        SAIGA_PARAM(shuffle_train_indices);
        SAIGA_PARAM(split_remove_neighbors);
        SAIGA_PARAM(split_index_file_train);
        SAIGA_PARAM(split_index_file_test);
        SAIGA_PARAM(train_on_eval);
        SAIGA_PARAM(train_factor);
        SAIGA_PARAM(num_workers_train);
        SAIGA_PARAM(num_workers_eval);
    }

    // ======== Train Control =========
    long random_seed             = 3746934646;
    bool do_train                = true;
    bool do_eval                 = true;
    int num_epochs               = 20;
    int save_checkpoints_its     = 5;
    bool eval_only_on_checkpoint = false;
    int batch_size               = 4;
    int inner_batch_size         = 1;
    int inner_sample_size        = 1;
    bool debug                   = false;

    std::string output_file_type = ".jpg";

    std::string name = "default";
    std::string checkpoint_directory     = "";

    // ======== Dataset splitting ========
    std::string split_method     = "";
    bool train_on_eval           = false;
    double train_factor          = 0.9;
    int max_images               = -1;
    int duplicate_train_factor   = 1;
    bool shuffle_initial_indices = false;
    bool shuffle_train_indices   = true;
    int num_workers_train        = 2;
    int num_workers_eval         = 2;

    // If these paths are set to existing files this index list is used for the split.
    // The file should contain one number per row in ascii.
    std::string split_index_file_train = "";
    std::string split_index_file_test  = "";

    // Let's say this value is set to 2, then if the idx 7 was selected as
    // test index the idxs 5,6,8,9 are deleted from train.
    int split_remove_neighbors = 0;


    // Splits the index array into [train, eval] indices using the parameters of this struct
    inline std::pair<std::vector<int>, std::vector<int>> Split(std::vector<int> all_indices) const;

    static inline std::vector<int> ReadIndexFile(const std::string& file);

    inline std::string ExperimentString() { return CurrentTimeString("%F_%H-%M-%S") + "_" + name; }
};



template <typename T>
inline std::pair<std::vector<T>, std::vector<T>> SplitDataset(std::vector<T> data, float ratio)
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
        if (l.empty()) continue;
        res.push_back(to_int(l));
    }

    return res;
}


inline std::vector<int> ReduceIndicesUniform(std::vector<int> all_indices, int target_size)
{
    if (target_size < 0)
    {
        return all_indices;
    }

    target_size = std::min<int>(target_size, all_indices.size());

    double step = (double)all_indices.size() / target_size;

    std::vector<int> result;
    for (int i = 0; i < target_size; ++i)
    {
        int x = round(i * step);
        result.push_back(all_indices[x]);
    }
    return result;
}


}  // namespace Saiga
