/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/util/ini/Params.h"


namespace Saiga
{
struct SAIGA_VISION_API TrainParams : public ParamsBase
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
    std::pair<std::vector<int>, std::vector<int>> Split(std::vector<int> all_indices) const;

    std::vector<int> ReadIndexFile(const std::string& file);
};

}  // namespace Saiga
