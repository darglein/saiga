/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/util/ini/ini.h"


namespace Saiga
{
struct SAIGA_VISION_API TrainParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(TrainParams);
    virtual void Params(Saiga::SimpleIni& ini_) override
    {
        SAIGA_PARAM_LONG(random_seed);
        SAIGA_PARAM_BOOL(do_train);
        SAIGA_PARAM_BOOL(do_eval);
        SAIGA_PARAM_LONG(batch_size);
        SAIGA_PARAM_LONG(inner_batch_size);
        SAIGA_PARAM_LONG(inner_sample_size);
        SAIGA_PARAM_LONG(num_epochs);
        SAIGA_PARAM_LONG(save_checkpoints_its);
        SAIGA_PARAM_BOOL(eval_only_on_checkpoint);
        SAIGA_PARAM_STRING(name);
        SAIGA_PARAM_BOOL(debug);
        SAIGA_PARAM_STRING(output_file_type);


        SAIGA_PARAM_STRING(split_method);
        SAIGA_PARAM_LONG(max_images);
        SAIGA_PARAM_LONG(duplicate_train_factor);
        SAIGA_PARAM_BOOL(shuffle_initial_indices);
        SAIGA_PARAM_BOOL(shuffle_train_indices);
        SAIGA_PARAM_LONG(split_remove_neighbors);
        SAIGA_PARAM_STRING(split_index_file_train);
        SAIGA_PARAM_STRING(split_index_file_test);
        SAIGA_PARAM_BOOL(train_on_eval);
        SAIGA_PARAM_DOUBLE(train_factor);
        SAIGA_PARAM_LONG(num_workers_train);
        SAIGA_PARAM_LONG(num_workers_eval);
    }

    // ======== Train Control =========
    uint64_t random_seed         = 3746934646;
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
