/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/ProgressBar.h"

#include <string>
#include <vector>

namespace Saiga
{

struct Unzipfile
{
    int file_id_in_zip = 0;
    std::string name;
    size_t compressed_size;
    size_t uncompressed_size;


    char* user_data_ptr = nullptr;
    std::vector<char> data;
};


// Returns only the meta info
SAIGA_CORE_API std::vector<Unzipfile> UnzipInfo(const std::string& path);

// Extracts all files inside a .zip into memory
SAIGA_CORE_API std::vector<Unzipfile> UnzipToMemory(const std::string& path,
                                                    ProgressBarManager* progress_bar = nullptr);

// Use this if you previously called unzip info
SAIGA_CORE_API void UnzipToMemory(const std::string& path, std::vector<Unzipfile>& info,
                                  ProgressBarManager* progress_bar = nullptr);

SAIGA_CORE_API void UnzipSingleFileToMemory(const std::string& path, Unzipfile& info,
                                  ProgressBarManager* progress_bar = nullptr);


}  // namespace Saiga
