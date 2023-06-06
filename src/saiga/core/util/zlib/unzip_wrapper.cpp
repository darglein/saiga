/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "unzip_wrapper.h"

#include "saiga/core/util/ProgressBar.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/zlib/unzip.h"

#include <iostream>
namespace Saiga
{

#define dir_delimter '/'
#define MAX_FILENAME 512


static bool UnzipCurrent(unzFile zipfile, Unzipfile& out_data, ProgressBarManager* progress_bar)
{
    // Entry is a file, so extract it.
    std::cout << "  Unzip " << out_data.name << " " << (out_data.compressed_size) << "B -> "
              << (out_data.uncompressed_size) << "B" << std::endl;
    if (unzOpenCurrentFile(zipfile) != UNZ_OK)
    {
        printf("could not open file\n");
        return false;
    }

    char* output_ptr = out_data.user_data_ptr;

    if (!output_ptr)
    {
        out_data.data.resize(out_data.uncompressed_size);
        output_ptr = out_data.data.data();
    }
    size_t current_data = 0;
    {
        // ProgressBar bar(std::cout, "  Unzip", file_info.uncompressed_size / (1000), 30, false, 100, "KB");

        int read_block_size = 10000;

        size_t num_blocks = (out_data.uncompressed_size + read_block_size - 1) / read_block_size;
        auto bar          = SAIGA_OPTIONAL_PROGRESS_BAR(progress_bar, "Unzip", num_blocks);

        int error = UNZ_OK;
        do
        {
            error = unzReadCurrentFile(zipfile, output_ptr + current_data, read_block_size);
            if (error < 0)
            {
                printf("error %d\n", error);
                unzCloseCurrentFile(zipfile);
                return false;
            }

            current_data += error;

            if (bar) bar->addProgress(1);
        } while (error > 0);
        SAIGA_ASSERT(current_data == out_data.uncompressed_size);
    }
    unzCloseCurrentFile(zipfile);
    return true;
}

std::vector<Unzipfile> UnzipInfo(const std::string& path)
{
    // Open the zip file
    unzFile zipfile = unzOpen(path.c_str());
    if (zipfile == NULL)
    {
        printf("%s: not found\n", path.c_str());
        return {};
    }

    // Get info about the zip file
    unz_global_info global_info;
    if (unzGetGlobalInfo(zipfile, &global_info) != UNZ_OK)
    {
        printf("could not read file global info\n");
        unzClose(zipfile);
        return {};
    }

    std::vector<Unzipfile> out_files;
    // Loop to extract all files
    uLong i;
    for (i = 0; i < global_info.number_entry; ++i)
    {
        // Get info about current file.
        unz_file_info64 file_info;
        char filename[MAX_FILENAME];
        if (unzGetCurrentFileInfo64(zipfile, &file_info, filename, MAX_FILENAME, NULL, 0, NULL, 0) != UNZ_OK)
        {
            printf("could not read file info\n");
            unzClose(zipfile);
            return {};
        }
        Unzipfile file;
        file.name              = filename;
        file.compressed_size   = file_info.compressed_size;
        file.uncompressed_size = file_info.uncompressed_size;
        file.file_id_in_zip    = i;
        out_files.emplace_back(std::move(file));

        // Go the the next entry listed in the zip file.
        if ((i + 1) < global_info.number_entry)
        {
            if (unzGoToNextFile(zipfile) != UNZ_OK)
            {
                printf("cound not read next file\n");
                unzClose(zipfile);
                return {};
            }
        }
    }

    unzClose(zipfile);
    return out_files;
}

std::vector<Unzipfile> UnzipToMemory(const std::string& path, ProgressBarManager* progress_bar)
{
    auto info = UnzipInfo(path);
    UnzipToMemory(path, info, progress_bar);
    return info;
}

void UnzipToMemory(const std::string& path, std::vector<Unzipfile>& info, ProgressBarManager* progress_bar)
{
    // Open the zip file
    unzFile zipfile = unzOpen(path.c_str());
    if (zipfile == NULL)
    {
        printf("%s: not found\n", path.c_str());
        return;
    }


    // Loop to extract all files
    for (uLong i = 0; i < info.size(); ++i)
    {
        SAIGA_ASSERT(i == info[i].file_id_in_zip);
        if (!UnzipCurrent(zipfile, info[i], progress_bar))
        {
            unzClose(zipfile);
            return;
        }

        // Go the the next entry listed in the zip file.
        if ((i + 1) < info.size())
        {
            if (unzGoToNextFile(zipfile) != UNZ_OK)
            {
                printf("cound not read next file\n");
                unzClose(zipfile);
                return;
            }
        }
    }
}
void UnzipSingleFileToMemory(const std::string& path, Unzipfile& info, ProgressBarManager* progress_bar)
{
    // Open the zip file
    unzFile zipfile = unzOpen(path.c_str());
    if (zipfile == NULL)
    {
        printf("%s: not found\n", path.c_str());
        return;
    }

    for (int i = 0; i < info.file_id_in_zip; ++i)
    {
        unzGoToNextFile(zipfile);
    }

    UnzipCurrent(zipfile, info, progress_bar);

    unzClose(zipfile);
}


}  // namespace Saiga
