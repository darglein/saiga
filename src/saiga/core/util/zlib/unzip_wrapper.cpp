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

std::vector<Unzipfile> UnzipToMemory(const std::string& path)
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
        unz_file_info file_info;
        char filename[MAX_FILENAME];
        if (unzGetCurrentFileInfo(zipfile, &file_info, filename, MAX_FILENAME, NULL, 0, NULL, 0) != UNZ_OK)
        {
            printf("could not read file info\n");
            unzClose(zipfile);
            return {};
        }

        // Check if this entry is a directory or file.
        const size_t filename_length = std::string(filename).size();
        if (filename[filename_length - 1] == dir_delimter)
        {
        }
        else
        {
            // Entry is a file, so extract it.
            std::cout << "  Unzip " << filename << " " << (file_info.compressed_size / (1000 * 1000)) << "MB -> "
                      << (file_info.uncompressed_size / (1000 * 1000)) << "MB" << std::endl;
            if (unzOpenCurrentFile(zipfile) != UNZ_OK)
            {
                printf("could not open file\n");
                unzClose(zipfile);
                return {};
            }

            Unzipfile file;
            file.name = filename;
            file.data.resize(file_info.uncompressed_size);
            size_t current_data = 0;
            {
                ProgressBar bar(std::cout, "  Unzip", file_info.uncompressed_size / (1000), 30, false, 100, "KB");
                int read_block_size = 10000;


                int error = UNZ_OK;
                do
                {
                    error = unzReadCurrentFile(zipfile, file.data.data() + current_data, read_block_size);
                    if (error < 0)
                    {
                        printf("error %d\n", error);
                        unzCloseCurrentFile(zipfile);
                        unzClose(zipfile);
                        return {};
                    }

                    current_data += error;

                    bar.addProgress(error / 1000);
                } while (error > 0);
                SAIGA_ASSERT(current_data == file_info.uncompressed_size);
            }
            out_files.emplace_back(std::move(file));
        }

        unzCloseCurrentFile(zipfile);

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


}  // namespace Saiga
