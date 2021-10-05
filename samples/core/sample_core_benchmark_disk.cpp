/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/Thread/omp.h"
using namespace Saiga;

int sizeMB         = 1000 * 1;
size_t blockSize   = sizeMB * 1000UL * 1000UL;
double blockSizeGB = (blockSize / (1000.0 * 1000.0 * 1000.0));


void write(const std::string& file, const std::vector<char>& data)
{
    FILE* dest = fopen(file.c_str(), "wb");
    if (dest == 0)
    {
        return;
    }

    auto written = fwrite(data.data(), 1, data.size(), dest);
    SAIGA_ASSERT(written == data.size());

    fclose(dest);
}

void write2(const std::string& file, const std::vector<char>& data)
{
    std::ofstream is(file, std::ios::binary | std::ios::out);
    if (!is.is_open())
    {
        std::cout << "File not found " << file << std::endl;
        return;
    }

    is.write(data.data(), data.size());
    is.close();
}



int main(int, char**)
{
    catchSegFaults();


    std::vector<char> data(blockSize);

    {
        SAIGA_BLOCK_TIMER();
        write2("test.dat", data);
    }
    std::cout << "Done." << std::endl;

    return 0;
}
