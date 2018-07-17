/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// (c) Seth Heeren 2013
//
// Based on src/opus_demo.c in opus-1.0.2
// License see http://www.opus-codec.org/license/
#include <fstream>

#include "internal/noGraphicsAPI.h"

#ifdef SAIGA_USE_OPUS
#include "saiga/sound/OpusCodec.h"

int opusTest(int argc, char *argv[])
{
    if(argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input> <output>\n";
        return 255;
    }

    std::basic_ifstream<char> fin (argv[1], std::ios::binary);
    std::basic_ofstream<char> fout(argv[2], std::ios::binary);

    if(!fin)  throw std::runtime_error("Could not open input file");
    if(!fout) throw std::runtime_error("Could not open output file");

    try
    {
        COpusCodec codec(48000, 1);

        size_t frames = 0;
        while(codec.decode_frame(fin).size())
        {
            frames++;
        }

        std::cout << "Successfully decoded " << frames << " frames\n";
    }
    catch(OpusErrorException const& e)
    {
        //std::cerr << "OpusErrorException: " << e.what() << "\n";
        return 255;
    }
    return 0;
}
#endif
