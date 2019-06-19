/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/util/crash.h"
#include "saiga/extra/gphoto2/gphoto.h"

using namespace Saiga;

int main(int argc, char* args[])
{
    std::cout << sizeof(bool) << std::endl;
    return 0;

    Saiga::GPhoto dslr;
    TemplatedImage<ucvec3> dimg;
    while (true)
    {
        auto img = dslr.tryGetImage();
        if (img)
        {
            img->saveJpg("dslr");
            img->saveRaw("dslr");
            std::cout << "saved." << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}
