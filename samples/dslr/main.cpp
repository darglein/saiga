/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/util/crash.h"
#include "saiga/gphoto2/gphoto.h"

using namespace Saiga;

int main( int argc, char* args[] )
{
    Saiga::GPhoto dslr;
    SAIGA_ASSERT(dslr.isOpenend());
    TemplatedImage<ucvec3> dimg;
    while(true)
    {
        auto img = dslr.tryGetImage();
        if(img)
        {
            img->saveJpg("dslr");
            img->saveRaw("dslr");
            cout << "saved." << endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}
