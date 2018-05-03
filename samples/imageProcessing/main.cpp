/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/crash.h"


#include "saiga/image/image.h"
#include "saiga/image/templatedImage.h"

using namespace Saiga;

int main(int argc, char *argv[]) {

    catchSegFaults();


    {
        // Test:
        // Read, modify, write a png image.
        TemplatedImage<ucvec3> img("textures/redie.png");
        SAIGA_ASSERT(img.type == UC3);
        ImageView<ucvec3> vimg = img.getImageView();
        vimg.setChannel(0,0);
        vimg.setChannel(1,0);

        auto img2 = img;
        img2.save("debug/blue.png");

        // Create a grayscale image from the blue channel
        Image imggray(img.height,img.width,UC1);
        for(int i = 0; i < img.height; ++i)
        {
            for(int j = 0; j < img.width; ++j)
            {
                imggray.at<unsigned char>(i,j) = img.at<ucvec3>(i,j)[2];
            }
        }
        imggray.save("debug/blue_gray.png");
    }

#ifdef SAIGA_USE_FREEIMAGE
    {
        // Test:
        // Read, modify, write a jpg image.
        Image img("textures/redie.jpg");
        ImageView<ucvec3> vimg = img.getImageView<ucvec3>();
        vimg.setChannel(0,0);
        vimg.setChannel(1,0);
        img.save("debug/blue.jpg");
    }
#endif
}
