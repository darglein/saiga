/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/image/floatTexels.h"
#include "saiga/image/image.h"
#include "saiga/image/templatedImage.h"
#include "saiga/framework/framework.h"
#include "saiga/time/timer.h"
#include "saiga/opencv/opencv.h"

using namespace Saiga;


void testReadWriteImage()
{
    std::string file = Image::searchPathes.getFile("redie.png");
    cv::Mat3b cvimg;
    {
        ScopedTimerPrint t("Opencv Load png");
        cvimg = cv::imread(file);
    }
    {
        ScopedTimerPrint t("Opencv Save png");
        cv::imwrite("redie3.png",cvimg);
    }

    TemplatedImage<ucvec3> img;
    {
        ScopedTimerPrint t("Saiga Load png");
        img.load("redie.png");
    }
    {
        ScopedTimerPrint t("Saiga Save png");
        img.save("redie2.png");
    }



}


int main(int argc, char *argv[])
{
    SaigaParameters sp;
    initSample(sp);
    initSaiga(sp);

    testReadWriteImage();
}
