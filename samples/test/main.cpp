#include "saiga/util/crash.h"


#include "saiga/cuda/cudaHelper.h"
#include "saiga/opengl/opengl.h"
#include "saiga/cuda/cusparseHelper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/tests/test.h"



#include "saiga/geometry/clipping.h"

#include "saiga/image/freeimage.h"
//#include <FreeImagePlus.h>
//#include "saiga/opengl/texture/textureLoader.h"

int main(int argc, char *argv[]) {

    catchSegFaults();


    cout << "asdf" << endl;



    Image img;
    ImageMetadata metaData;
//    FIP::load("textures/test.CR2",img,&metaData);
//    FIP::load("textures/8D0A1390.jpg",img,&metaData);
    auto ret = FIP::load("textures/A002C015_130612_R4MX.848124.tif",img,&metaData);
    SAIGA_ASSERT(ret);
    img.to8bitImage();
    ret = FIP::save("textures/test.jpg",img);
    SAIGA_ASSERT(ret);
//    img.flipY();

//    FIP::save("textures/test_flipped.jpg",img);
//    img.flipY();
//    FIP::save("textures/test_flipped2.jpg",img);
    cout << metaData << endl;

}
