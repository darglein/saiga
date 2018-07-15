#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_wrapper.h"

namespace Saiga {


bool loadImageSTB(const std::string &path, Image &img)
{
    int x,y,n;
    unsigned char *data = stbi_load(path.c_str(), &x, &y, &n, 0);
    int pitch = x * n;
    if(!data)
        return false;

    ImageType it;

    switch(n)
    {
    case 1:
        it = UC1;
        break;
    case 2:
        it = UC2;
        break;
    case 3:
        it = UC3;
        break;
    case 4:
        it = UC4;
        break;
    default:
        SAIGA_ASSERT(0);
        break;
    }

    img.create(y,x,it);

    for(int i = 0; i < y; ++i)
    {
        auto dst = img.rowPtr(i);
        auto src = data + i * pitch;
        memcpy(dst,src,pitch);
    }

    stbi_image_free(data);
    return true;
}



}
