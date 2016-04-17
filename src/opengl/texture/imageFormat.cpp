#include "saiga/opengl/texture/imageFormat.h"
#include "saiga/opengl/texture/image2.h"

void asksdfkg(){
    Texel<2,8,ImageFormat::SignedIntegral> t;
    t.r = 0.5f;
    t.g = 1.0f;

//    TemplatedImage<2,8,ImageFormat::SignedIntegral,false> i(5,5);
    TemplatedImage<3,8,ImageFormat::UnsignedNormalized,false> i(5,5);
    i.create();
    auto texel = i.getTexel(3,4);
    i.flipRB();
}
