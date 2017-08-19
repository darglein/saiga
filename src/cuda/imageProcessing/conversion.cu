#include "saiga/cuda/imageProcessing/conversion.h"

namespace Saiga {
namespace CUDA {

template<int BLOCK_W, int BLOCK_H>
__global__
static void d_convertRGBtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, unsigned char alpha)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int xp = blockIdx.x*BLOCK_W + tx;
    const int yp = blockIdx.y*BLOCK_H + ty;

    if(xp >= src.width || yp >= src.height)
        return;

    uchar3 v3 = src(xp,yp);
    uchar4 v4;
    v4.x = v3.x;
    v4.y = v3.y;
    v4.z = v3.z;
    v4.w = alpha;
    dst(xp,yp) = v4;
}


void convertRGBtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, unsigned char alpha){
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    const int BLOCK_W = 16;
    const int BLOCK_H = 16;
    dim3 blocks(iDivUp(src.width, BLOCK_W), iDivUp(src.height, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_convertRGBtoRGBA<BLOCK_W,BLOCK_H> <<<blocks, threads>>>(src,dst,alpha);
}

template<int BLOCK_W, int BLOCK_H>
__global__
static void d_convertRGBAtoGrayscale(ImageView<uchar4> src, ImageView<float> dst)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int xp = blockIdx.x*BLOCK_W + tx;
    const int yp = blockIdx.y*BLOCK_H + ty;

    if(xp >= src.width || yp >= src.height)
        return;

    uchar4 u = src(xp,yp);
    vec3 uv = vec3(u.x,u.y,u.z) * (1.0f / 255.0f);
    vec3 conv(0.2126,0.7152,0.0722);
    float v =  dot(uv,conv);
    dst(xp,yp) = v;
}

void convertRGBAtoGrayscale(ImageView<uchar4> src, ImageView<float> dst){
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    const int BLOCK_W = 16;
    const int BLOCK_H = 16;
    dim3 blocks(iDivUp(src.width, BLOCK_W), iDivUp(src.height, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_convertRGBAtoGrayscale<BLOCK_W,BLOCK_H> <<<blocks, threads>>>(src,dst);
}

}
}


