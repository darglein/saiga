#pragma once

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/imageProcessing/imageView.h"
#include "saiga/image/image.h"

namespace Saiga {
namespace CUDA {



template<typename T>
void copyImage(ImageView<T> imgSrc, ImageView<T> imgDst, enum cudaMemcpyKind kind){
    cudaMemcpy2D(imgDst.data,imgDst.pitchBytes,imgSrc.data,imgSrc.pitchBytes,imgSrc.width*sizeof(T),imgSrc.height,kind);
}

//creates a cpu Image from a deviec imageview
template<typename T>
inline
Image deviceImageViewToImage(ImageView<T> img){
    ImageFormat imageFormat;
    if (typeid(T) == typeid(float)){
        imageFormat = ImageFormat(1,32,ImageElementFormat::FloatingPoint);
    }else if(typeid(T) == typeid(uchar4)){
        imageFormat = ImageFormat(4,8,ImageElementFormat::UnsignedNormalized);
    }else if(typeid(T) == typeid(uchar3)){
        imageFormat = ImageFormat(3,8,ImageElementFormat::UnsignedNormalized);
    }
    Image h_img(imageFormat,img.width,img.height,img.pitchBytes,0);
    copyImage(img,h_img.getImageView<T>(),cudaMemcpyDeviceToHost);
    return h_img;
}


SAIGA_GLOBAL void resizeDeviceVector(thrust::device_vector<uint8_t>& v, int size);

//supported types:
//float, uchar3, uchar4
template<typename T>
struct CudaImage : public ImageView<T>{
    thrust::device_vector<uint8_t> v;

    CudaImage(){}

    CudaImage(int w, int h , int p)
        : ImageView<T>(w,h,p,0) {
        resizeDeviceVector(v,this->size());
        this->data = thrust::raw_pointer_cast(v.data());
    }

    CudaImage(int w, int h)
        : ImageView<T>(w,h,0) {
//        v.resize(this->size());
        resizeDeviceVector(v,this->size());
        this->data = thrust::raw_pointer_cast(v.data());
    }


    CudaImage(Image& h_img) : CudaImage<T>(h_img.width,h_img.height,h_img.getBytesPerRow()){
#if !defined(SAIGA_RELEASE)
        if (typeid(T) == typeid(float)){
            SAIGA_ASSERT(h_img.Format().getChannels() == 1
                         && h_img.Format().getElementFormat() == ImageElementFormat::FloatingPoint
                         && h_img.Format().getBitDepth() == 32);
        }else if(typeid(T) == typeid(uchar4)){
            SAIGA_ASSERT(h_img.Format().getChannels() == 4
                         && h_img.Format().getElementFormat() == ImageElementFormat::UnsignedNormalized
                         && h_img.Format().getBitDepth() == 8);
        }else if(typeid(T) == typeid(uchar3)){
            SAIGA_ASSERT(h_img.Format().getChannels() == 3
                         && h_img.Format().getElementFormat() == ImageElementFormat::UnsignedNormalized
                         && h_img.Format().getBitDepth() == 8);
        }
#endif
        copyImage(h_img.getImageView<T>(),*this,cudaMemcpyHostToDevice);
    }

    inline
    operator Image(){
        return deviceImageViewToImage(*this);
    }

    //copy and swap idiom
    //http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
    CudaImage(CudaImage const& other) : ImageView<T>(other){
        v = other.v;
        this->data = thrust::raw_pointer_cast(v.data());
    }

    CudaImage& operator=(CudaImage other){
        swap(*this, other);
        return *this;
    }

    template<typename T2>
    friend void swap(CudaImage<T2>& first, CudaImage<T2>& second);
};

template<typename T>
inline
void swap(CudaImage<T> &first, CudaImage<T> &second)
{
    using std::swap;
    first.v.swap(second.v);
    swap(first.width,second.width);
    swap(first.height,second.height);
    swap(first.pitchBytes,second.pitchBytes);
    swap(first.data,second.data);
}

}
}
