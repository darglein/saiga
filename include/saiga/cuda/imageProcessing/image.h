/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/imageProcessing/imageView.h"
#include "saiga/image/image.h"

namespace Saiga {
namespace CUDA {



template<typename T>
void copyImage(ImageView<T> imgSrc, ImageView<T> imgDst, enum cudaMemcpyKind kind){
    CHECK_CUDA_ERROR(cudaMemcpy2D(imgDst.data,imgDst.pitchBytes,imgSrc.data,imgSrc.pitchBytes,imgSrc.width*sizeof(T),imgSrc.height,kind));
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
        create();
    }

    CudaImage(int w, int h)
        : ImageView<T>(w,h,0) {
        create();
    }


    CudaImage(ImageView<T> h_img) {
        upload(h_img);
    }

    //upload a host imageview to the device
    inline
    void upload(ImageView<T> h_img){
        this->ImageView<T>::operator=(h_img);
        create();
        copyImage(h_img,*this,cudaMemcpyHostToDevice);
    }

    //download a host imageview from the device
    inline
    void download(ImageView<T> h_img){
        copyImage(*this,h_img,cudaMemcpyDeviceToHost);
    }

    inline void create(){
        resizeDeviceVector(v,this->size());
        this->data = thrust::raw_pointer_cast(v.data());
    }

    inline void create(int w, int h){
        create(w,h,w*sizeof(T));
    }


    inline void create(int w, int h , int p){
        this->width = w;
        this->height = h;
        this->pitchBytes = p;
        create();
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


template<typename T>
void convert(Image& src, CudaImage<T>& dst){
    dst.upload(src.getImageView<T>());
}


template<typename T>
void convert(CudaImage<T>& src, Image& dst){
    dst.setFormatFromImageView(src);
    CUDA::copyImage(src,dst.getImageView<T>(),cudaMemcpyDeviceToHost);
}

}
}
