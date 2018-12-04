/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"
#include "saiga/image/image.h"

namespace Saiga
{
namespace CUDA
{
/**
 * Does a cudamemcpy2d between these images.
 * kind is one of the following:
 * cudaMemcpyHostToDevice
 * cudaMemcpyDeviceToHost
 * cudaMemcpyDeviceToDevice
 * cudaMemcpyHostToHost
 */

template <typename T>
void copyImage(ImageView<T> imgSrc, ImageView<T> imgDst, enum cudaMemcpyKind kind)
{
    SAIGA_ASSERT(imgSrc.width == imgDst.width && imgSrc.height == imgDst.height);
    CHECK_CUDA_ERROR(cudaMemcpy2D(imgDst.data, imgDst.pitchBytes, imgSrc.data, imgSrc.pitchBytes,
                                  imgSrc.width * sizeof(T), imgSrc.height, kind));
}



// with these two functions we are able to use CudaImage from cpp files.
SAIGA_GLOBAL void resizeDeviceVector(thrust::device_vector<unsigned char>& v, int size);
SAIGA_GLOBAL void copyDeviceVector(const thrust::device_vector<unsigned char>& src,
                                   thrust::device_vector<unsigned char>& dst);



template <typename T>
struct CudaImage : public ImageBase
{
    thrust::device_vector<unsigned char> vdata;

    CudaImage() {}

    /**
     * Creates an uninitialized device image with the given parameters.
     */
    CudaImage(int h, int w, int p = 0) : ImageBase(h, w, p == 0 ? sizeof(T) * w : p) { create(); }


    CudaImage(const ImageView<T>& h_img) { upload(h_img); }



    /**
     * Uploads the given host-imageview.
     * Allocates the required memory, if necessary.
     */
    void upload(ImageView<T> h_img)
    {
        this->ImageBase::operator=(h_img);
        create();
        copyImage(h_img, getImageView(), cudaMemcpyHostToDevice);
    }

    // download a host imageview from the device
    inline void download(ImageView<T> h_img) { copyImage(getImageView(), h_img, cudaMemcpyDeviceToHost); }

    /**
     * Allocates the device memory from image parameters.
     */
    void create() { resizeDeviceVector(vdata, this->size()); }

    void create(int h, int w, int p = 0)
    {
        this->width      = w;
        this->height     = h;
        this->pitchBytes = p == 0 ? sizeof(T) * w : p;
        create();
    }

    T* data() { return reinterpret_cast<T*>(vdata.data().get()); }


    ImageView<T> getImageView()
    {
        ImageView<T> res(*this);
        res.data = data();
        return res;
    }

    operator ImageView<T>() { return getImageView(); }
};


template <typename T>
void convert(Image& src, CudaImage<T>& dst)
{
    dst.upload(src.getImageView<T>());
}


template <typename T>
void convert(CudaImage<T>& src, Image& dst)
{
    dst.setFormatFromImageView(src);

    //    dst.type = ImageTypeTemplate<T>::type;
    CUDA::copyImage(src, dst.getImageView<T>(), cudaMemcpyDeviceToHost);
}

}  // namespace CUDA
}  // namespace Saiga
