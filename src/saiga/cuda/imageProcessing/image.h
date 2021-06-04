/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"
// cuda helper must currently before arrayview
#include "saiga/core/image/image.h"

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
void CopyImage(ImageView<const T> src, ImageView<T> dst, enum cudaMemcpyKind kind)
{
    SAIGA_ASSERT(src.dimensions() == dst.dimensions());
    CHECK_CUDA_ERROR(
        cudaMemcpy2D(dst.data, dst.pitchBytes, src.data, src.pitchBytes, src.width * sizeof(T), src.height, kind));
}

template <typename T>
void CopyImageAsync(ImageView<const T> src, ImageView<T> dst, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    SAIGA_ASSERT(src.dimensions() == dst.dimensions());
    CHECK_CUDA_ERROR(cudaMemcpy2DAsync(dst.data, dst.pitchBytes, src.data, src.pitchBytes, src.width * sizeof(T),
                                       src.height, kind, stream));
}



template <typename T>
struct CudaImage : public ImageBase
{
    ImageType type = TYPE_UNKNOWN;


    CudaImage() {}

    ~CudaImage() { clear(); }


    CudaImage(const CudaImage<T>& other) { *this = other; }

    CudaImage& operator=(const CudaImage<T>& other)
    {
        create(other.h, other.w);
        CopyImage(other.getConstImageView(), getImageView(), cudaMemcpyDeviceToDevice);
        return *this;
    }

    /**
     * Creates an uninitialized device image with the given parameters.
     */
    CudaImage(int h, int w) : ImageBase(h, w, 0) { create(); }


    CudaImage(ImageView<const T> h_img) { upload(h_img); }

    void clear()
    {
        if (data_)
        {
            cudaFree(data_);
            data_ = nullptr;
        }
    }


    /**
     * Uploads the given host-imageview.
     * Allocates the required memory, if necessary.
     */
    void upload(ImageView<const T> h_img, cudaStream_t stream = 0)
    {
        this->ImageBase::operator=(h_img);
        create();
        CopyImageAsync(h_img, getImageView(), cudaMemcpyHostToDevice, stream);
    }

    // download a host imageview from the device
    void download(ImageView<T> h_img, cudaStream_t stream = 0)
    {
        CopyImageAsync(getConstImageView(), h_img, cudaMemcpyDeviceToHost, stream);
    }

    void download(TemplatedImage<T>& h_img, cudaStream_t stream = 0)
    {
        h_img.create(h, w);
        CopyImageAsync(getConstImageView(), h_img.getImageView(), cudaMemcpyDeviceToHost, stream);
    }

    /**
     * Allocates the device memory from image parameters.
     */
    void create()
    {
        clear();
        size_t pitch = 0;
        cudaMallocPitch(&data_, &pitch, width * sizeof(T), height);
        SAIGA_ASSERT(data_);
        SAIGA_ASSERT(pitch > 0);
        pitchBytes = pitch;
    }

    void create(int h, int w)
    {
        this->width  = w;
        this->height = h;
        create();
    }

    T* data() { return reinterpret_cast<T*>(data_); }
    const T* data() const { return reinterpret_cast<const T*>(data_); }


    ImageView<T> getImageView()
    {
        ImageView<T> res(*this);
        res.data = data();
        return res;
    }

    ImageView<const T> getConstImageView() const
    {
        ImageView<const T> res(*this);
        res.data = data_;
        return res;
    }

    cudaResourceDesc GetResourceDescriptor()
    {
        cudaResourceDesc resDesc = {};
        resDesc.resType          = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();

        resDesc.res.pitch2D.devPtr       = (void*)data();
        resDesc.res.pitch2D.width        = width;
        resDesc.res.pitch2D.height       = height;
        resDesc.res.pitch2D.pitchInBytes = pitchBytes;

        return resDesc;
    }

    cudaTextureObject_t GetTextureObject(cudaTextureAddressMode address_mode = cudaAddressModeWrap,
                                         cudaTextureFilterMode filter_mode   = cudaFilterModePoint,
                                         cudaTextureReadMode read_mode       = cudaReadModeElementType,
                                         int normalized_coords               = 0)
    {
        // Specify texture
        auto resDesc = GetResourceDescriptor();

        // Specify texture object parameters
        cudaTextureDesc texDesc  = {};
        texDesc.addressMode[0]   = address_mode;
        texDesc.addressMode[1]   = address_mode;
        texDesc.filterMode       = filter_mode;
        texDesc.readMode         = read_mode;
        texDesc.normalizedCoords = normalized_coords;

        // Create texture object
        cudaTextureObject_t texObj = 0;
        CHECK_CUDA_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        return texObj;
    }

   protected:
    unsigned char* data_ = nullptr;
};


}  // namespace CUDA
}  // namespace Saiga
