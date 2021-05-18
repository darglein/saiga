/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"
#include "saiga/opengl/opengl.h"

#include <iostream>

#include <cuda_gl_interop.h>


namespace Saiga
{
namespace CUDA
{
class Interop
{
   public:
    ~Interop() { unregisterBuffer(); }

    // Registers an OpenGL buffer object. Wrapper for cudaGraphicsGLRegisterBuffer
    void registerBuffer(int glbuffer)
    {
        this->gl_buffer = glbuffer;
        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&graphic_resource, glbuffer, cudaGraphicsRegisterFlagsNone));
    }

    void unregisterBuffer()
    {
        if (mapped) unmap();
        if (graphic_resource) CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(graphic_resource));
        graphic_resource = nullptr;
    }

    // Maps graphic resource to be accessed by CUDA. Wrapper for cudaGraphicsMapResources
    void map()
    {
        // map OpenGL buffer object for writing from CUDA
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &graphic_resource, 0));
        CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, graphic_resource));

        mapped = true;
    }

    // Unmap graphics resources. Wrapper for cudaGraphicsUnmapResources
    void unmap()
    {
        // unmap buffer object
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &graphic_resource, 0));
        mapped = false;
    }

    // Get an device pointer through which to access a mapped graphics resource. Wrapper for
    // cudaGraphicsResourceGetMappedPointer
    void* getDevicePtr()
    {
        return device_ptr;
    }

    size_t get_size() { return size; }

    //     Register an OpenGL texture or renderbuffer object.
    void initImage(unsigned int gl_buffer, GLenum gl_target)
    {
        this->gl_buffer = gl_buffer;
        this->gl_target = gl_target;
        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&graphic_resource, gl_buffer, gl_target,
                                                     cudaGraphicsRegisterFlagsSurfaceLoadStore));
    }
    void mapImage()
    {
        // map OpenGL buffer object for writing from CUDA
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &graphic_resource, 0));
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, graphic_resource, 0, 0));

        //    std::cout<<array<<std::endl;
        device_ptr = array;
        mapped     = true;
    }

    cudaArray_t array;
   private:
    unsigned int gl_buffer;
    cudaGraphicsResource* graphic_resource = nullptr;
    void* device_ptr;
    size_t size;
    bool mapped = false;


    GLenum gl_target;
};

}  // namespace CUDA
}  // namespace Saiga
