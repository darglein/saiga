/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"
#include "saiga/opengl/opengl.h"

#include <iostream>
#include <utility>

#include <cuda_gl_interop.h>


namespace Saiga
{
namespace CUDA
{
// Owns a registered OpenGL resource, so it can be moved but not copied (a copy would unregister it a second time).
class Interop
{
   public:
    Interop() = default;
    ~Interop() { unregisterBuffer(); }

    Interop(const Interop&)            = delete;
    Interop& operator=(const Interop&) = delete;

    Interop(Interop&& other) noexcept { *this = std::move(other); }
    Interop& operator=(Interop&& other) noexcept
    {
        if (this != &other)
        {
            unregisterBuffer();
            array            = other.array;
            gl_buffer        = other.gl_buffer;
            graphic_resource = other.graphic_resource;
            device_ptr       = other.device_ptr;
            size             = other.size;
            mapped           = other.mapped;
            gl_target        = other.gl_target;

            other.graphic_resource = nullptr;
            other.device_ptr       = nullptr;
            other.array            = nullptr;
            other.mapped           = false;
        }
        return *this;
    }

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
    void* getDevicePtr() { return device_ptr; }

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

    cudaArray_t array = nullptr;

   private:
    unsigned int gl_buffer                 = 0;
    cudaGraphicsResource* graphic_resource = nullptr;
    void* device_ptr                       = nullptr;
    size_t size                            = 0;
    bool mapped                            = false;


    GLenum gl_target = GL_NONE;
};

}  // namespace CUDA
}  // namespace Saiga
