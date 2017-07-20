/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/cuda/cudaHelper.h"

#include <cuda_gl_interop.h>
#include <iostream>

struct cudaGraphicsResource;

namespace Saiga {
namespace CUDA{

class SAIGA_GLOBAL Interop{
private:
    unsigned int gl_buffer;
    cudaGraphicsResource* graphic_resource;
    void* device_ptr;
    size_t  size;
    bool mapped;


    GLenum gl_target;
public:
    Interop();
    ~Interop();

    //Registers an OpenGL buffer object. Wrapper for cudaGraphicsGLRegisterBuffer
    void registerBuffer(int glbuffer);

    void unregisterBuffer();

    //Maps graphic resource to be accessed by CUDA. Wrapper for cudaGraphicsMapResources
    void map();

    //Unmap graphics resources. Wrapper for cudaGraphicsUnmapResources
    void unmap();

    //Get an device pointer through which to access a mapped graphics resource. Wrapper for cudaGraphicsResourceGetMappedPointer
    void* getDevicePtr();

    size_t  get_size(){return size;}
    void initImage(unsigned int gl_buffer, GLenum gl_target);
    void mapImage();
};

}
}
