
#ifdef _MSC_VER
#include <windows.h>
#endif

#include "saiga/cuda/interop.h"
#include "saiga/cuda/cudaHelper.h"

namespace CUDA{



Interop::Interop():mapped(false){

}
Interop::~Interop(){
    if(mapped)
        unmap();
}

void Interop::registerBuffer(int glbuffer){
    this->gl_buffer = glbuffer;
    CHECK_CUDA_ERROR( cudaGraphicsGLRegisterBuffer(&graphic_resource, glbuffer, cudaGraphicsRegisterFlagsNone));
}

void Interop::unregisterBuffer()
{
    if(mapped)
        unmap();
    CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(graphic_resource));
}



void Interop::map(){
    // map OpenGL buffer object for writing from CUDA
    CHECK_CUDA_ERROR( cudaGraphicsMapResources(1, &graphic_resource, 0));

    mapped = true;
}



void Interop::unmap(){
    // unmap buffer object
    CHECK_CUDA_ERROR( cudaGraphicsUnmapResources(1, &graphic_resource, 0));
    mapped = false;
}

void *Interop::getDevicePtr(){
    CHECK_CUDA_ERROR( cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, graphic_resource));
    return device_ptr;
}





void Interop::mapImage(){
    // map OpenGL buffer object for writing from CUDA
    cudaArray *array;// = (cudaArray *)device_ptr;
    CHECK_CUDA_ERROR( cudaGraphicsMapResources(1, &graphic_resource, 0));
    CHECK_CUDA_ERROR( cudaGraphicsSubResourceGetMappedArray(&array, graphic_resource, 0,0));

//    std::cout<<array<<std::endl;
    device_ptr = array;
    mapped = true;
}

void Interop::initImage(unsigned int gl_buffer, GLenum gl_target)
{
    this->gl_buffer = gl_buffer;
    this->gl_target = gl_target;
    CHECK_CUDA_ERROR( cudaGraphicsGLRegisterImage(&graphic_resource, gl_buffer,gl_target, cudaGraphicsRegisterFlagsSurfaceLoadStore));

}


}
