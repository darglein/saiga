#pragma once


#include "saiga/core/image/imageView.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/vision/features/Features.h"

#include <vector>


namespace Saiga
{
namespace CUDA
{
class SAIGA_CUDA_API ORB
{
   public:
    ORB();

    void ComputeDescriptors(cudaTextureObject_t tex, Saiga::ImageView<unsigned char> image,
                            Saiga::ArrayView<Saiga::KeyPoint<float> > _keypoints,
                            Saiga::ArrayView<Saiga::DescriptorORB> _descriptors, cudaStream_t stream);

    void ComputeAngles(cudaTextureObject_t tex, Saiga::ImageView<unsigned char> image,
                       Saiga::ArrayView<Saiga::KeyPoint<float> > keypoints, int minBorderX, int minBorderY, int octave,
                       int size, cudaStream_t stream);
};



}  // namespace CUDA
}  // namespace Saiga
