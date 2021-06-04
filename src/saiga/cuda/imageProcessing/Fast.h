/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/all.h"
#include "saiga/core/time/all.h"
#include "saiga/cuda/cudaTimer.h"
#include "saiga/cuda/event.h"
#include "saiga/cuda/pinned_vector.h"
#include "saiga/cuda/stream.h"
#include "saiga/vision/features/Features.h"


namespace Saiga
{
namespace CUDA
{
// Detect fast keypoints using a tile-based approach with 2 thresholds.
// First, per tile, we search with a contrast threshold of highThreshold.
// If no keypoints where found, we search again with lowThreshold.
// The stream passed to detect and download must be the same.
// Download blocks until the download is completed.
//
// Usage:
//
//   // Detect keypoints on GPU image
//   fast->detect(fast_image_view, stream);
//
//   // Download keypoints.
//   N = fast->download(h_keypoints, stream);
class SAIGA_CUDA_API Fast
{
   public:
    Fast(int highThreshold, int lowThreshold, int maxKeypoints = 10000);
    ~Fast();

    // Detect keypoints on the image and store them in a member variable.
    // After detection this function schedules a predictive download so that 'Download' is non-blocking in most cases.
    // Detect is always non-blocking when using a stream different to 0.
    void Detect(Saiga::ImageView<unsigned char> d_image, cudaStream_t stream);


    // Wait until download is completed and then copy into keypoint array.
    // Blocks until download is completed.
    // After this function the keypoints can be used by the CPU.
    // The number N of actual detected keypoints is returned.
    // Make sure to only use the first N keypoints in the output array.
    int Download(Saiga::ArrayView<Saiga::KeyPoint<float>> keypoints, cudaStream_t stream);

    int MaxKeypoints() { return maxKeypoints; }

   private:
    thrust::device_vector<short2> counter_keypoint_location;
    thrust::device_vector<float> keypoint_score;

    Saiga::pinned_vector<short2> h_counter_keypoint_location;
    Saiga::pinned_vector<float> h_keypoint_score;


    Saiga::CUDA::CudaEvent detection_finished;

    int actual_max_keypoints = 0;
    int highThreshold;
    int lowThreshold;
    int maxKeypoints;
};


}  // namespace CUDA
}  // namespace Saiga
