/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/image/image.h"

#include "torch/torch.h"


namespace Saiga
{
/**
 * Convert an image view to a floating point tensor in the range [0,1].
 * Only uchar images are supported so far.
 * TODO: Add normalizations for other types
 */
template <typename T>
at::Tensor ImageViewToTensor(ImageView<T> img)
{
    using ScalarType = typename ImageTypeTemplate<T>::ChannelType;
    constexpr int c  = channels(ImageTypeTemplate<T>::type);


    auto type         = at::typeMetaToScalarType(caffe2::TypeMeta::Make<ScalarType>());
    at::Tensor tensor = torch::from_blob(img.data, {img.h, img.w, c}, type);

    // In pytorch image tensors are usually represented as channel first.
    tensor = tensor.permute({2, 0, 1});

    // Convert to float
    if constexpr (!std::is_same<ScalarType, float>::value)
    {
        tensor = tensor.toType(at::kFloat);
    }

    // Normalize to [0,1]
    if constexpr (std::is_same<ScalarType, unsigned char>::value)
    {
        tensor = (1.f / 255.f) * tensor;
    }

    return tensor;
}

}  // namespace Saiga
