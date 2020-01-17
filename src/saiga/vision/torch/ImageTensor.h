/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
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


/**
 * Convert an image view to a floating point tensor in the range [0,1].
 * Only uchar images are supported so far.
 * TODO: Add normalizations for other types
 */
template <typename T>
TemplatedImage<T> TensorToImage(at::Tensor tensor)
{
    using ScalarType = typename ImageTypeTemplate<T>::ChannelType;
    constexpr int c  = channels(ImageTypeTemplate<T>::type);
    auto type        = at::typeMetaToScalarType(caffe2::TypeMeta::Make<ScalarType>());



    //    std::cout << tensor.sizes() << std::endl;
    // In pytorch image tensors are usually represented as channel first.
    tensor = tensor.permute({1, 2, 0});

    SAIGA_ASSERT(tensor.dtype() == at::kFloat);



    // Normalize to [0,1]
    if constexpr (std::is_same<ScalarType, unsigned char>::value)
    {
        tensor = 255.f * tensor;
        tensor = tensor.clamp(0, 255);
        tensor = tensor.toType(at::kByte);
    }

    int h = tensor.size(0);
    int w = tensor.size(1);
    ImageView<T> out_view(h, w, tensor.data_ptr<unsigned char>());

    TemplatedImage<T> img(h, w);
    out_view.copyTo(img.getImageView());

    return img;
}

}  // namespace Saiga
