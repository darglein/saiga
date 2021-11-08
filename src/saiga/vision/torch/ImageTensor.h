/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/util/statistics.h"
#include "saiga/core/util/table.h"

#include "TorchHelper.h"
#include "torch/torch.h"

#include <torch/script.h>

namespace Saiga
{
/**
 * Convert an image view to a floating point tensor in the range [0,1].
 * Only uchar images are supported so far.
 * TODO: Add normalizations for other types
 */
template <typename T>
at::Tensor ImageViewToTensor(ImageView<T> img, bool normalize = true)
{
    using ScalarType = typename ImageTypeTemplate<T>::ChannelType;
    constexpr int c  = channels(ImageTypeTemplate<T>::type);


    auto type = at::typeMetaToScalarType(caffe2::TypeMeta::Make<ScalarType>());
    at::Tensor tensor =
        torch::from_blob(img.data, {img.h, img.w, c}, {(long)(img.pitchBytes / sizeof(ScalarType)), c, 1}, type);

    // In pytorch image tensors are usually represented as channel first.
    tensor = tensor.permute({2, 0, 1}).clone();

    if (normalize)
    {
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
    }

    return tensor;
}


/**
 * Convert a tensor to an image view of the given type
 */
template <typename T>
TemplatedImage<T> TensorToImage(at::Tensor tensor)
{
    SAIGA_ASSERT(tensor.defined());
    SAIGA_ASSERT(tensor.dim() == 3 || tensor.dim() == 4);
    tensor           = tensor.clone();
    using ScalarType = typename ImageTypeTemplate<T>::ChannelType;

    if (tensor.dim() == 4)
    {
        SAIGA_ASSERT(tensor.size(0) == 1);
        tensor = tensor.squeeze(0);
    }
    SAIGA_ASSERT(tensor.dim() == 3);
    SAIGA_ASSERT(channels(ImageTypeTemplate<T>::type) == tensor.size(0));

    // In pytorch image tensors are usually represented as channel first.
    tensor = tensor.permute({1, 2, 0});
    tensor = tensor.cpu().contiguous();



    // Convert to byte
    if (tensor.dtype() == torch::kFloat32 && std::is_same<ScalarType, unsigned char>::value)
    {
        tensor = 255.f * tensor;
        tensor = tensor.clamp(0, 255);
        tensor = tensor.toType(at::kByte);
    }

    // SAIGA_ASSERT(tensor.dtype() == torch::kByte);


    int h = tensor.size(0);
    int w = tensor.size(1);

    ImageView<T> out_view(h, w, tensor.stride(0) * sizeof(ScalarType), tensor.data_ptr<ScalarType>());

    TemplatedImage<T> img(h, w);
    out_view.copyTo(img.getImageView());

    return img;
}


/**
 * Save the tensor so it can be loaded from python and C++.
 * For Python loading just use
 *    x = torch.load(path)
 *
 * Python -> c++
 *   std::vector<char> f = get_the_bytes();
 *   torch::IValue x = torch::pickle_load(f);
 */
inline bool SaveTensor(at::Tensor t, const std::string& file)
{
    auto bytes = torch::jit::pickle_save(t);
    if (bytes.size() == 0)
    {
        return false;
    }
    std::ofstream fout(file, std::ios::out | std::ios::binary);
    if (!fout.is_open())
    {
        return false;
    }
    fout.write(bytes.data(), bytes.size());
    fout.close();
    return true;
}


// Converts a list of images into a 2D grid of images.
// Useful for displaying multiple images at once.
//
// Input:
//      image_batch [N, c, h, w]
// Output:
//      [c, new_h, new_w]
inline torch::Tensor ImageBatchToImageGrid(torch::Tensor image_batch)
{
    SAIGA_ASSERT(image_batch.dim() == 4);
    int N = image_batch.size(0);
    int c = image_batch.size(1);
    int h = image_batch.size(2);
    int w = image_batch.size(3);

    int grid_w  = ceil(sqrt(N));
    int grid_h  = ceil(double(N) / grid_w);
    int N_after = grid_w * grid_h;

    if (N < N_after)
    {
        // pad with black images
        auto sizes   = image_batch.sizes().vec();
        sizes[0]     = N_after - N;
        auto padding = torch::zeros({sizes}, image_batch.options());
        image_batch  = torch::cat({image_batch, padding}, 0);
    }

    // [ gh, gw, c, h, w]
    image_batch = image_batch.reshape({grid_h, grid_w, c, h, w});
    // [c, gh, h, gw, w]
    image_batch = image_batch.permute({2, 0, 3, 1, 4});
    // [c, gh * h, gw * w]
    image_batch = image_batch.reshape({c, grid_h * h, grid_w * w});
    return image_batch;
}


// Converts a list of images into a row where all images are placed next to each other.
//
// Input:
//      image_batch [N, c, h, w]
// Output:
//      [1, c, h, N * w]
inline torch::Tensor ImageBatchToImageRow(torch::Tensor image_batch)
{
    SAIGA_ASSERT(image_batch.dim() == 4);
    int N = image_batch.size(0);
    int c = image_batch.size(1);
    int h = image_batch.size(2);
    int w = image_batch.size(3);

    // [c,  h, N, w]
    image_batch = image_batch.permute({1, 2, 0, 3});
    // [1, c, h, N*w]
    image_batch = image_batch.reshape({1, c, h, N * w});
    return image_batch;
}

/**
 * RGB image normalization of a 3 channel float-tensor using the Pytorch standart weights.
 */
inline at::Tensor NormalizeRGB(at::Tensor x)
{
    torch::data::transforms::Normalize<> color_normalize =
        torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    return color_normalize(x);
}

/**
 * Inverse normalization to the function above.
 */
inline at::Tensor UnNormalizeRGB(at::Tensor x)
{
    torch::data::transforms::Normalize<> un_color_normalize1 =
        torch::data::transforms::Normalize<>({-0.485, -0.456, -0.406}, {1, 1, 1});
    torch::data::transforms::Normalize<> un_color_normalize2 =
        torch::data::transforms::Normalize<>({0, 0, 0}, {1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225});
    x = un_color_normalize2(x);
    x = un_color_normalize1(x);
    return x;
}

inline torch::Tensor FilterTensor(Matrix<float, -1, -1> kernel)
{
    return torch::from_blob(kernel.data(), {1, 1, kernel.rows(), kernel.cols()}).clone();
}



inline torch::Tensor Filter2dIndependentChannels(torch::Tensor x, Matrix<float, -1, -1> kernel, int padding)
{
    SAIGA_ASSERT(x.dim() == 4);
    torch::Tensor K = FilterTensor(kernel);
    K               = K.repeat({x.size(1), 1, 1, 1}).to(x.device());
    auto res        = torch::conv2d(x, K, {}, 1, padding, 1, x.size(1));
    return res;
}

// Not very efficient gauss blur, because the kernel is always created on the fly.
// Also the filter is not separated.
inline torch::Tensor GaussBlur(torch::Tensor image, int radius, float sigma, int padding)
{
    SAIGA_ASSERT(image.dim() == 4);
    auto K = FilterTensor(gaussianBlurKernel2d(radius, sigma));
    K      = K.repeat({image.size(1), 1, 1, 1}).to(image.device());
    return torch::conv2d(image, K, {}, 1, padding, 1, image.size(1));
}


}  // namespace Saiga
