/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/util/table.h"
#include "saiga/vision/torch/torch.h"


namespace Saiga
{
// Cuts of 'border_size' pixels from each image border
inline torch::Tensor CenterCrop2D(torch::Tensor x, int border_size)
{
    // [b, c, h, w]
    SAIGA_ASSERT(x.dim() == 4);
    return x.slice(2, border_size, x.size(2) - 2 * border_size).slice(3, border_size, x.size(3) - 2 * border_size);
}

// Center crops the tensor x to target size.
// Used for example in unets if the input is not a power of 2, because we lose some pixels after downsampling
inline torch::Tensor CenterCrop2D(torch::Tensor x, std::vector<int64_t> target_size)
{
    // [b, c, h, w]
    SAIGA_ASSERT(x.dim() == 4);

    int diff_h = int(x.size(2) - target_size[2]) / 2;
    int diff_w = int(x.size(3) - target_size[3]) / 2;

    return x.slice(2, diff_h, diff_h + target_size[2]).slice(3, diff_w, diff_w + target_size[3]);
}

// emplace 'src' into the center of 'target'
inline torch::Tensor CenterEmplace(torch::Tensor src, torch::Tensor target)
{
    auto src_size = src.sizes();
    // [b, c, h, w]
    SAIGA_ASSERT(src.dim() == 4);

    int diff_h = int(target.size(2) - src_size[2]) / 2;
    int diff_w = int(target.size(3) - src_size[3]) / 2;

    auto ref_copy = target.clone();

    ref_copy.slice(2, diff_h, diff_h + src_size[2]).slice(3, diff_w, diff_w + src_size[3]) = src;
    return ref_copy;
}

inline std::vector<int64_t> IndexToCoordinate(int64_t index, std::vector<int64_t> sizes)
{
    std::vector<int64_t> result(sizes.size());
    for (int d = sizes.size() - 1; d >= 0; --d)
    {
        result[d] = index % sizes[d];
        index /= sizes[d];
    }
    return result;
}

// input:
//      image [3, ...]
//      mask  [1, ...]
// output:
//      image [3, ...]
inline torch::Tensor SetMaskToColor(torch::Tensor image, torch::Tensor mask, vec3 color)
{
    if (mask.dim() == image.dim() - 1)
    {
        mask = mask.unsqueeze(0);
    }
    mask          = mask.to(image.dtype()).to(image.device());
    auto inv_mask = 1 - mask;

    // set mask to 0
    image = image * inv_mask;

    std::vector<int64_t> color_sizes(image.dim(), 1);
    color_sizes[0] = 3;
    torch::Tensor color_tensor =
        torch::from_blob(color.data(), color_sizes, torch::kFloat).to(image.dtype()).to(image.device());

    // add color to image
    image = image + mask * color_tensor;
    return image;
}

inline std::string TensorInfo(at::Tensor t)
{
    torch::NoGradGuard ngg;
    std::stringstream strm;
    if (!t.defined())
    {
        strm << "[undefined tensor]";
        return strm.str();
    }
    auto device    = t.device();
    auto requ_grad = t.requires_grad();
    void* ptr      = t.data_ptr();

    if (t.numel() == 0)
    {
        strm << "[empty tensor " << t.sizes() << "]";
        return strm.str();
    }

    auto type = t.dtype();

    // below 1MB size
    if (t.numel() * t.element_size() < int64_t(1000) * 1000)
    {
        if (t.dtype() == at::kFloat || t.dtype() == at::kHalf)
        {
            t = t.to(torch::kDouble);
        }
    }

    double mi   = t.min().item().toDouble();
    double ma   = t.max().item().toDouble();
    double sum  = t.sum().item().toDouble();
    double mean = sum / t.numel();
    double sdev = 0;
    if (t.scalar_type() == torch::kFloat32)
    {
        sdev = t.std().item().toDouble();
    }

    if (t.dim() == 0 && t.numel() == 1)
    {
        strm << "Scalar Tensor " << type << " " << device << " req-grad " << requ_grad << " Value: " << mi;
    }
    else
    {
        strm << "Tensor " << t.sizes() << " " << t.strides() << " " << type << " " << device << " Min/Max " << mi << " "
             << ma << " Mean " << mean << " Sum " << sum << " sdev " << sdev << " req-grad " << requ_grad << " ptr "
             << ptr;
    }

    return strm.str();
}

/**
 * Writes some information of the given tensor to std::cout.
 */
inline void PrintTensorInfo(at::Tensor t)
{
    std::cout << TensorInfo(t) << std::endl;
}

#ifndef TINY_TORCH
inline void PrintModelParams(torch::nn::Module module)
{
    Table tab({40, 25, 10, 15});
    size_t sum = 0;

    tab << "Name"
        << "Params"
        << "Params"
        << "Sum";
    for (auto& t : module.named_parameters())
    {
        size_t local_sum = 1;
        for (auto i : t.value().sizes())
        {
            local_sum *= i;
        }
        sum += local_sum;
        std::stringstream strm;
        strm << t.value().sizes();
        tab << t.key() << strm.str() << local_sum << sum;
    }
    std::cout << std::endl;
}

inline void PrintModelParamsCompact(torch::nn::Module module)
{
    size_t sum = 0;

    for (auto& t : module.named_parameters())
    {
        size_t local_sum = 1;
        for (auto i : t.value().sizes())
        {
            local_sum *= i;
        }
        sum += local_sum;
    }
    std::cout << "Total Model Params: " << sum << std::endl;
}

inline torch::nn::AnyModule NormFromString(const std::string& str, int channels)
{
    if (str == "bn")
    {
        return torch::nn::AnyModule(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels).momentum(0.01)));
    }
    else if (str == "bn_no_stat")
    {
        return torch::nn::AnyModule(
            torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(channels).track_running_stats(false)));
    }
    else if (str == "in")
    {
        return torch::nn::AnyModule(torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(channels)));
    }
    else if (str == "id")
    {
        return torch::nn::AnyModule(torch::nn::Identity());
    }
    else
    {
        SAIGA_EXIT_ERROR("Unknown norm " + str);
        return {};
    }
}

inline torch::nn::AnyModule ActivationFromString(const std::string& str, float beta = 2.f)
{
    if (str == "id" || str == "none" || str.empty())
    {
        return torch::nn::AnyModule(torch::nn::Identity());
    }
    else if (str == "sigmoid")
    {
        return torch::nn::AnyModule(torch::nn::Sigmoid());
    }
    else if (str == "tanh")
    {
        return torch::nn::AnyModule(torch::nn::Tanh());
    }
    else if (str == "elu")
    {
        return torch::nn::AnyModule(torch::nn::ELU());
    }
    else if (str == "relu")
    {
        return torch::nn::AnyModule(torch::nn::ReLU());
    }
    else if (str == "silu")
    {
        return torch::nn::AnyModule(torch::nn::SiLU());
    }
    else if (str == "softplus")
    {
        return torch::nn::AnyModule(torch::nn::Softplus(torch::nn::SoftplusOptions().beta(beta)));
    }
    else if (str == "softplus2")
    {
        return torch::nn::AnyModule(torch::nn::Softplus(torch::nn::SoftplusOptions().beta(2)));
    }
    else if (str == "softplus4")
    {
        return torch::nn::AnyModule(torch::nn::Softplus(torch::nn::SoftplusOptions().beta(4)));
    }
    else
    {
        SAIGA_EXIT_ERROR("Unknown activation " + str);
        return {};
    }
}


inline torch::nn::AnyModule Pooling2DFromString(const std::string& str, torch::ExpandingArray<2> kernel_size)
{
    if (str == "average")
    {
        return torch::nn::AnyModule(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(kernel_size)));
    }
    else if (str == "max")
    {
        return torch::nn::AnyModule(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(kernel_size)));
    }
    else
    {
        SAIGA_EXIT_ERROR("Unknown activation " + str);
        return {};
    }
}


//  After each epoch we mulitply the learning rate by the decay
inline double ExponentialDecay(int epoch_id, int max_epochs, double decay)
{
    SAIGA_ASSERT(decay >= 0 && decay <= 1);
    return decay;
}

// After <step_size> steps the learning rate is multiplied by decay
inline double SteppedExponentialDecay(int epoch_id, int max_epochs, int step_size, double decay)
{
    SAIGA_ASSERT(epoch_id > 0);
    SAIGA_ASSERT(decay >= 0 && decay <= 1);

    if (epoch_id % step_size == 0)
    {
        return decay;
    }
    else
    {
        return 1;
    }
}


// Set the LR of all param groups in this optimizer
template <typename OptionsType>
void SetLR(torch::optim::Optimizer* optimizer, double lr)
{
    for (auto& pg : optimizer->param_groups())
    {
        auto opt = dynamic_cast<OptionsType*>(&pg.options());
        SAIGA_ASSERT(opt);
        opt->lr() = lr;
    }
}


inline void UpdateLR(torch::optim::Optimizer* optimizer, double factor)
{
    for (auto& pg : optimizer->param_groups())
    {
        auto opt_adam = dynamic_cast<torch::optim::AdamOptions*>(&pg.options());
        if (opt_adam)
        {
            opt_adam->lr() = opt_adam->lr() * factor;
        }

        auto opt_sgd = dynamic_cast<torch::optim::SGDOptions*>(&pg.options());
        if (opt_sgd)
        {
            opt_sgd->lr() = opt_sgd->lr() * factor;
        }

        auto opt_rms = dynamic_cast<torch::optim::RMSpropOptions*>(&pg.options());
        if (opt_rms)
        {
            opt_rms->lr() = opt_rms->lr() * factor;
        }
    }
}
#endif

}  // namespace Saiga
