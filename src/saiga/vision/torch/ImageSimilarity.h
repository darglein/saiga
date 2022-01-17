/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "ImageTensor.h"
#include "TorchHelper.h"

#include <torch/script.h>

namespace Saiga
{
// Peak signal-to-noise ratio (PSNR) is an engineering term for the ratio between the maximum possible power of a signal
// and the power of corrupting noise that affects the fidelity of its representation. Because many signals have a very
// wide dynamic range, PSNR is usually expressed as a logarithmic quantity using the decibel scale.
// https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
//
class PSNRImpl : public torch::nn::Module
{
   public:
    PSNRImpl(float min_value = 0, float max_value = 1) : min_value(min_value), max_value(max_value) {}
    torch::Tensor forward(torch::Tensor input, torch::Tensor target)
    {
        float distance = max_value - min_value;

        input  = torch::clamp(input, min_value, max_value);
        target = torch::clamp(target, min_value, max_value);

        return 10 * torch::log10((distance * distance) / torch::mse_loss(input, target));
    }

   private:
    float min_value, max_value;
};
TORCH_MODULE(PSNR);


// Structured Similarity Index (SSIM)
// https://en.wikipedia.org/wiki/Structural_similarity
// High value means high similarity
//
// To use it as a loss function you should do
//      loss_ssim = 1 - SSIM(output,target)
//
// This code is a Caffe implementation of
// https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/ssim.html
class SSIMImpl : public torch::nn::Module
{
   public:
    SSIMImpl(int radius = 2, float max_value = 1)
    {
        kernel_raw = FilterTensor(gaussianBlurKernel2d(radius, 1.5f));
        C1         = pow(0.01 * max_value, 2);
        C2         = pow(0.03 * max_value, 2);
        padding    = radius;
        register_buffer("kernel_raw", kernel_raw);
    }
    torch::Tensor forward(torch::Tensor img1, torch::Tensor img2)
    {
        SAIGA_ASSERT(img1.dim() == 4);
        SAIGA_ASSERT(img2.dim() == 4);
        auto kernel = kernel_raw.repeat({img2.size(1), 1, 1, 1});

        auto mu1 = torch::conv2d(img1, kernel, {}, 1, padding, 1, img1.size(1));
        auto mu2 = torch::conv2d(img2, kernel, {}, 1, padding, 1, img2.size(1));

        auto mu11 = mu1 * mu1;
        auto mu22 = mu2 * mu2;
        auto mu12 = mu1 * mu2;

        auto sigma11 = torch::conv2d(img1 * img1, kernel, {}, 1, padding, 1, img1.size(1)) - mu11;
        auto sigma22 = torch::conv2d(img2 * img2, kernel, {}, 1, padding, 1, img1.size(1)) - mu22;
        auto sigma12 = torch::conv2d(img1 * img2, kernel, {}, 1, padding, 1, img1.size(1)) - mu12;

        auto ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu11 + mu22 + C1) * (sigma11 + sigma22 + C2));

        return ssim_map.mean();
    }

   private:
    torch::Tensor kernel_raw;
    int padding;

    float C1, C2;
};
TORCH_MODULE(SSIM);


// Structured Similarity Index (SSIM)
// https://en.wikipedia.org/wiki/Structural_similarity
// High value means high similarity
//
// To use it as a loss function you should do
//      loss_ssim = 1 - SSIM(output,target)
//
// This code is a Caffe implementation of
// https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/ssim.html
class SSIM3DImpl : public torch::nn::Module
{
   public:
    SSIM3DImpl(int radius = 2, float max_value = 1)
    {
        int filter_size = radius * 2 + 1;
        auto t          = FilterTensor(gaussianBlurKernel1d<float>(radius, 1.5)).squeeze(0).squeeze(0);
        auto t2d        = t.mm(t.t());
        auto t3d        = t.mm(t2d.reshape({1, -1})).reshape({filter_size, filter_size, filter_size});

        kernel_raw = t3d.unsqueeze(0).unsqueeze(0);
        C1         = pow(0.01 * max_value, 2);
        C2         = pow(0.03 * max_value, 2);
        padding    = radius;
        register_buffer("kernel_raw", kernel_raw);
    }
    torch::Tensor forward(torch::Tensor img1, torch::Tensor img2)
    {
        SAIGA_ASSERT(img1.dim() == 5);
        SAIGA_ASSERT(img2.dim() == 5);
        auto kernel = kernel_raw.repeat({img2.size(1), 1, 1, 1, 1});

        auto mu1 = torch::conv3d(img1, kernel, {}, 1, padding, 1, img1.size(1));
        auto mu2 = torch::conv3d(img2, kernel, {}, 1, padding, 1, img2.size(1));

        auto mu11 = mu1 * mu1;
        auto mu22 = mu2 * mu2;
        auto mu12 = mu1 * mu2;

        auto sigma11 = torch::conv3d(img1 * img1, kernel, {}, 1, padding, 1, img1.size(1)) - mu11;
        auto sigma22 = torch::conv3d(img2 * img2, kernel, {}, 1, padding, 1, img1.size(1)) - mu22;
        auto sigma12 = torch::conv3d(img1 * img2, kernel, {}, 1, padding, 1, img1.size(1)) - mu12;

        auto ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu11 + mu22 + C1) * (sigma11 + sigma22 + C2));

        return ssim_map.mean();
    }

   private:
    torch::Tensor kernel_raw;
    int padding;

    float C1, C2;
};
TORCH_MODULE(SSIM3D);

// Caffe Version for lpips similarity score using a traced python module.
//   1. Use the python code below to create the traced module
//   2. Load this traced module into this class and use it in C++ :)
//
// Python Tracing:
//
//     import lpips
//     loss_fn_alex = lpips.LPIPS(net='alex')
//     example = torch.rand(1, 3, 224, 224)
//     traced_script_module = torch.jit.trace(loss_fn_alex, (example, example))
//     traced_script_module.save("traced_lpips.pt")
//
// C++ Usage:
//
//     LPIPS lpips("traced_lpips.pt");
//     lpips.module.eval();
//     auto loss = lpips.forward(output, target);
//
//
//  Source: https://github.com/richzhang/PerceptualSimilarity
//
class LPIPS
{
   public:
    LPIPS(const std::string& file) { module = torch::jit::load(file); }

    torch::Tensor forward(torch::Tensor input, torch::Tensor target)
    {
        SAIGA_ASSERT(input.dim() == 4);
        SAIGA_ASSERT(target.dim() == 4);

        // LPIPS expects the input to be in the range [-1, 1]
        input  = input * 2 - 1;
        target = target * 2 - 1;

        module.to(input.device());
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        inputs.push_back(target);
        return module.forward(inputs).toTensor();
    }

    torch::jit::script::Module module;
};


}  // namespace Saiga
